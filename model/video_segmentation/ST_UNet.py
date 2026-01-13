import cv2
import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from model.image_segmentation.UNet import SpatialDecoder
from model.utils.ResNet3d import ResNet3d, Bottleneck, BasicBlock
from utils.loss_functions import DiceLoss
from utils.data_augmentation import LED
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, \
    calculate_recall, calculate_f1_score, calculate_false_alarm, calculate_miss_rate


class CBAMLayer3d(nn.Module):
    def __init__(self, channel, time_steps, reduction=16, spatial_kernel=7):
        super(CBAMLayer3d, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool3d((time_steps, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((time_steps, 1, 1))

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x, spatial_out


class S3DSpatialEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, 32, 3, 1, 1, bias=False)

        resnet18_3d = ResNet3d(BasicBlock, [2, 2, 2, 2], 32)

        backbone = torch.nn.Sequential(*list(resnet18_3d.children())[:-2])

        self.layer1 = nn.Sequential(*backbone[:5])
        self.layer2 = nn.Sequential(*backbone[5:7])
        self.layer3 = backbone[7]
        # self.CBAMLayer = CBAMLayer3d(512, time_steps, reduction=4)
        self.layer4 = backbone[8]
        self.layer5 = backbone[9]

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x3 = self.CBAMLayer(x3)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x0, x1, x2, x3, x4, x5


class ST_UNet(LightningModule):
    def __init__(self, settings: DictConfig):
        super().__init__()
        self.lr: float = settings.model.lr
        self.lr_decay: float = settings.model.lr_decay
        self.lr_patience: int = settings.model.lr_patience
        self.optimizer: str = settings.model.optimizer

        self.loss_bce = nn.BCELoss()
        self.loss_dice = DiceLoss()

        self.augmentation: bool = settings.model.augmentation
        if self.augmentation:
            self.data_aug = LED(settings.model.input_bands, 7, 64)
            self.input_bands: int = int(1 + settings.model.input_bands * (settings.model.input_bands - 1) / 2)
        else:
            self.input_bands: int = settings.model.input_bands
        self.bilinear: bool = settings.model.bilinear
        self.num_classes: int = settings.model.num_classes
        self.time_steps: int = settings.model.time_steps

        self.s3d_spatial_encoder = S3DSpatialEncoder(self.input_bands)

        self.spatial_decoder = SpatialDecoder(self.num_classes, self.bilinear)

        # load pretrained model or initialize modules
        self.pretrained: bool = settings.model.pretrained
        if self.pretrained:
            pretrained_parameters = torch.load(settings.model.pretrained_parameters_path)
            _, pretrained_channel, _, _, _ = pretrained_parameters["s3d_spatial_encoder.conv.weight"].shape
            pretrained_parameters["s3d_spatial_encoder.conv.weight"] = pretrained_parameters["s3d_spatial_encoder.conv.weight"].repeat(1, int(self.input_bands / pretrained_channel), 1, 1, 1)
            if self.input_bands % pretrained_channel > 0:
                idx = int(0 - (self.input_bands % pretrained_channel))
                pretrained_parameters["s3d_spatial_encoder.conv.weight"] = torch.cat([pretrained_parameters["s3d_spatial_encoder.conv.weight"], pretrained_parameters["s3d_spatial_encoder.conv.weight"][:, idx:]], 1)
            else:
                pass
            self.load_state_dict(pretrained_parameters)
        else:
            self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            pass

    def forward(self, x: torch.Tensor):
        if self.augmentation:
            with torch.no_grad():
                x = self.data_aug(x)
        else:
            pass
        B, C, T, H, W = x.shape
        x0, x1, x2, x3, x4, x5 = self.s3d_spatial_encoder(x)

        x0 = x0.transpose(1, 2).reshape((B * T, x0.size(1), x0.size(3), x0.size(4)))
        x1 = x1.transpose(1, 2).reshape((B * T, x1.size(1), x1.size(3), x1.size(4)))
        x2 = x2.transpose(1, 2).reshape((B * T, x2.size(1), x2.size(3), x2.size(4)))
        x3 = x3.transpose(1, 2).reshape((B * T, x3.size(1), x3.size(3), x3.size(4)))
        x4 = x4.transpose(1, 2).reshape((B * T, x4.size(1), x4.size(3), x4.size(4)))
        x5 = x5.transpose(1, 2).reshape((B * T, x5.size(1), x5.size(3), x5.size(4)))

        y = self.spatial_decoder(x0, x1, x2, x3, x4, x5)
        return torch.sigmoid(y)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        else:
            raise Exception(f"No optimizer implemented!")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=self.lr_decay,
                                                               patience=self.lr_patience)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        video = batch["msi_input"]  # [B, C, T, H, W]
        assert video.ndim == 5
        B, C, T, H, W = video.shape
        assert H % 32 == 0 and W % 32 == 0

        gt_mask = batch["gt_output"]  # [B, 1, T, H, W]
        assert gt_mask.ndim == 5
        gt_mask = gt_mask.transpose(1, 2).reshape((B * T, self.num_classes, H, W))

        pred_mask = self.forward(video)  # [B*T, 1, H, W]

        loss_bce = self.loss_bce(pred_mask, gt_mask)
        loss_dice = self.loss_dice(pred_mask, gt_mask)
        loss = loss_bce + loss_dice

        self.log(f"train_loss", loss, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        video = batch["msi_input"]  # [B, C, T, H, W]
        assert video.ndim == 5
        B, C, T, H, W = video.shape
        assert H % 32 == 0 and W % 32 == 0

        gt_mask = batch["gt_output"]  # [B, 1, T, H, W]
        assert gt_mask.ndim == 5
        gt_mask = gt_mask.transpose(1, 2).reshape((B * T, self.num_classes, H, W))

        pred_mask = self.forward(video)  # [B*T, 1, H, W]

        loss_bce = self.loss_bce(pred_mask, gt_mask)
        loss_dice = self.loss_dice(pred_mask, gt_mask)
        loss = loss_bce + loss_dice

        self.log(f"val_loss", loss, sync_dist=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        video = batch["msi_input"]  # [B, C, T, H, W]
        B, C, T, H, W = video.shape
        gt_mask = batch["gt_output"]  # [B, 1, T, H, W]
        gt_mask = gt_mask.transpose(1, 2).reshape((B * T, self.num_classes, H, W))

        pred_mask = self.forward(video)  # [B*T, 1, H, W]

        loss_bce = self.loss_bce(pred_mask, gt_mask)
        loss_dice = self.loss_dice(pred_mask, gt_mask)
        loss = loss_bce + loss_dice

        pred_mask = (pred_mask.squeeze(1) > 0.5).detach().cpu().numpy()
        gt_mask = (gt_mask.squeeze(1) > 0).detach().cpu().numpy()

        save_dir = "E:/Python/MS_Gas_Segmentation/log/STUNet-Gas-Segmentation/2024-11-03_08-51-37"
        for i in range(pred_mask.shape[0]):
            cv2.imwrite(
                f"{save_dir}/{batch_idx:03d}_{i:03d}.png",
                cv2.merge([
                    np.zeros_like(gt_mask[i, ...], dtype=np.uint8),
                    255 * gt_mask[i, ...].astype(np.uint8),
                    255 * pred_mask[i, ...].astype(np.uint8),
                    (1 + 127 * np.logical_or(gt_mask[i, ...], pred_mask[i, ...])).astype(np.uint8)
                ])
            )

        kappa = torch.tensor([calculate_kappa(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        iou = torch.tensor([calculate_iou(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        accuracy = torch.tensor([calculate_accuracy(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        precision = torch.tensor([calculate_precision(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        recall = torch.tensor([calculate_recall(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        f1_score = torch.tensor([calculate_f1_score(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        false_alarm = torch.tensor([calculate_false_alarm(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()
        miss_rate = torch.tensor([calculate_miss_rate(pred_mask[i, ...], gt_mask[i, ...]) for i in range(pred_mask.shape[0])]).float().mean()

        self.log("test_loss", loss, sync_dist=True)
        self.log("kappa", kappa, sync_dist=True)
        self.log("iou", iou, sync_dist=True)
        self.log("accuracy", accuracy, sync_dist=True)
        self.log("precision", precision, sync_dist=True)
        self.log("recall", recall, sync_dist=True)
        self.log("f1_score", f1_score, sync_dist=True)
        self.log("false_alarm", false_alarm, sync_dist=True)
        self.log("miss_rate", miss_rate, sync_dist=True)

        return {"loss": loss, "kappa": kappa, "iou": iou, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "false_alarm": false_alarm, "miss_rate": miss_rate}
