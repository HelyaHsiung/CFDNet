import cv2
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from model.utils.FBResNet2d import FBResNet, BasicBlockShift
from model.image_segmentation.UNet import SpatialDecoder, CBAMLayer2d
from utils.loss_functions import DiceLoss
from utils.data_augmentation import LED
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, \
    calculate_recall, calculate_f1_score, calculate_false_alarm, calculate_miss_rate


class TDN_Net(LightningModule):
    def __init__(self, settings: DictConfig):
        super().__init__()
        self.lr: float = settings.model.lr
        self.lr_decay: float = settings.model.lr_decay
        self.lr_patience: int = settings.model.lr_patience
        self.optimizer: str = settings.model.optimizer

        self.loss_bce = torch.nn.BCELoss()
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

        self.apha: float = settings.model.apha
        self.belta: float = settings.model.belta

        self.conv = nn.Conv2d(self.input_bands, 32, 3, 1, 1, bias=False)

        fbresnet18_2d = FBResNet(BasicBlockShift, [2, 2, 2, 2], 32, self.time_steps)
        backbone = torch.nn.Sequential(*list(fbresnet18_2d.children())[:-1])

        self.layer1 = nn.Sequential(*backbone[:3])
        self.layer2 = nn.Sequential(*backbone[3:5])
        self.layer3 = backbone[5]
        # self.CBAMLayer = CBAMLayer2d(512, reduction=4)
        self.layer4 = backbone[6]
        self.layer5 = backbone[7]

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1_5 = nn.Sequential(nn.Conv2d(4 * self.input_bands, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 = copy.deepcopy(backbone[4])

        self.spatial_decoder = SpatialDecoder(self.num_classes, self.bilinear)

        # load pretrained model or initialize modules
        self.pretrained: bool = settings.model.pretrained
        if self.pretrained:
            pretrained_parameters = torch.load(settings.model.pretrained_parameters_path)
            _, pretrained_channel, _, _ = pretrained_parameters["conv.weight"].shape
            pretrained_parameters["conv.weight"] = pretrained_parameters["conv.weight"].repeat(1, int(self.input_bands / pretrained_channel), 1, 1)
            pretrained_parameters["conv1_5.0.weight"] = pretrained_parameters["conv1_5.0.weight"].repeat(1, int(self.input_bands / pretrained_channel), 1, 1)
            if self.input_bands % pretrained_channel > 0:
                idx = int(0 - (self.input_bands % pretrained_channel))
                pretrained_parameters["conv.weight"] = torch.cat([pretrained_parameters["conv.weight"], pretrained_parameters["conv.weight"][:, idx:]], 1)
                pretrained_parameters["conv1_5.0.weight"] = torch.cat([pretrained_parameters["conv1_5.0.weight"], pretrained_parameters["conv1_5.0.weight"][:, 4*idx:]], 1)
            else:
                pass
            self.load_state_dict(pretrained_parameters)
        else:
            self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            pass

    def get_x_diff(self, x):
        B, C, T, H, W = x.shape
        x_diff_list = []
        for t in range(2):
            x_diff_list.append(
                self.maxpool_diff(self.conv1_5(self.avg_diff(torch.cat([
                    x[:, :, 1, :, :] - x[:, :, 0, :, :],
                    x[:, :, 2, :, :] - x[:, :, 1, :, :],
                    x[:, :, 3, :, :] - x[:, :, 2, :, :],
                    x[:, :, 4, :, :] - x[:, :, 3, :, :]
                ], 1))))
            )
        for t in range(2, T-2, 1):
            x_diff_list.append(
                self.maxpool_diff(self.conv1_5(self.avg_diff(torch.cat([
                    x[:, :, t-1, :, :] - x[:, :, t-2, :, :],
                    x[:, :, t, :, :] - x[:, :, t-1, :, :],
                    x[:, :, t+1, :, :] - x[:, :, t, :, :],
                    x[:, :, t+2, :, :] - x[:, :, t+1, :, :]
                ], 1))))
            )
        for t in range(2):
            x_diff_list.append(
                self.maxpool_diff(self.conv1_5(self.avg_diff(torch.cat([
                    x[:, :, -4, :, :] - x[:, :, -5, :, :],
                    x[:, :, -3, :, :] - x[:, :, -4, :, :],
                    x[:, :, -2, :, :] - x[:, :, -3, :, :],
                    x[:, :, -1, :, :] - x[:, :, -2, :, :]
                ], 1))))
            )
        x_diff = torch.cat(x_diff_list, 0)
        return x_diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.augmentation:
            with torch.no_grad():
                x = self.data_aug(x)
        else:
            pass
        B, C, T, H, W = x.shape
        x_diff = self.get_x_diff(x)
        x = x.transpose(1, 2).reshape((B * T, C, H, W))  # [B*T, C, H, W]
        x0 = self.conv(x)
        x1 = self.layer1(x0)
        # fusion layer1
        x1 = self.apha * x1 + self.belta * F.interpolate(x_diff, x1.shape[2:])
        x2 = self.layer2(x1)
        # fusion layer2
        x_diff = self.resnext_layer1(x_diff)
        x2 = self.apha * x2 + self.belta * F.interpolate(x_diff, x2.shape[2:])
        x3 = self.layer3(x2)
        # x3 = self.CBAMLayer(x3)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

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

    def shared_step(self, batch, stage):
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

        self.log(f"{stage}_loss", loss, sync_dist=True)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

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

        save_dir = "E:/Python/MS_Gas_Segmentation/log/TDNNet-Gas-Segmentation/2024-11-22_08-39-21"
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
