import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from collections import OrderedDict
from pytorch_lightning import LightningModule
from model.image_segmentation.UNet import SpatialEncoder, SpatialDecoder
from utils.data_augmentation import LED
from utils.loss_functions import DiceLoss
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, \
    calculate_recall, calculate_f1_score, calculate_false_alarm, calculate_miss_rate


class _ConvBatchNormReLU(nn.Sequential):
    """Convolution Unit"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn", nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.95, affine=True)
        )
        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _PyramidPoolModule(nn.Sequential):
    """Pyramid Pooling Module"""

    def __init__(self, in_channels, pyramids=[6, 3, 2, 1]):
        super(_PyramidPoolModule, self).__init__()
        self.chnl_reduction = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
        out_channels = in_channels // (2 * len(pyramids))
        self.stages = nn.Module()
        for i, p in enumerate(pyramids):
            self.stages.add_module(
                "s{}".format(i),
                nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AdaptiveAvgPool2d(output_size=p)),
                            (
                                "conv",
                                _ConvBatchNormReLU(
                                    in_channels, out_channels, 1, 1, 0, 1
                                ),
                            ),
                        ]
                    )
                ),
            )

    def forward(self, x):
        hs = [self.chnl_reduction(x)]
        height, width = x.size()[2:]
        for stage in self.stages.children():
            h = stage(x)
            h = F.interpolate(h, (height, width), mode="bilinear")
            hs.append(h)
        return torch.cat(hs, dim=1)


class PUNet(LightningModule):
    """Pyramid Scene Parsing Network"""

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

        self.spatial_encoder = SpatialEncoder(self.input_bands)
        self.ppm = _PyramidPoolModule(in_channels=512, pyramids=[6, 3, 2, 1])
        self.spatial_decoder = SpatialDecoder(self.num_classes, self.bilinear)

        # load pretrained model or initialize modules
        self.pretrained: bool = settings.model.pretrained
        if self.pretrained:
            pretrained_parameters = torch.load(settings.model.pretrained_parameters_path)
            _, pretrained_channel, _, _ = pretrained_parameters["spatial_encoder.conv.weight"].shape
            pretrained_parameters["spatial_encoder.conv.weight"] = pretrained_parameters["spatial_encoder.conv.weight"].repeat(1, self.input_bands // pretrained_channel, 1, 1)
            if self.input_bands % pretrained_channel > 0:
                idx = int(0 - (self.input_bands % pretrained_channel))
                pretrained_parameters["spatial_encoder.conv.weight"] = torch.cat([pretrained_parameters["spatial_encoder.conv.weight"], pretrained_parameters["spatial_encoder.conv.weight"][:, idx:]],
                                                                                 1)
            else:
                pass
            self.load_state_dict(pretrained_parameters, False)
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

    def forward(self, x):
        if self.augmentation:
            with torch.no_grad():
                x = self.data_aug(x)
        else:
            pass
        x0, x1, x2, x3, x4, x5 = self.spatial_encoder(x)
        x5 = self.ppm(x5)
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
        image = batch["msi_input"]
        assert image.ndim == 4                      # [B, C, H, W]
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt_mask = batch["gt_output"]
        assert gt_mask.ndim == 4                    # [B, 1, H, W]

        pred_mask = self.forward(image)             # [B, 1, H, W]

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
        image = batch["msi_input"]
        gt_mask = batch["gt_output"]
        pred_mask = self.forward(image)  # [B, 1, H, W]

        loss_bce = self.loss_bce(pred_mask, gt_mask)
        loss_dice = self.loss_dice(pred_mask, gt_mask)
        loss = loss_bce + loss_dice

        pred_mask = (pred_mask.squeeze(1) > 0.5).detach().cpu().numpy()
        gt_mask = (gt_mask.squeeze(1) > 0).detach().cpu().numpy()

        save_dir = "E:/Python/MS_Gas_Segmentation/log/PUNet-Gas-Segmentation/2024-11-21_19-59-51"
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
