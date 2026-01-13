import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from collections import OrderedDict
from torchvision.ops import masks_to_boxes
from pytorch_lightning import LightningModule
from torchvision.models.detection import MaskRCNN
from model.image_segmentation.UNet import SpatialEncoder
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from utils.data_augmentation import LED
from utils.loss_functions import DiceLoss
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, \
    calculate_recall, calculate_f1_score, calculate_false_alarm, calculate_miss_rate


class MASK_RCNN(LightningModule):
    """Mask Region Convolution Neural Network"""

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
            image_mean = [0.5401, 0.3647, 0.3846, 0.3914, 0.5906, 0.3738, 0.4037, 0.4467, 0.6249, 0.4347, 0.5287, 0.7313, 0.4729, 0.6697, 0.6241, 0.3523]
            image_std = [0.1838, 0.1089, 0.0360, 0.0612, 0.1562, 0.1695, 0.1895, 0.2087, 0.1304, 0.1962, 0.2458, 0.1255, 0.1014, 0.1601, 0.1818, 0.0963]
        else:
            self.input_bands: int = settings.model.input_bands
            image_mean = [8594.4326, 8805.7236, 8735.1416, 8608.8760, 8339.1953, 8733.9590]
            image_std = [437.5285, 175.5317, 313.9593, 434.3400, 498.1225, 201.4868]

        self.spatial_encoder = SpatialEncoder(self.input_bands)
        # load pretrained model or initialize modules
        self.pretrained: bool = settings.model.pretrained
        if self.pretrained:
            pretrained_parameters = torch.load(settings.model.pretrained_parameters_path)
            _, pretrained_channel, _, _ = pretrained_parameters["spatial_encoder.conv.weight"].shape
            pretrained_parameters["spatial_encoder.conv.weight"] = pretrained_parameters["spatial_encoder.conv.weight"].repeat(1, self.input_bands // pretrained_channel, 1, 1)
            if self.input_bands % pretrained_channel > 0:
                idx = int(0 - (self.input_bands % pretrained_channel))
                pretrained_parameters["spatial_encoder.conv.weight"] = torch.cat([pretrained_parameters["spatial_encoder.conv.weight"], pretrained_parameters["spatial_encoder.conv.weight"][:, idx:]], 1)
            else:
                pass
            self.load_state_dict(pretrained_parameters, False)
        else:
            self.apply(self.initialize_weights)
        backbone_with_fpn = BackboneWithFPN(self.spatial_encoder,
                                            return_layers={'layer2': '0', 'layer3': '1', 'layer4': '2', 'layer5': '3'},
                                            in_channels_list=[64, 128, 256, 512],
                                            out_channels=256)
        self.num_classes = settings.model.num_classes
        self.box_detections_per_img = settings.model.box_detections_per_img
        self.net = MaskRCNN(
            backbone=backbone_with_fpn,
            num_classes=self.num_classes,  # 类别数
            min_size=480,  # 最小输入尺寸
            max_size=640,  # 最大输入尺寸
            image_mean=image_mean,
            image_std=image_std,
            box_detections_per_img=self.box_detections_per_img,   # 每张图片最多探测框数量
        )
        del self.spatial_encoder

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

    def create_targets(self, y):
        B, _, H, W = y.shape
        targets = []
        for i in range(B):
            masks = y[i]  # [1, H, W]
            if masks.max() > 0:
                boxes = masks_to_boxes(masks)  # [N, 4]
                labels = torch.ones(len(boxes), dtype=torch.int64).to(self.device)
                scores = torch.ones(len(boxes), dtype=torch.float32).to(self.device)
                target = {
                    "boxes": boxes,  # [N, 4]
                    "labels": labels,  # [N]
                    "scores": scores,  # [N]
                    "masks": masks,  # [N, H, W], be careful that the shape of results is [N, 1, H, W] while shape of targets is [N, H, W]
                }
            else:
                target = {
                    "boxes": torch.empty((0, 4), dtype=torch.float32).to(self.device),
                    "labels": torch.tensor([], dtype=torch.int64).to(self.device),
                    "scores": torch.tensor([], dtype=torch.float32).to(self.device),
                    "masks": torch.empty((0, H, W), dtype=torch.float32).to(self.device)
                }
            targets.append(target)
        return targets

    def boxes_to_masks(self, detections):
        B = len(detections)
        H, W = detections[0]['masks'].shape[2:]
        pred_mask = torch.zeros((B, 1, H, W), dtype=torch.float32).to(self.device)
        for i in range(B):
            for n in range(len(detections[i]['labels'])):
                if detections[i]['labels'][n].item() == 1:
                    pred_mask[i, :, :, :] = pred_mask[i, :, :, :] + detections[i]['masks'][n, :, :, :]
        return pred_mask.clamp(min=0.0, max=1.0)

    def forward(self, x, y=None):
        if self.augmentation:
            with torch.no_grad():
                x = self.data_aug(x)
        else:
            pass
        if self.training:
            assert y, "y should not be None in training stage."
            losses = self.net(x, self.create_targets(y))
            return losses
        else:
            detections = self.net(x)
            return self.boxes_to_masks(detections)

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
        image = batch["msi_input"]
        assert image.ndim == 4  # [B, C, H, W]
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt_mask = batch["gt_output"]
        assert gt_mask.ndim == 4  # [B, 1, H, W]

        losses = self.forward(image, gt_mask)

        loss = sum(loss for loss in losses.values())  # 计算所有损失项的和

        self.log("train_loss", loss, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image = batch["msi_input"]
        assert image.ndim == 4  # [B, C, H, W]
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt_mask = batch["gt_output"]
        assert gt_mask.ndim == 4  # [B, 1, H, W]

        pred_mask = self.forward(image, gt_mask)

        loss_bce = self.loss_bce(pred_mask, gt_mask)
        loss_dice = self.loss_dice(pred_mask, gt_mask)
        loss = loss_bce + loss_dice

        self.log(f"val_loss", loss, sync_dist=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        image = batch["msi_input"]
        gt_mask = batch["gt_output"]
        pred_mask = self.forward(image, gt_mask)

        loss_bce = self.loss_bce(pred_mask, gt_mask)
        loss_dice = self.loss_dice(pred_mask, gt_mask)
        loss = loss_bce + loss_dice

        pred_mask = (pred_mask.squeeze(1) > 0.5).detach().cpu().numpy()
        gt_mask = (gt_mask.squeeze(1) > 0).detach().cpu().numpy()

        save_dir = "E:/Python/MS_Gas_Segmentation/log/MASKRCNN-Gas-Segmentation/2024-12-25_23-28-28"
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
