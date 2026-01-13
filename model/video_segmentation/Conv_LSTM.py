import cv2
import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from model.image_segmentation.UNet import SpatialEncoder, SpatialDecoder
from utils.data_augmentation import LED
from utils.loss_functions import DiceLoss
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, \
    calculate_recall, calculate_f1_score, calculate_false_alarm, calculate_miss_rate


class ConvLSTMCell(nn.Module):
    """
    https://github.com/ndrplz/ConvLSTM_pytorch/tree/master
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(LightningModule):
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

        self.spatial_encoder = SpatialEncoder(self.input_bands)
        self.lstm_cell_list = nn.ModuleList([ConvLSTMCell(512, 64, (3, 3), settings.model.bias) for _ in range(self.time_steps)])
        self.conv = nn.Conv2d(64, 512, 3, 1, 1, bias=False)
        self.spatial_decoder = SpatialDecoder(self.num_classes, self.bilinear)

        # load pretrained model or initialize modules
        self.pretrained: bool = settings.model.pretrained
        if self.pretrained:
            pretrained_parameters = torch.load(settings.model.pretrained_parameters_path)
            _, pretrained_channel, _, _ = pretrained_parameters["spatial_encoder.conv.weight"].shape
            pretrained_parameters["spatial_encoder.conv.weight"] = pretrained_parameters["spatial_encoder.conv.weight"].repeat(1, int(self.input_bands / pretrained_channel), 1, 1)
            if self.input_bands % pretrained_channel > 0:
                idx = int(0 - (self.input_bands % pretrained_channel))
                pretrained_parameters["spatial_encoder.conv.weight"] = torch.cat([pretrained_parameters["spatial_encoder.conv.weight"], pretrained_parameters["spatial_encoder.conv.weight"][:, idx:]], 1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.augmentation:
            with torch.no_grad():
                x = self.data_aug(x)
        else:
            pass
        B, C, T, H, W = x.shape
        h = torch.zeros((B, 64, H // 32, W // 32)).to(self.device)
        c = torch.zeros((B, 64, H // 32, W // 32)).to(self.device)
        y = torch.zeros((B, self.num_classes, T, H, W)).to(self.device)
        for t in range(T):
            x0, x1, x2, x3, x4, x5 = self.spatial_encoder(x[:, :, t, :, :])

            h, c = self.lstm_cell_list[t](x5, (h, c))

            y[:, :, t, :, :] = self.spatial_decoder(x0, x1, x2, x3, x4, self.conv(h))
        y = y.transpose(1, 2).reshape((B * T, self.num_classes, H, W))
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
        video = batch["msi_input"]                      # [B, C, T, H, W]
        assert video.ndim == 5
        B, C, T, H, W = video.shape
        assert H % 32 == 0 and W % 32 == 0

        gt_mask = batch["gt_output"]                    # [B, 1, T, H, W]
        assert gt_mask.ndim == 5
        gt_mask = gt_mask.transpose(1, 2).reshape((B * T, self.num_classes, H, W))

        pred_mask = self.forward(video)                 # [B*T, 1, H, W]

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

        save_dir = "E:/Python/MS_Gas_Segmentation/log/ConvLSTM-Gas-Segmentation/2024-11-21_20-04-32"
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
