import cv2
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from model.image_segmentation.UNet import Up
from model.utils.FBResNet2d import FBResNet, BasicBlockShift
from utils.data_augmentation import LED
from utils.loss_functions import DiceLoss
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score


class TemporalSaliencyDiffusion(nn.Module):
    def __init__(self, in_ch, n_segments):
        super().__init__()
        self.reduction = 16
        self.n_segments = n_segments
        self.delta_t = 1

        self.chnl_reduction = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // self.reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch // self.reduction)
        )

        self.get_diffusion = nn.Sequential(
            nn.Conv2d(2 * in_ch // self.reduction, in_ch // self.reduction, 3, 1, 1),
            nn.BatchNorm2d(in_ch // self.reduction),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_ch // self.reduction, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.get_velocity = nn.Sequential(
            nn.Conv2d(2 * in_ch // self.reduction, in_ch // self.reduction, 3, 1, 1),
            nn.BatchNorm2d(in_ch // self.reduction),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_ch // self.reduction, 2, 3, 1, 1),
            nn.Sigmoid()
        )

        self.get_saliency = nn.Sequential(
            nn.Conv2d(in_ch // self.reduction, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.register_buffer(
            'laplacian_kernel',
            torch.nn.Parameter(
                torch.tensor([[0., 1., 0.],
                              [1., -4., 1.],
                              [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                requires_grad=False
            )
        )
        self.register_buffer('X', None)
        self.register_buffer('Y', None)

    def diffuse(self, d, diffusion):
        d = d + diffusion * self.delta_t * F.conv2d(d, self.laplacian_kernel, padding=1)
        return d

    def advect(self, d, u, v, H, W):
        if self.X is None or self.Y is None:
            Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            self.X = X.unsqueeze(0).unsqueeze(0).to(d.device)  # [1, 1, H, W]
            self.Y = Y.unsqueeze(0).unsqueeze(0).to(d.device)  # [1, 1, H, W]
            self.X.requires_grad = False
            self.Y.requires_grad = False
            del X, Y
        else:
            pass

        x = (self.X - self.delta_t * W * u).clamp(0, W - 1) * 2.0 / (W - 1) - 1.0
        y = (self.Y - self.delta_t * H * v).clamp(0, H - 1) * 2.0 / (H - 1) - 1.0

        grid = torch.stack((x.squeeze(1), y.squeeze(1)), dim=-1)  # [7, H, W, 2]
        d = F.grid_sample(d, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return d

    def forward(self, x):
        x = self.chnl_reduction(x)                                                      # [BT, c, H, W]
        BT, c, H, W = x.shape
        x1 = x.reshape(-1, self.n_segments, c, H, W)                                    # [B, T, c, H, W]
        x1_pre_cur_cat = torch.cat([x1[:, :-1], x1[:, 1:]], 2).reshape(-1, 2*c, H, W)   # [B(T-1), 2c, H, W]
        diffusion = self.get_diffusion(x1_pre_cur_cat) * 0.1                            # [B(T-1), 1, H, W]     [0, 0.1]
        velocity = 0.5 * self.get_velocity(x1_pre_cur_cat) - 0.25                       # [B(T-1), 2, H, W]     [-0.25, +0.25]
        u, v = torch.split(velocity, 1, 1)                      # [B(T-1), 1, H, W]

        saliency = self.get_saliency(x)                                                 # [BT, 1, H, W]
        saliency = saliency.reshape(-1, self.n_segments, 1, H, W)                       # [B, T, 1, H, W]
        saliency_pre = saliency[:, :-1, :, :, :].reshape(-1, 1, H, W)                   # [B(T-1), 1, H, W]

        saliency_pas = self.diffuse(saliency_pre, diffusion)                            # [B(T-1), 1, H, W]
        saliency_pas = self.advect(saliency_pas, u, v, H, W)                            # [B(T-1), 1, H, W]

        saliency = (torch.cat([saliency[:, 0:1, :, :, :], saliency_pas.reshape((-1, self.n_segments - 1, 1, H, W))], 1)).clamp(0.0, 1.0)
        saliency = saliency.reshape(-1, 1, H, W)                                        # [BT, 1, H, W]
        return saliency


class TSD_Net(LightningModule):
    """
    Temporal Saliency Diffusion Model for Multi-spectral Infrared Video Gas Segmentation
    """
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
        self.layer4 = backbone[6]
        self.layer5 = backbone[7]

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1_5 = nn.Sequential(nn.Conv2d(4 * self.input_bands, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 = copy.deepcopy(backbone[4])

        self.layer6 = Up(512, 256, False)
        self.saliency_1 = TemporalSaliencyDiffusion(in_ch=256, n_segments=self.time_steps)
        self.layer7 = Up(256, 128, False)
        self.saliency_2 = TemporalSaliencyDiffusion(in_ch=128, n_segments=self.time_steps)
        self.dropout = nn.Dropout(p=0.25)
        self.layer8 = Up(128, 64, False)
        self.saliency_3 = TemporalSaliencyDiffusion(in_ch=64, n_segments=self.time_steps)
        self.layer9 = Up(64, 64, False)
        self.saliency_4 = TemporalSaliencyDiffusion(in_ch=64, n_segments=self.time_steps)
        self.layer10 = Up(64, 32, False)
        self.out = nn.Conv2d(32, self.num_classes, kernel_size=1, bias=False)

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
                pretrained_parameters["conv1_5.0.weight"] = torch.cat([pretrained_parameters["conv1_5.0.weight"], pretrained_parameters["conv1_5.0.weight"][:, 4 * idx:]], 1)
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
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.1)
            nn.init.constant_(m.bias, 0.0)
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

    def forward(self, x: torch.Tensor):
        if self.augmentation:
            with torch.no_grad():
                x = self.data_aug(x)
        else:
            pass
        B, C, T, H, W = x.shape
        x_diff = self.get_x_diff(x)
        x = x.transpose(1, 2).reshape((B * T, C, H, W))                         # [B*T, C, H, W]
        x0 = self.conv(x)
        x1 = self.layer1(x0)
        # fusion layer1
        x1 = self.apha * x1 + self.belta * F.interpolate(x_diff, x1.shape[2:])

        x2 = self.layer2(x1)
        # fusion layer2
        x_diff = self.resnext_layer1(x_diff)

        x2 = self.apha * x2 + self.belta * F.interpolate(x_diff, x2.shape[2:])
        x3 = self.layer3(x2)                                                    # [B*T, 128, H/8, W/8]
        x4 = self.layer4(x3)                                                    # [B*T, 256, H/16, W/16]
        x5 = self.layer5(x4)                                                    # [B*T, 512, H/32, W/32]

        y4 = self.layer6(x5, x4)                                                # [B*T, 256, H/16, W/16]
        y_sal_1 = self.saliency_1(y4)
        y4 = y4 + y4 * y_sal_1
        y3 = self.layer7(y4, x3)                                                # [B*T, 128, H/8, W/8]
        y_sal_2 = self.saliency_2(y3)
        y3 = y3 + y3 * y_sal_2
        y3 = self.dropout(y3)
        y2 = self.layer8(y3, x2)                                                # [B*T, 64, H/4, W/4]
        y_sal_3 = self.saliency_3(y2)
        y2 = y2 + y2 * y_sal_3
        y1 = self.layer9(y2, x1)                                                # [B*T, 64, H/2, W/2]
        y_sal_4 = self.saliency_4(y1)                                             # [B*T, 1, H/2, W/2]
        y1 = y1 + y1 * y_sal_4                                                  # [B*T, 64, H/2, W/2]
        y0 = self.layer10(y1, x0)                                               # [B*T, 32, H, W]
        y = torch.sigmoid(self.out(y0))                                         # [B*T, 1, H, W]
        return y, y_sal_4, y_sal_3, y_sal_2, y_sal_1

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

        pred_mask, pred_sal_4, pred_sal_3, pred_sal_2, pred_sal_1 = self.forward(video)  # [B*T, 1, H, W]
        # pred_mask = self.forward(video)

        loss_1 = self.loss_bce(pred_mask, gt_mask) + self.loss_dice(pred_mask, gt_mask)
        loss_2 = self.loss_bce(pred_sal_4, gt_mask[:, :, ::2, ::2]) + self.loss_dice(pred_sal_4, gt_mask[:, :, ::2, ::2])
        loss_3 = self.loss_bce(pred_sal_3, gt_mask[:, :, ::4, ::4]) + self.loss_dice(pred_sal_3, gt_mask[:, :, ::4, ::4])
        loss_4 = self.loss_bce(pred_sal_2, gt_mask[:, :, ::8, ::8]) + self.loss_dice(pred_sal_2, gt_mask[:, :, ::8, ::8])
        loss_5 = self.loss_bce(pred_sal_1, gt_mask[:, :, ::16, ::16]) + self.loss_dice(pred_sal_1, gt_mask[:, :, ::16, ::16])
        loss = 0.2 * loss_1 + 0.2 * loss_2 + 0.2 * loss_3 + 0.2 * loss_4 + 0.2 * loss_5
        # loss = loss_1

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

        pred_mask, _, _, _, _ = self.forward(video)  # [B*T, 1, H, W]
        # pred_mask = self.forward(video)

        loss = self.loss_bce(pred_mask, gt_mask) + self.loss_dice(pred_mask, gt_mask)

        self.log(f"val_loss", loss, sync_dist=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        video = batch["msi_input"]  # [B, C, T, H, W]
        B, C, T, H, W = video.shape
        gt_mask = batch["gt_output"]  # [B, 1, T, H, W]
        gt_mask = gt_mask.transpose(1, 2).reshape((B * T, self.num_classes, H, W))

        pred_mask, _, _, _, _ = self.forward(video)  # [B*T, 1, H, W]
        # pred_mask = self.forward(video)

        loss = self.loss_bce(pred_mask, gt_mask) + self.loss_dice(pred_mask, gt_mask)

        pred_mask = (pred_mask.squeeze(1) > 0.5).detach().cpu().numpy()
        gt_mask = (gt_mask.squeeze(1) > 0).detach().cpu().numpy()

        save_dir = f"E:/Python/MS_Gas_Segmentation/log/TSDNet-Gas-Segmentation/2024-12-04_13-26-27/infer/"
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

        self.log("test_loss", loss, sync_dist=True)
        self.log("kappa", kappa, sync_dist=True)
        self.log("iou", iou, sync_dist=True)
        self.log("accuracy", accuracy, sync_dist=True)
        self.log("precision", precision, sync_dist=True)
        self.log("recall", recall, sync_dist=True)
        self.log("f1_score", f1_score, sync_dist=True)

        return {"loss": loss, "kappa": kappa, "iou": iou, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
