import cv2
import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from spatial_correlation_sampler import SpatialCorrelationSampler
from model.utils.ResNet2d import ResNet2d, Bottleneck, BasicBlock
from model.image_segmentation.UNet import SpatialDecoder, CBAMLayer2d
from utils.loss_functions import DiceLoss
from utils.data_augmentation import LED
from utils.assess import calculate_kappa, calculate_iou, calculate_accuracy, calculate_precision, \
    calculate_recall, calculate_f1_score, calculate_false_alarm, calculate_miss_rate


class Matching_layer_scs(nn.Module):
    def __init__(self, ks, patch, stride, pad, patch_dilation):
        super(Matching_layer_scs, self).__init__()
        self.relu = nn.ReLU()
        self.patch = patch
        self.correlation_sampler = SpatialCorrelationSampler(ks, patch, stride, pad, patch_dilation)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        corr = self.correlation_sampler(feature1, feature2)
        corr = corr.view(b, self.patch * self.patch, h1 * w1)  # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr


class Flow_refinement(nn.Module):
    def __init__(self, num_segments, expansion=1, pos=2):
        super(Flow_refinement, self).__init__()
        self.num_segments = num_segments
        self.expansion = expansion
        self.pos = pos
        self.out_channel = 64 * (2 ** (self.pos - 1)) * self.expansion
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, self.c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(),
            nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False),
            nn.BatchNorm2d(self.c3),
            nn.ReLU(),
            nn.Conv2d(self.c3, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res, match_v):
        if match_v is not None:
            x = torch.cat([x, match_v], dim=1)
        _, c, h, w = x.size()
        x = x.view(-1, self.num_segments - 1, c, h, w)

        x = torch.cat([x, x[:, -1:, :, :, :]], dim=1)  # (b,t,3,h,w)
        x = x.view(-1, c, h, w)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + res

        return x


class MotionSqueeze(nn.Module):
    def __init__(self, block, num_segments):
        super(MotionSqueeze, self).__init__()
        self.patch = 15
        self.patch_dilation = 1
        self.num_segments = num_segments
        self.matching_layer = Matching_layer_scs(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)

        self.flow_refinement = Flow_refinement(num_segments=num_segments, expansion=block.expansion, pos=2)
        self.soft_argmax = nn.Softmax(dim=1)

        self.chnl_reduction = nn.Sequential(
            nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def apply_gaussian_kernel(self, corr, h, w, p, sigma=5):
        b, c, s = corr.size()

        x = torch.arange(p, dtype=torch.float).to(corr.device).detach()
        y = torch.arange(p, dtype=torch.float).to(corr.device).detach()

        idx = corr.max(dim=1)[1]  # b x hw    get maximum value along channel
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()

        x = x.view(1, 1, p, 1, 1).expand(1, 1, p, h, w).to(corr.device).detach()
        y = y.view(1, p, 1, 1, 1).expand(1, p, 1, h, w).to(corr.device).detach()

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, p * p, h * w)  # .permute(0,2,1).contiguous()

        return gauss_kernel * corr

    def match_to_flow_soft(self, match, k, h, w, temperature=1, mode='softmax'):
        b, c, s = match.size()
        # idx = torch.arange(h * w, dtype=torch.float32).to(match.device)
        # idx_x = idx % w
        # idx_x = idx_x.repeat(b, k, 1).to(match.device)
        # idx_y = torch.floor(idx / w)
        # idx_y = idx_y.repeat(b, k, 1).to(match.device)

        # soft_idx_x = idx_x[:, :1]
        # soft_idx_y = idx_y[:, :1]
        displacement = (self.patch - 1) / 2

        topk_value, topk_idx = torch.topk(match, k, dim=1)  # (B*T-1, k, H*W)
        topk_value = topk_value.view(-1, k, h, w)

        match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
        match = match * temperature
        match_pre = self.soft_argmax(match)
        smax = match_pre
        smax = smax.view(b, self.patch, self.patch, h, w)
        x_kernel = torch.arange(-displacement * self.patch_dilation, displacement * self.patch_dilation + 1, step=self.patch_dilation, dtype=torch.float).to(match.device)
        y_kernel = torch.arange(-displacement * self.patch_dilation, displacement * self.patch_dilation + 1, step=self.patch_dilation, dtype=torch.float).to(match.device)
        x_mult = x_kernel.expand(b, self.patch).view(b, self.patch, 1, 1)
        y_mult = y_kernel.expand(b, self.patch).view(b, self.patch, 1, 1)

        smax_x = smax.sum(dim=1, keepdim=False)  # (b,w=k,h,w)
        smax_y = smax.sum(dim=2, keepdim=False)  # (b,h=k,h,w)
        flow_x = (smax_x * x_mult).sum(dim=1, keepdim=True).view(-1, 1, h * w)  # (b,1,h,w)
        flow_y = (smax_y * y_mult).sum(dim=1, keepdim=True).view(-1, 1, h * w)  # (b,1,h,w)

        flow_x = (flow_x / (self.patch_dilation * displacement))
        flow_y = (flow_y / (self.patch_dilation * displacement))

        return flow_x, flow_y, topk_value

    def forward(self, x, temperature=100):
        x1 = self.chnl_reduction(x)  # 128 * block.expansion  --> 64

        size = x1.size()
        x1 = x1.view((-1, self.num_segments) + size[1:])  # N T C H W
        x1 = x1.permute(0, 2, 1, 3, 4).contiguous()  # B C T H W

        # match to flow
        k = 1
        temperature = temperature
        b, c, t, h, w = x1.size()
        # t = t - 1

        x1_pre = x1[:, :, :-1].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x1_post = x1[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

        match = self.matching_layer(x1_pre, x1_post)  # (B*T-1, patch*patch, H*W)
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature)
        flow = torch.cat([u, v], dim=1).view(-1, 2 * k, h, w)  # (b, 2, h, w)

        x = self.flow_refinement(flow, x, confidence)

        return x


class MSNet(LightningModule):
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

        self.conv = nn.Conv2d(self.input_bands, 32, 3, 1, 1, bias=False)

        resnet18_2d = ResNet2d(BasicBlock, [2, 2, 2, 2], 32)
        backbone = torch.nn.Sequential(*list(resnet18_2d.children())[:-2])

        self.layer1 = nn.Sequential(*backbone[:3])
        self.layer2 = nn.Sequential(*backbone[3:5])
        self.layer3 = backbone[5]
        self.MotionSqueezeLayer = MotionSqueeze(BasicBlock, self.time_steps)
        # self.CBAMLayer = CBAMLayer2d(512, reduction=4)
        self.layer4 = backbone[6]
        self.layer5 = backbone[7]

        self.spatial_decoder = SpatialDecoder(self.num_classes, self.bilinear)

        # load pretrained model or initialize modules
        self.pretrained: bool = settings.model.pretrained
        if self.pretrained:
            pretrained_parameters = torch.load(settings.model.pretrained_parameters_path)
            _, pretrained_channel, _, _ = pretrained_parameters["conv.weight"].shape
            pretrained_parameters["conv.weight"] = pretrained_parameters["conv.weight"].repeat(1, int(self.input_bands / pretrained_channel), 1, 1)
            if self.input_bands % pretrained_channel > 0:
                idx = int(0 - (self.input_bands % pretrained_channel))
                pretrained_parameters["conv.weight"] = torch.cat([pretrained_parameters["conv.weight"], pretrained_parameters["conv.weight"][:, idx:]], 1)
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

        x = x.transpose(1, 2).reshape((B*T, C, H, W))         # [B*T, C, H, W]
        x0 = self.conv(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.MotionSqueezeLayer(x3)
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
        
        pred_mask = (pred_mask.squeeze(1) > 0.5).detach().cpu().numpy()
        gt_mask = (gt_mask.squeeze(1) > 0).detach().cpu().numpy()
        save_dir = "D:/CodeField/Python/xhy23/MS_Gas_Segmentation/log/MSNet-Gas-Segmentation/2024-11-21_20-07-48"
        for i in range(pred_mask.shape[0]):
            cv2.imwrite(
                f"{save_dir}/{i:03d}.png",
                cv2.merge([
                    np.zeros_like(gt_mask[i, ...], dtype=np.uint8),
                    255 * gt_mask[i, ...].astype(np.uint8),
                    255 * pred_mask[i, ...].astype(np.uint8),
                    (1 + 127 * np.logical_or(gt_mask[i, ...], pred_mask[i, ...])).astype(np.uint8)
                ])
            )

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

        save_dir = "E:/Python/MS_Gas_Segmentation/log/MSNet-Gas-Segmentation/2024-11-21_20-07-48"
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
