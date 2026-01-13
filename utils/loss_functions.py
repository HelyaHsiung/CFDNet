import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
"""
DiceLoss was edited by Dr. Liu Yinhe
"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        if multiclass:
            loss = 1 - self.multiclass_dice_coeff(pred, target, reduce_batch_first=True)
        else:
            loss = 1 - self.dice_coeff(pred, target, reduce_batch_first=False)
        return loss

    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target, smooth=1e-6):
        # Flatten the tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection

        # Compute IoU
        IoU = (intersection + smooth) / (union + smooth)

        # Compute IoU loss
        loss = 1 - IoU

        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel filter kernels
        self.maxpool_layer = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, pred, target):
        assert pred.ndim == target.ndim
        assert pred.shape[1] == 1
        if pred.ndim == 4:
            pass  # [B, 1, H, W] -> [B, 1, H, W]
        elif pred.ndim == 5:
            pred = pred.transpose(1, 2).flatten(0, 1)  # [B, 1, T, H, W] -> [B, T, 1, H, W] -> [B*T, 1, H, W]
            target = target.transpose(1, 2).flatten(0, 1)  # [B, 1, T, H, W] -> [B, T, 1, H, W] -> [B*T, 1, H, W]
        else:
            raise Exception("Need for prediction or target with dimensions being 4 or 5.")
        # Ensure the predictions and targets are binary
        pred = (pred > 0.5).bool()
        target = (target > 0.5).bool()

        # Apply morph trophy calculation to get the boundaries
        pred_boundary = self.gen_edge(pred)         # [B*T, C, H, W]
        target_boundary = self.gen_edge(target)     # [B*T, C, H, W]
        target_distance = self.gen_dist(target_boundary.detach().cpu().numpy()).to(target_boundary.device)

        # Calculate the Binary Cross Entropy loss between the boundaries
        multiplied = torch.einsum("bcwh, bcwh->bcwh", pred_boundary, target_distance).float()
        edge_loss = multiplied.mean()

        return edge_loss

    def gen_edge(self, bool_map):
        eroded_map = torch.logical_and(bool_map, (self.maxpool_layer((~bool_map).float()).bool()))
        return eroded_map.int()

    def gen_dist(self, edge_map):
        N, _, _, _ = edge_map.shape
        dist = np.zeros_like(edge_map, dtype="<f4")
        for i in range(N):
            dist[i, 0, :, :] = distance(1 - edge_map[i, 0, :, :]) * (1 - edge_map[i, 0, :, :])
        return torch.tensor(dist).float()


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, pred, target, mu, logvar):
        # 计算重构损失（使用MSE）
        recon_loss = nn.functional.mse_loss(pred, target, reduction='mean')
        
        # 计算KL散度损失
        kl_loss = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + kl_loss
        return total_loss
    