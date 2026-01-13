import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Type, Union

"""
Helya Hsiung, haiyangxiong@whu.edu.cn, 2024.12.29
http://rsidea.whu.edu.cn/
https://github.com/HelyaHsiung/CFDNet
"""


class FDModule(nn.Module):
    """
    Fluid Dynamics Module
    """

    def __init__(self, in_ch, n_segment=8, reduction=16):
        super(FDModule, self).__init__()
        self.delta_t = 1
        self.n_segments: int = n_segment
        self.reduction: int = reduction

        self.chnl_reduction = nn.Conv2d(in_ch, in_ch // self.reduction, kernel_size=1, stride=1, padding=0, bias=False) if self.reduction > 1 else None

        self.get_diffusion = nn.Sequential(
            nn.Conv2d(2 * in_ch // self.reduction, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.get_velocity = nn.Sequential(
            nn.Conv2d(2 * in_ch // self.reduction, 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

        self.register_buffer('laplacian_kernel', torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).detach())
        self.register_buffer('X', None)
        self.register_buffer('Y', None)

        self.chnl_restore = nn.Conv2d(in_ch // self.reduction, in_ch, kernel_size=1, stride=1, padding=0, bias=False) if self.reduction > 1 else None

        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def diffuse(self, d, diffusion):
        d = d + diffusion * self.delta_t * F.conv2d(d, self.laplacian_kernel.expand(d.shape[1], d.shape[1], -1, -1), padding=1)
        return d

    def advect(self, d, u, v, H, W):
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        X = X.unsqueeze(0).unsqueeze(0).to(d.device).detach()  # [1, 1, H, W]
        Y = Y.unsqueeze(0).unsqueeze(0).to(d.device).detach()  # [1, 1, H, W]

        x = (X - self.delta_t * W * u).clamp(0, W - 1) * 2.0 / (W - 1) - 1.0
        y = (Y - self.delta_t * H * v).clamp(0, H - 1) * 2.0 / (H - 1) - 1.0

        grid = torch.stack((x.squeeze(1), y.squeeze(1)), dim=-1)  # [T-1, H, W, 2]
        d = F.grid_sample(d, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return d

    def forward(self, x):
        if self.reduction > 1:
            x1 = self.chnl_reduction(x)  # [BT, C//r, H, W]
        else:
            x1 = x
        BT, c, H, W = x1.shape
        x1 = x1.reshape(-1, self.n_segments, c, H, W)  # [B, T, c, H, W]
        x1_pre_pas_cat = torch.cat([x1[:, :-1], x1[:, 1:]], 2).reshape(-1, 2 * c, H, W)  # [B(T-1), 2c, H, W]
        diffusion = self.get_diffusion(x1_pre_pas_cat) * 0.05  # [B(T-1), 1, H, W]     [0, 0.05]
        velocity = 0.05 * self.get_velocity(x1_pre_pas_cat) - 0.025  # [B(T-1), 2, H, W]     [-0.025, +0.025]
        u, v = torch.split(velocity, 1, 1)  # [B(T-1), 1, H, W]

        x1_pre = x1[:, :-1, :, :, :].reshape(-1, c, H, W)  # [B(T-1), c, H, W]
        x1_pas = self.diffuse(x1_pre, diffusion)  # [B(T-1), c, H, W]
        x1_pas = self.advect(x1_pas, u, v, H, W)  # [B(T-1), c, H, W]

        x1 = (torch.cat([x1[:, 0:1, :, :, :], x1_pas.reshape((-1, self.n_segments - 1, c, H, W))], 1)).contiguous()
        if self.reduction > 1:
            x1 = self.chnl_restore(x1.reshape(-1, c, H, W))  # [BT, C, H, W]
        else:
            x1 = x1.reshape(-1, c, H, W)

        output = x1 + x * self.sigmoid(self.avg_pool(x1 - x))
        return output


class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels: int = input_channels
        self.n_segment: int = n_segment
        self.fold_div: int = n_div
        self.fold: int = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.input_channels, self.input_channels,
                              kernel_size=3, stride=1, padding=1, groups=self.input_channels,
                              bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1  # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right
            if 2 * self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1  # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1  # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch: int = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1)  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2)  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            num_segments: int,
            reduction: int,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockShift(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            num_segments: int,
            reduction: int,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)

        self.num_segments: int = num_segments
        self.reduction: int = reduction
        self.cfd = FDModule(planes, n_segment=self.num_segments, reduction=self.reduction)
        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.cfd(out)
        out = self.shift(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            num_segments: int,
            reduction: int,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckShift(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            num_segments: int,
            reduction: int,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.num_segments: int = num_segments
        self.reduction: int = reduction
        self.cfd = FDModule(width, n_segment=self.num_segments, reduction=self.reduction)
        self.shift = ShiftModule(width, n_segment=self.num_segments, n_div=8, mode='shift')
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.cfd(out)
        out = self.shift(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FDResNet(nn.Module):
    """Fluid Dynsmics Residual Network

    Args:
        block (nn.Module): BaiscBlock
        layers (int): layers
        in_ch (int): channels
        num_segments (int): time sequence length
        num_classes (int): number of classes
    """

    def __init__(
            self,
            block: Type[Union[BasicBlock, BasicBlockShift, Bottleneck, BottleneckShift]] = None,
            layers: List[int] = None,
            in_ch: int = 3,
            num_segments: int = 8,
            reduction: int = 16,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if block is None:
            block = BasicBlock
        if layers is None:
            layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes: int = 64
        self.dilation: int = 1
        self.num_segments: int = num_segments
        self.reduction: int = reduction

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups: int = groups
        self.base_width: int = width_per_group
        self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments, self.reduction, block, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments, self.reduction, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.num_segments, self.reduction, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer4 = self._make_layer(self.num_segments, self.reduction, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[0])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, num_segments, reduction, block, planes: int, blocks: int, stride=1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, reduction, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                num_segments,
                reduction,
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            ))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)
        return x
