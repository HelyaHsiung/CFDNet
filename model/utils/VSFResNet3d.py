"""
@article{zhou2024gaseous,
  title={Gaseous object detection},
  author={Zhou, Kailai and Wang, Yibo and Lv, Tao and Shen, Qiu and Cao, Xun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
https://github.com/CalayZhou/Gaseous-Object-Detection.git
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from Shift3D.modules.Shift3D import DeformConvPack
from typing import Callable, List, Optional, Type, Union


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class VSFData(nn.Module):
    def __init__(self,
            num_segments=8,
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            deformable_groups=3,
            is_inputdata=True
        ):
        super().__init__()
        self.num_segments = num_segments
        self.inputDeform_Conv3dShift3D = DeformConvPack(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            deformable_groups=deformable_groups,
            bias=False,
            is_inputdata=is_inputdata
        )

    def forward(self, x):
        # bt, c, h, w
        x = x.reshape(-1, self.num_segments, x.shape[1], x.shape[2], x.shape[3]).permute([0, 2, 1, 3, 4]).contiguous()  # b, c, t, h, w
        b, c, t, h, w = x.shape  # b, c, t, h, w
        x = self.inputDeform_Conv3dShift3D(x)  # b, c, t, h, w
        x = x.permute([0, 2, 1, 3, 4])  # bt, c, h, w
        x = x.contiguous().view(b * t, c, h, w)  # bt, c, h, w
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VSFBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(VSFBasicBlock, self).__init__()
        self.reduction = 16
        self.num_segments = num_segments
        mid_inplanes = max(8, planes // self.reduction)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.chnl_reduction = nn.Conv2d(planes, mid_inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.Deform_Conv3dShift3D = DeformConvPack(in_channels=mid_inplanes, out_channels=mid_inplanes,
                                                   kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0],
                                                   deformable_groups=mid_inplanes, bias=False, n_segment=self.num_segments, is_inputdata=False)
        self.chnl_restore = nn.Conv2d(mid_inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_t = conv1x1(planes, planes)
        nn.init.xavier_normal_(self.conv_t.weight)
        self.bn_t = nn.BatchNorm2d(num_features=planes)
        self.sigmoid_t = nn.Sigmoid()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # ----------VSF-----------
        bt, _, h, w = out.shape
        out_3d = self.chnl_reduction(out)  # bt, c, h, w
        out_3d = out_3d.view(bt // self.num_segments, self.num_segments, -1, h, w)
        out_3d = out_3d.permute([0, 2, 1, 3, 4])
        out_3d = out_3d.contiguous().view(bt // self.num_segments, -1, self.num_segments, h, w)  # b, c, t, h, w
        out_3d = self.Deform_Conv3dShift3D(out_3d)
        out_3d = out_3d.permute([0, 2, 1, 3, 4])  # b, t, c, h, w
        out_3d = out_3d.contiguous().view(bt, -1, h, w)  # bt, c, h, w
        out_3d = self.chnl_restore(out_3d)

        out_diff = out_3d - out  # bt c h w
        y = self.avg_pool(out_diff)  # bt c 1 1
        y = self.conv_t(y)  # bt c 1 1
        y = self.bn_t(y)
        y = self.sigmoid_t(y)
        y = (y - 0.5) * 2
        out_final = out_3d + y.expand_as(out_3d) * out

        out = self.conv2(out_final)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VSFBottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(VSFBottleneck, self).__init__()
        self.reduction = 16
        self.num_segments = num_segments
        mid_inplanes = max(8, planes // self.reduction)

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.chnl_reduction = nn.Conv2d(planes, mid_inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.Deform_Conv3dShift3D = DeformConvPack(in_channels=mid_inplanes, out_channels=mid_inplanes,
                                                   kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0],
                                                   deformable_groups=mid_inplanes, bias=False, n_segment=self.num_segments, is_inputdata=False)
        self.chnl_restore = nn.Conv2d(mid_inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_t = conv1x1(planes, planes)
        nn.init.xavier_normal_(self.conv_t.weight)
        self.bn_t = nn.BatchNorm2d(num_features=planes)
        self.sigmoid_t = nn.Sigmoid()

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # ----------VSF-----------
        bt, _, h, w = out.shape
        out_3d = self.chnl_reduction(out)  # bt, c, h, w
        out_3d = out_3d.view(bt // self.num_segments, self.num_segments, -1, h, w)
        out_3d = out_3d.permute([0, 2, 1, 3, 4])
        out_3d = out_3d.contiguous().view(bt // self.num_segments, -1, self.num_segments, h, w)  # b, c, t, h, w
        out_3d = self.Deform_Conv3dShift3D(out_3d)
        out_3d = out_3d.permute([0, 2, 1, 3, 4])  # b, t, c, h, w
        out_3d = out_3d.contiguous().view(bt, -1, h, w)  # bt, c, h, w
        out_3d = self.chnl_restore(out_3d)

        out_diff = out_3d - out  # bt c h w
        y = self.avg_pool(out_diff)  # bt c 1 1
        y = self.conv_t(y)  # bt c 1 1
        y = self.bn_t(y)
        y = self.sigmoid_t(y)
        y = (y - 0.5) * 2
        out_final = out_3d + y.expand_as(out_3d) * out  # bt, c, h, w

        out = self.conv2(out_final)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class VSFResNet3d(nn.Module):
    def __init__(self,
            block: Type[Union[BasicBlock, VSFBasicBlock, Bottleneck, VSFBottleneck]] = None,
            layers: List[int] = None,
            in_ch: int = 3,
            num_segments: int = 8,
            num_classes: int = 1000):
        super().__init__()
        
        if block is None:
            block = BasicBlock
        if layers is None:
            layers = [2, 2, 2, 2]
        self.inplanes = 64
        self.num_segments = num_segments

        self.vsf_data = VSFData(num_segments=num_segments, in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0, deformable_groups=in_ch, is_inputdata=True)
        self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments, block, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments, block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_segments, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vsf_data(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)
        return x

