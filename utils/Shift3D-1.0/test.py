# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import torch.nn as nn
import torch as tr
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import math
import torch.nn.init as init
from mmengine.model import BaseModule
from Shift3D.modules.Shift3D import DeformConvPack
import cv2

_MOMENTUM = 0.1


# class L2Norm(nn.Module):
class L2Norm(BaseModule):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class Bottleneck(nn.Module):
class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self, inplanes, planes, num_frames, stride=1, downsample=None, init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg)

        self.num_frames = num_frames
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        # self.me = MEModule(planes,reduction=16,n_segment = self.num_frames)
        self.conv_t = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, bias=False)
        nn.init.xavier_normal_(self.conv_t.weight)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.tanh = nn.Tanh()
        self.bn_t = nn.BatchNorm2d(num_features=planes)
        self.sigmoid_t = nn.Sigmoid()

        print("Voxel Shift Field!")
        self.Deform_Conv3dShift3D = DeformConvPack(in_channels=planes, out_channels=planes,
                                                   kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0],
                                                   deformable_groups=planes, bias=False, n_segment=self.num_frames, is_inputdata=False)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # ----------VSF-----------
        bt, c, h, w = out.shape
        out = out.view(bt // self.num_frames, self.num_frames, c, h, w)
        out = out.permute([0, 2, 1, 3, 4])
        out = out.contiguous().view(bt // self.num_frames, c, self.num_frames, h, w)

        out_3d = self.Deform_Conv3dShift3D(out)

        out = out.permute([0, 2, 1, 3, 4])
        out = out.contiguous().view(bt, c, h, w)

        out_3d = out_3d.permute([0, 2, 1, 3, 4])
        out_3d = out_3d.contiguous().view(bt, c, h, w)

        out_diff = out_3d - out  # nt c h w
        y = self.avg_pool(out_diff)  # nt c 1 1
        y = self.conv_t(y)  # bt c 1 1
        y = self.bn_t(y)
        y = self.sigmoid_t(y)
        y = (y - 0.5) * 2
        out_final = out_3d + y.expand_as(out_3d) * out

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


class ResNet3DShift_multihead(BaseModule):
    def __init__(self, n_segment, init_cfg=None):
        super(ResNet3DShift_multihead, self).__init__(init_cfg)
        block = Bottleneck
        layers = [3, 4, 6, 3]
        scale = 4
        self.n_segment = n_segment
        self.num_frames = n_segment
        self.output_channel = 64
        self.inplanes = 64
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.temporal_fusion_module = nn.ModuleList()
        for i in range(4):
            input_channels = 256 * 2 ** (i)
            self.temporal_fusion_module.append(nn.Sequential(
                nn.Conv2d(input_channels, input_channels // self.n_segment, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(input_channels // self.n_segment, momentum=0.01),
                nn.ReLU(inplace=True)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.inputDeform_Conv3dShift3D = DeformConvPack(in_channels=3, out_channels=3, kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0], deformable_groups=3, bias=False, is_inputdata=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_frames, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.num_frames))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input.reshape(-1, self.n_segment, input.shape[1], input.shape[2], input.shape[3]).permute([0, 2, 1, 3, 4]).contiguous()

        b, c, t, h, w = x.shape
        x_cir = self.inputDeform_Conv3dShift3D(x)
        x = x_cir.permute([0, 2, 1, 3, 4])
        x = x.contiguous().view(b * t, c, h, w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # FUSION
        x_all_list = [x1, x2, x3, x4]
        x_stage1_list = []
        x_stage2_list = []
        count = 0
        for xi in x_all_list:
            xi_list = xi.view((b, t) + xi.size()[1:]).split(1, dim=1)
            xi_stage1 = []
            xi_stage2 = []
            for j in range(len(xi_list)):
                xi_list_j_compress = self.temporal_fusion_module[count](xi_list[j].squeeze(1))
                xi_stage1.append(xi_list_j_compress)
                xi_stage2.append(xi_list[j].squeeze(1))
            xi_stage1_cat = torch.cat(xi_stage1, dim=1)
            x_stage1_list.append(xi_stage1_cat)
            x_stage2_list.append(xi_stage2)
            count += 1

        x1_stage1, x2_stage1, x3_stage1, x4_stage1 = x_stage1_list[0], x_stage1_list[1], x_stage1_list[2], x_stage1_list[3]
        x1_stage2, x2_stage2, x3_stage2, x4_stage2 = x_stage2_list[0], x_stage2_list[1], x_stage2_list[2], x_stage2_list[3]

        outs_stage2 = []
        outs_stage1 = []
        outs_stage1.append(x1_stage1)
        outs_stage1.append(x2_stage1)
        outs_stage1.append(x3_stage1)
        outs_stage1.append(x4_stage1)

        for i in range(self.num_frames):
            outs = []
            outs.append(x1_stage2[i].squeeze(1))
            outs.append(x2_stage2[i].squeeze(1))
            outs.append(x3_stage2[i].squeeze(1))
            outs.append(x4_stage2[i].squeeze(1))
            outs_stage2.append(tuple(outs))

        return tuple(outs_stage2), tuple(outs_stage1)


if __name__ == '__main__':
    x1 = torch.rand(1 * 8, 3, 288, 288).cuda()
    print(x1.shape)
    ResNet3DShift = ResNet3DShift_multihead(8).cuda()
    out1 = ResNet3DShift(x1)
    print("done!")
