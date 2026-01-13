#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple

from Shift3D.functions.Shift3D_func import DeformConvFunction


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, n_segment = 8, is_inputdata = False):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        self.is_inputdata = is_inputdata
        self.weight = nn.Parameter(torch.eye(
            out_channels, in_channels).unsqueeze(2).unsqueeze(2).unsqueeze(2))

        self.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_channels))
        if not self.use_bias:
            self.bias.requires_grad = False
        
        if self.is_inputdata:
            bias_value = 1.0
            self.offset_bias_uniform = torch.arange(-bias_value,bias_value+2*bias_value/(self.deformable_groups-1),2*bias_value/(self.deformable_groups-1)).unsqueeze(1)
            self.offset_bias_zeros = torch.zeros_like(self.offset_bias_uniform)
            
            self.offset_bias = torch.cat([self.offset_bias_uniform,self.offset_bias_zeros,self.offset_bias_zeros],dim=1).view(-1).unsqueeze(1)#.unsqueeze(1).unsqueeze(1).unsqueeze(0)
            self.offset_bias = self.offset_bias.repeat(1,n_segment)
            self.offset_bias[0,0] = 7
            self.offset_bias[6,7] = -7
            self.offset_bias = self.offset_bias.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            self.offset_bias = nn.Parameter(self.offset_bias)
            self.offset_bias.requires_grad = False
        else:
            bias_subone = torch.zeros(in_channels//8)-1
            bias_plusone = torch.zeros(in_channels//8)+1
            bias_subtwo = torch.zeros(in_channels//8)-2
            bias_plustwo = torch.zeros(in_channels//8)+2

            bias_zeros =  torch.zeros(in_channels//4*2)
            self.offset_bias_uniform = torch.cat([bias_subtwo,bias_plustwo,bias_subone,bias_plusone,bias_zeros],dim=0).unsqueeze(1)
            self.offset_bias_zeros = torch.zeros_like(self.offset_bias_uniform)
            self.offset_bias = torch.cat([self.offset_bias_uniform,self.offset_bias_zeros,self.offset_bias_zeros],dim=1).view(-1).unsqueeze(1)
            self.offset_bias = self.offset_bias.repeat(1,n_segment)
            self.offset_bias = self.offset_bias.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            self.offset_bias = nn.Parameter(self.offset_bias)

    def reset_parameters(self):
        n = self.in_channels
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
            offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv = DeformConvFunction.apply

class DeformConvPack(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, n_segment = 8, is_inputdata = False,lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias, n_segment, is_inputdata)

        self.is_inputdata = is_inputdata
        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        if not self.is_inputdata:
            self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=[3,3,3],#,self.kernel_size,
                                          stride=self.stride,
                                          padding=[1,1,1],#self.padding,
                                          bias=True)
            self.conv_offset.lr_mult = lr_mult
            self.init_offset()


    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        if not self.is_inputdata:
            offset_conv = self.conv_offset(input)
            offset = offset_conv + self.offset_bias
        else:
            b,c,t,h,w = input.shape
            offset_conv_zeros = torch.zeros([b,c*3,t,h,w]).cuda()
            offset_conv_zeros = nn.Parameter(offset_conv_zeros)
            offset_conv_zeros.requires_grad = False
            offset = offset_conv_zeros + self.offset_bias

        return DeformConvFunction.apply(input, offset,
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)

