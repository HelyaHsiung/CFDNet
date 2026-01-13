#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable
from Shift3D import core


class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset,  bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        # ctx.kernel_size = _triple(weight.shape[2:5])
        ctx.kernel_size = _triple([1,1,1])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        output = core.deform_conv_forward(input,  bias,
                                             offset,
                                             ctx.kernel_size[0], ctx.kernel_size[1],ctx.kernel_size[2],
                                             ctx.stride[0], ctx.stride[1],ctx.stride[2],
                                             ctx.padding[0], ctx.padding[1],ctx.padding[2],
                                             ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
                                             ctx.group,
                                             ctx.deformable_groups,
                                             ctx.im2col_step)
        ctx.save_for_backward(input, offset, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_bias = \
            core.deform_conv_backward(input, #weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                     ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)
        return grad_input, grad_offset, grad_bias,\
            None, None, None, None, None, None
