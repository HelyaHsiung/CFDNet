import torch
from torch import Tensor
from typing import List

def deform_conv_forward(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    deformable_groups: int,
    im2col_step: int,
    kernel_h: int,
    kernel_w: int,
    channels_in: int,
    height_in: int,
    width_in: int,
    batch_size: int
) -> Tensor: ...
"""
Performs deformable convolution forward pass.
"""

def deform_conv_backward(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    deformable_groups: int,
    im2col_step: int,
    kernel_h: int,
    kernel_w: int,
    channels_in: int,
    height_in: int,
    width_in: int,
    batch_size: int
) -> List[Tensor]: ...
"""
Computes gradients for input, offset and weight in deformable convolution.
"""
