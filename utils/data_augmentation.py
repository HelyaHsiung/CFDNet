import torch
import torch.nn as nn
from itertools import combinations

"""
Helya Hsiung, haiyangxiong@whu.edu.cn, 2024.11.21
http://rsidea.whu.edu.cn/
"""


class LED(nn.Module):
    """
    Local Entropy and combination of Differentiation Module (LED Module), used for enhancement
    """
    def __init__(self, in_ch, patch, bins):
        super().__init__()
        self.in_ch = in_ch
        self.patch = patch
        self.bins = bins
        self.eps = 1e-6
        
    def get_local_entropy(self, x):
        """
        use sliding window to calculate local entropy
        Args:
            input_tensor: (B, 1, H, W) or (B, 1, T, H, W)
            eps: minimal value
        Returns:
            entropy_map: (B, 1, H, W) or (B, 1, T, H, W)
        """
        if x.ndim == 5:
            video_flag = True
            B, C, T, H, W = x.shape
            x = x.transpose(1, 2).reshape((B * T, C, H, W))                                                         # [BT, C, H, W]
        elif x.ndim == 4:
            video_flag = False
            B, C, H, W = x.shape
        else:
            raise ValueError(f"Except ndim of x to be 4 or 5, but x.ndim = {x.ndim}.")
        
        x = x.reshape(-1, H, W)                                                                                     # [BTC, H, W]
        # qualify to 0 ~ bins-1
        x = (x - torch.amin(x, (1, 2), True)) / (torch.amax(x, (1, 2), True) - torch.amin(x, (1, 2), True) + 1e-6)  # [BTC, H, W]
        x = (x * (self.bins - 1)).unsqueeze(-1).long()                                                              # [BTC, H, W, 1]

        # onehot code
        one_hot = torch.zeros((x.shape[0], H, W, self.bins), device=x.device, requires_grad=False)                  # [BTC, H, W, bins]
        one_hot.scatter_(-1, x, 1)
        one_hot = one_hot.float().permute(0, 3, 1, 2).contiguous()                                                  # [BTC, bins, H, W]

        # statistic frequency
        kernel = torch.ones(self.bins, 1, self.patch, self.patch, device=x.device, requires_grad=False)
        histogram = nn.functional.conv2d(one_hot, kernel, padding=self.patch // 2, groups=self.bins)                # [BTC, bins, H, W]

        # calculate probability
        prob = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(self.eps)                                   # [BTC, bins, H, W]

        # get entropy
        entropy_map = - torch.sum(prob * torch.log2(prob.clamp_min(self.eps)), dim=1)                               # [BTC, H, W]
        
        entropy_map = entropy_map.reshape(-1, C, H, W)                                                              # [BT, C, H, W]
        if video_flag:
            entropy_map = entropy_map.reshape(B, T, C, H, W).transpose(1, 2).contiguous()
        else:
            pass
        return entropy_map

    def normalize(self, x):
        min_x = torch.amin(x, dim=(-2, -1), keepdim=True)
        max_x = torch.amax(x, dim=(-2, -1), keepdim=True)

        x = (x - min_x) / (max_x - min_x + self.eps)

        return x
    
    def forward(self, x):
        # 组合选择, 6选2的所有组合
        combs = list(combinations(range(self.in_ch), 2))
        out = []

        for c1, c2 in combs:
            # 计算差分并将其添加到输出列表中
            diff = x[:, c1] - x[:, c2]  # (B, H, W) or (B, T, H, W)
            out.append(diff.unsqueeze(1))  # 添加新的维度，形状为 (B, 1, H, W) or (B, 1, T, H, W)
        
        out.append(self.get_local_entropy(out[8]))
        
        # 将所有差分以及局部熵组合在一起，得到形状为 (B, 16, H, W) or (B, 16, T, H, W)
        out = torch.cat(out, dim=1)  # 在通道维度拼接
        out = self.normalize(out).to(x.device)
        return out
