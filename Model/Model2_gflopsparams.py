#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D CNN + FCN for two-channel volumetric output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------
# 基础块：Conv3D + BN + ReLU
# ---------------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------
# 下采样模块
# ---------------------------------
class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock3D(in_ch, out_ch),
            ConvBlock3D(out_ch, out_ch)
        )
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down  # 返回特征和下采样结果


# ---------------------------------
# 上采样模块
# ---------------------------------
class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock3D(in_ch, out_ch),
            ConvBlock3D(out_ch, out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 保证尺寸匹配（由于除法可能有误差）
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------
# 主网络: 3D-UNet 风格
# ---------------------------------
class CNN3D_FCN(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, out_channels=2):
        super().__init__()
        # 编码器
        self.down1 = DownBlock3D(in_channels, base_channels)
        self.down2 = DownBlock3D(base_channels, base_channels * 2)
        self.down3 = DownBlock3D(base_channels * 2, base_channels * 4)

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock3D(base_channels * 4, base_channels * 8),
            ConvBlock3D(base_channels * 8, base_channels * 8)
        )

        # 解码器
        self.up3 = UpBlock3D(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2)
        self.up1 = UpBlock3D(base_channels * 2, base_channels)

        # 输出头 (FCN部分)
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x = self.down1(x)  # 1/2
        x2, x = self.down2(x)  # 1/4
        x3, x = self.down3(x)  # 1/8

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # 输出映射
        out = self.out_conv(x)  # [B, 2, D, H, W]
        return out


# ---------------------------------
# ✅ 测试样例
# ---------------------------------
if __name__ == '__main__':
    import os
    import torch
    from thop import profile, clever_format

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 确保输入张量维度正确 (B, C, D, H, W)
    # 这里修正了原代码的维度错误（缺少通道维度）
    x = torch.randn(1, 3, 32, 32, 32).cuda()  # 1个样本，3通道，32x32x32体素

    # 初始化模型并移动到GPU
    model = CNN3D_FCN(in_channels=3, base_channels=32, out_channels=2).cuda()
    model.eval()  # 推理模式

    # 计算FLOPs和参数数量
    with torch.no_grad():  # 禁用梯度计算，提高效率
        flops, params = profile(model, inputs=(x,))
    
    # 格式化输出（自动转换为合适的单位，如GFLOPs、MParams）
    flops, params = clever_format([flops, params], "%.3f")

    print(f"模型计算量: {flops}")
    print(f"模型参数数量: {params}")
