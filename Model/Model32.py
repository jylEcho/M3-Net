#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:39:11 2022

@author: jsyoon
"""

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from safetensors.torch import load_file  # ✅ 确保导入在这里
from timm.layers import to_2tuple, trunc_normal_, DropPath, LayerNorm2d  # ✅ 替换旧路径
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

# =============================
#  你的子模块（保持不变）
# =============================
class FeatureExtractor(nn.Module):
    def __init__(self,in_channel, num_features):
        super().__init__()
        self.pool = SelectAdaptivePool2d(1,pool_type='avg')
        self.norm = LayerNorm2d(in_channel)
        self.fc = nn.Linear(in_channel,num_features)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0) 

    def forward(self,x):
        x = self.pool(x)
        x = self.norm(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x   

class InputFilter(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.upsampling = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.norm = LayerNorm2d(out_channel)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0) 

    def forward(self,x):
        x = self.upsampling(x)
        x = self.conv(x)
        x = self.norm(x)
        return x   

# =============================
#  主模型定义（关键修改区）
# =============================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
                
        # 创建ConvNeXt骨干网络（不加载预训练）
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
        
        # 加载本地预训练权重（ImageNet 1000类）
        local_weights_path = "/root/autodl-tmp/model.safetensors"
        state_dict = load_file(local_weights_path)

        # 删除分类头不匹配参数
        keys_to_remove = [k for k in state_dict.keys() if "head.fc" in k]
        for k in keys_to_remove:
            del state_dict[k]
            print(f"跳过加载层: {k}")

        # 加载剩余部分
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("未加载的参数:", missing)
        print("意外的参数:", unexpected)

        # 保存需要的模块
        # self.stem = InputFilter(32,96)
        self.stem = InputFilter(56,96)
        # self.stem = InputFilter(56,96)
        self.stages = model.stages
        self.clf = model.head
 
    def forward(self, x):   
        x = self.stem(x)
        # print("x1_shape:", x.shape)
        x = self.stages(x)
        # print("x2_shape:", x.shape)
        x_out = self.clf(x)
        # print("x3_shape:", x_out.shape)
        return x_out
