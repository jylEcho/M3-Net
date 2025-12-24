#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:18:34 2023
@author: jsyoonDL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, time, random, numpy as np

# from model.Model import Model as Model64
# from model.Model_xlarge import Model as Model32
# from model.ResNet34 import Model as Model96

from model.Model_base import Model as Model32
from model.ResNet18 import Model as Model96
from model.ResNet34 import Model as Model64

from model.ModelV10a import Model
from train import train

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ======= reproducibility =======
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ======= main =======
seed_num = 0
set_seed(seed_num)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loss_function = nn.CrossEntropyLoss().cuda()

# ======= 加载三个模型权重 =======
net32 = Model32()
net64 = Model64()
net96 = Model96()

# ckpt32 = torch.load('/root/autodl-tmp/model_trained/proposed_convnext_xlarge32/trained_model.pt')
# ckpt64 = torch.load('/root/autodl-tmp/model_trained/proposed_convnext_tiny64/trained_model.pt')
# ckpt96 = torch.load('/root/autodl-tmp/model_trained/proposed_ResNet34_96/trained_model.pt')

ckpt32 = torch.load('/root/autodl-tmp/model_trained/proposed_convnext_base32-all/trained_model.pt')
ckpt64 = torch.load('/root/autodl-tmp/model_trained/proposed_ResNet34_64-all/trained_model.pt')
ckpt96 = torch.load('/root/autodl-tmp/model_trained/proposed_ResNet18_96-all/trained_model.pt')

net32.load_state_dict(ckpt32, strict=False)
net64.load_state_dict(ckpt64, strict=False)
net96.load_state_dict(ckpt96, strict=False)

# ======= 构建融合模型 =======
model = Model(net32, net64, net96, freeze_backbones=True)
model.cuda()

# ======= optimizer 只优化融合层参数 =======
# optimizer = optim.AdamW(model.fusion_fc.parameters(), lr=1e-4, weight_decay=1e-5)
# 只优化融合层（CrossAttention + Transformer + 分类头）
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4, 
    weight_decay=1e-5
)


data_path = '/root/autodl-tmp/npy32'
model_path = '/root/autodl-tmp/model_trained/proposed_ablation_a'
os.makedirs(model_path, exist_ok=True)

params = {
    'num_epochs': 200,
    'batch_size': 32,
    'seed_num': seed_num,
    'optimizer': optimizer,
    'loss_function': loss_function,
    'data_path': data_path,
    'model_path': model_path,
    'acc_best': 0,
    'norm': 0,
    'lambda': 1e-3,
}

train(model, params)
torch.cuda.empty_cache()
