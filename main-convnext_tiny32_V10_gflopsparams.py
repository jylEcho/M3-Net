#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:18:34 2023
@author: jsyoonDL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os, time, random, numpy as np

from model.Model_base import Model as Model32
from model.ResNet34 import Model as Model64
from model.ResNet18 import Model as Model96
from model.ModelV10_gflopsparams import Model
from train3patch import train

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ===========================================
# ============ reproducibility ==============
# ===========================================
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ===========================================
# ============= model loading ===============
# ===========================================
def build_model():
    net32 = Model32()
    net64 = Model64()
    net96 = Model96()

    ckpt32 = torch.load('/root/autodl-tmp/model_trained/proposed_convnext_base32/trained_model.pt')
    ckpt64 = torch.load('/root/autodl-tmp/model_trained/proposed_ResNet34_64/trained_model.pt')
    ckpt96 = torch.load('/root/autodl-tmp/model_trained/proposed_ResNet18_96/trained_model.pt')

    net32.load_state_dict(ckpt32, strict=False)
    net64.load_state_dict(ckpt64, strict=False)
    net96.load_state_dict(ckpt96, strict=False)

    model = Model(net32, net64, net96, freeze_backbones=True)
    model.cuda()
    return model


# ===========================================
# ================ TRAIN MAIN ================
# ===========================================
def train_main():
    seed_num = 0
    set_seed(seed_num)

    model = build_model()

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.fc_head.parameters(), lr=1e-4, weight_decay=1e-5)

    data_path = '/root/autodl-tmp/npy32'
    model_path = '/root/autodl-tmp/model_trained/proposed_Ours32_gflopsparams'
    os.makedirs(model_path, exist_ok=True)

    params = {
        'num_epochs': 200,
        'batch_size': 16,
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


# ===========================================
# ================ FLOPS MAIN ================
# ===========================================
def flops_main():
    from thop import profile, clever_format

    model = build_model()
    model.eval()

    # ====== 构造测试输入 ======
    x32 = torch.randn(1, 56, 32, 32).cuda()
    x64 = torch.randn(1, 56, 64, 64).cuda()
    x96 = torch.randn(1, 56, 96, 96).cuda()

    # ====== 计算 FLOPs & Params ======
    macs, params = profile(model, inputs=(x32, x64, x96))
    macs, params = clever_format([macs, params], "%.3f")

    print("===================================")
    print(" ModelV10  Complexity Summary")
    print("===================================")
    print(f"GFLOPs : {macs}")
    print(f"Params : {params}")
    print("===================================")


# ===========================================
# ================ ENTRY POINT ===============
# ===========================================
if __name__ == "__main__":
    flops_main()        # 只执行 FLOPs
    # train_main()      # 如需训练，取消注释
