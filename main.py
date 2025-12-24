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

import os
import time

import numpy as np
import random

# from model.Model import Model
# from model.Model2 import Model
from model.ResNet18 import ResNet3D, Model
from train import train


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#%%
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.enabled = False
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
#%% model proposed
seed_num = 0
set_seed(seed_num)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
loss_function = nn.CrossEntropyLoss().cuda()
model_path = '/root/autodl-tmp/model_trained/proposed_resNet188'                                                                                                                                                                                                           
model = Model(
    model_name="resnet3d18",  # 你也可以写成 "resnet3d34"
    num_classes=2,            # 二分类任务
    input_channels=1          # 输入通道=1（CT图像）
)
# model = Model(
#     model_name="resnet3d34",  # 你也可以写成 "resnet3d34"
#     num_classes=2,            # 二分类任务
#     input_channels=1          # 输入通道=1（CT图像）
# )
# model = resnet3d18(num_classes=2, input_channels=1)
# model_config = [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]]
# model = Model(model_config)  
model = Model()
model.cuda()

save_path = model_path
os.makedirs(save_path, exist_ok=True)        

optimizer = optim.AdamW(model.parameters(),lr=1e-4, weight_decay=1e-5)  
data_path = '/root/autodl-tmp/npy32'  
params = {
    'num_epochs': 200,
    'batch_size': 8,
    'seed_num':seed_num,
    'optimizer':optimizer,
    'loss_function':loss_function,
    'data_path': data_path,
    'model_path': model_path,
    'acc_best': 0,
    'norm': 0,
    'lambda':1e-3
    }            
                                         
train(model, params)
                                              
torch.cuda.empty_cache()
    
