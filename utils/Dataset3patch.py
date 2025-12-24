#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:04:04 2023

@author: jsyoonDL
"""

from torch.utils.data import Dataset
import glob
import torch
import numpy as np
#%% Custom dataset train
class Dataset(Dataset):
    def __init__(self, path, mode ='train'):
        super().__init__() 
        
       
        with open('{}/splits{}.txt'.format(path,mode), 'r') as fin:
            data_list = [line.replace('\n','') for line in fin]
            
        
        self.img_path_list = sorted(data_list)
        self.label_list =  [x.split('/')[-2] for x in self.img_path_list]

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = int(self.label_list[index]=='True')

        # 自动推导出其它尺度路径
        img_path_64 = img_path.replace('npy32', 'npy64')
        img_path_96 = img_path.replace('npy32', 'npy96')
        
        # # ---- Debug 输出 ----
        # print(f"img_path_32: {img_path}")
        # print(f"img_path_64: {img_path_64}")
        # print(f"img_path_96: {img_path_96}")
        # print(f"label: {label}")
        
        img32 = np.load(img_path).astype(float)
        img64 = np.load(img_path_64).astype(float)
        img96 = np.load(img_path_96).astype(float)
        
        def normalize(img):
            return (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img32 = normalize(img32)
        img64 = normalize(img64)
        img96 = normalize(img96)

        # ---- 转为 Tensor 并统一维度 (C, D, H, W) ----
        img32 = torch.tensor(img32.transpose(2,1,0), dtype=torch.float32)
        img64 = torch.tensor(img64.transpose(2,1,0), dtype=torch.float32)
        img96 = torch.tensor(img96.transpose(2,1,0), dtype=torch.float32)

        # 返回多尺度 
        # print(f"img32:", img32.shape)
        # print(f"img64:", img64.shape)
        # print(f"img96:", img96.shape)
        # print(f"label:", label.shape)
        
        return img32, img64, img96, label


    def __len__(self):
        return len(self.img_path_list)
    
    def get_labels(self):
        return self.label_list