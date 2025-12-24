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
        
        # 使用提供的路径和模式加载文件列表
        with open('{}/splits{}.txt'.format(path,mode), 'r') as fin:
            data_list = [line.replace('\n','') for line in fin]
            
        
        self.img_path_list = sorted(data_list)
        # 存储所有完整路径
        self.filepaths = self.img_path_list 
        self.label_list = [x.split('/')[-2] for x in self.img_path_list]

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        # True -> 1, False -> 0
        label = int(self.label_list[index]=='True') 

        # 自动推导出其它尺度路径
        # 假设所有尺度的路径结构相同，仅根目录名称不同 (npy32 -> npy64/npy96)
        img_path_64 = img_path.replace('npy32', 'npy64')
        img_path_96 = img_path.replace('npy32', 'npy96')
        
        # 加载数据
        try:
            img32 = np.load(img_path).astype(float)
            img64 = np.load(img_path_64).astype(float)
            img96 = np.load(img_path_96).astype(float)
        except Exception as e:
            print(f"Error loading files for path: {img_path}. Error: {e}")
            raise e
            
        def normalize(img):
            # 归一化函数
            return (img - img.min()) / (img.max() - img.min() + 1e-8)
            
        img32 = normalize(img32)
        img64 = normalize(img64)
        img96 = normalize(img96)

        # ---- 转为 Tensor 并统一维度 ----
        img32 = torch.tensor(img32.transpose(2,1,0), dtype=torch.float32)
        img64 = torch.tensor(img64.transpose(2,1,0), dtype=torch.float32)
        img96 = torch.tensor(img96.transpose(2,1,0), dtype=torch.float32)

        # 💥 关键修改点: 返回 5 个元素（路径是第 5 个）
        return img32, img64, img96, torch.tensor(label, dtype=torch.long), img_path


    def __len__(self):
        return len(self.img_path_list)
    
    def get_labels(self):
        return self.label_list