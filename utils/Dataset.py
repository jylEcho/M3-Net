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
import os  # 新增：用于提取文件名

#%% Custom dataset train
class Dataset(Dataset):
    def __init__(self, path, mode ='train'):
        super().__init__() 
        
        with open('{}/splits{}.txt'.format(path,mode), 'r') as fin:
            data_list = [line.replace('\n','') for line in fin]
            
        self.img_path_list = sorted(data_list)
        self.label_list =  [x.split('/')[-2] for x in self.img_path_list]  # 原始标签（'True'/'False'字符串）

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        # 原始标签（字符串格式：'True'/'False'），同时保留数值标签（0/1）用于模型计算
        label_str = self.label_list[index]
        label = int(label_str == 'True')  # 0->False，1->True
        
        img = np.load(img_path).astype(float)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        img = torch.Tensor(img.transpose(2,1,0))  # 调整维度顺序（根据模型输入要求）

        # 新增：返回 图像张量 + 数值标签 + 原始标签字符串 + 文件路径
        return img, label, label_str, img_path

    def __len__(self):
        return len(self.img_path_list)
    
    def get_labels(self):
        return self.label_list