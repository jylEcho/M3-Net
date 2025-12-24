#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:46:00 2023

@author: jsyoonDL
"""

import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR

from tqdm import tqdm
import os

from util.Dataset_source import Dataset
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from torch.utils.data import Subset
from util.DataAug import DataAugmentation
import numpy
import random

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#%%
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seedTrue
    os.environ['PYTHONHASHSEED'] = str(seed)
#%% train
def train(model, params):

    # ===== params =====
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    optimizer = params['optimizer']
    loss_function = params['loss_function']
    data_path = params['data_path']
    model_path = params['model_path']
    norm = params['norm']
    l_lambda = params['lambda']

    best = 0.0

    # ===== dataset =====
    ds_tr = Dataset(data_path, 'train')
    ds_val = Dataset(data_path, 'val')

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    augmentation = DataAugmentation()

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(dl_tr),
        eta_min=0
    )

    # ===== training loop =====
    for epoch in range(num_epochs):

        # -------------------- TRAIN --------------------
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        with tqdm(dl_tr, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            for data in tepoch:
                inputs, labels = data

                inputs = augmentation(inputs)
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                # ----- regularization -----
                if norm in [1, 2]:
                    l_norm = torch.norm(
                        torch.cat([p.view(-1) for p in model.parameters()]),
                        p=norm
                    )
                    loss = loss + l_lambda * l_norm

                loss.backward()
                optimizer.step()
                scheduler.step()

                # ----- metrics -----
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                acc = 100.0 * correct / total
                tepoch.set_postfix(
                    loss=running_loss / total,
                    acc=acc,
                    lr=optimizer.param_groups[0]['lr']
                )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # -------------------- VALIDATION (每个 epoch) --------------------
        model.eval()
        total = 0
        correct = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for data in dl_val:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                val_loss_sum += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss_sum / total
        val_acc = 100.0 * correct / total

        # -------------------- SAVE BEST --------------------
        if val_acc >= best:
            best = val_acc
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path + '/trained_model.pt')

        # -------------------- LOG --------------------
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}% | "
            f"Best Acc: {best:.2f}%"
        )

    return best

