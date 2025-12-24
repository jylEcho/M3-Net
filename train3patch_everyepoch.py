#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:46:00 2023
@author: jsyoonDL
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy
import random

from util.Dataset3patch import Dataset
from util.DataAug import DataAugmentation

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#%% seed
def set_seed(seed=0):
    """Reproducibility"""
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

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
                img32, img64, img96, labels = data

                img32 = augmentation(img32)
                img64 = augmentation(img64)
                img96 = augmentation(img96)

                img32 = img32.cuda()
                img64 = img64.cuda()
                img96 = img96.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = model(img32, img64, img96)
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
                img32, img64, img96, labels = data

                img32 = img32.cuda()
                img64 = img64.cuda()
                img96 = img96.cuda()
                labels = labels.cuda()

                outputs = model(img32, img64, img96)
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
