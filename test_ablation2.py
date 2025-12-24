#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 2025
Test script for three-scale fusion model (ModelV4)
@author: lijinyue
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, time, random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

from torchmetrics.functional import auroc, specificity, precision_recall, precision_recall_curve, auc

# ===== import your dataset and model =====
from util.Dataset import Dataset
from model.ModelV10ablation2 import Model

# from model.Model import Model as Model64
# from model.Model_xlarge import Model as Model32
# from model.ResNet34 import Model as Model96

from model.Model_base import Model as Model32
from model.ResNet18 import Model as Model96
from model.ResNet34 import Model as Model64

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ===== reproducibility =====
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


# ===== performance report =====
def metric_report(y_true, y_pred):
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP, FN, TP, TN = FP.astype(float), FN.astype(float), TP.astype(float), TN.astype(float)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)

    Report = [TPR, TNR, PPV, ACC]
    Report = pd.DataFrame(Report, index=['Sensitivity', 'Specificity', 'Precision', 'ACC'])
    return Report.T


# ===== test function =====
def test(model, params):
    batch_size = params['batch_size']
    loss_function = params['loss_function']
    device = params['device']
    data_path = params['data_path']
    model_name = params['model_name']

    ds = Dataset(data_path, 'test')
    classes = ['False', 'True']

    dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)

    total, correct = 0, 0
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    lb_list = torch.zeros(0, dtype=torch.long, device='cpu')

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dl, 0), desc='Testing'):
            img32, img64, img96, labels = data
            img32 = img32.to(device)
            img64 = img64.to(device)
            img96 = img96.to(device)
            labels = labels.to(device)

            outputs = model(img32, img64, img96)
            predict_proba = torch.nn.Softmax(dim=-1)(outputs)
            _, predicted = torch.max(predict_proba, 1)

            pred_list = torch.cat([pred_list, predicted.view(-1).cpu()])
            lb_list = torch.cat([lb_list, labels.view(-1).cpu()])

            if i == 0:
                pred_score_list = predict_proba.detach().cpu().numpy()
            else:
                pred_score_list = np.concatenate(
                    [pred_score_list, predict_proba.detach().cpu().numpy()], axis=0
                )

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # ===== metrics =====
    acc = 100 * correct / total
    f1 = f1_score(lb_list, pred_list, average='weighted')
    roc_auc = auroc(torch.tensor(pred_score_list, dtype=torch.float32), lb_list, num_classes=2)
    bacc = balanced_accuracy_score(lb_list, pred_list)
    spec = specificity(pred_list, lb_list, average='weighted', num_classes=2)
    pre, rec = precision_recall(pred_list, lb_list, average='weighted', num_classes=2)

    pre_vec, rec_vec, thresholds = precision_recall_curve(
        torch.tensor(pred_score_list, dtype=torch.float32), lb_list, num_classes=2
    )
    dlen = len(ds)
    pr_auc = 0
    for idx in range(2):
        w = len((lb_list == idx).nonzero(as_tuple=False)) / dlen
        pr_auc += w * auc(rec_vec[idx], pre_vec[idx])

    # ===== confusion matrix =====
    cf_matrix = confusion_matrix(lb_list.numpy(), pred_list.numpy())
    df_cm = pd.DataFrame(cf_matrix.astype('int'), index=classes, columns=classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt="g")
    con_path = os.path.join('confusion_matrix_test', model_name)
    os.makedirs(con_path, exist_ok=True)
    plt.savefig(os.path.join(con_path, 'output.png'))

    print(
        f'Acc: {acc:.2f}, BAcc: {bacc*100:.2f}, Spec: {spec:.4f}, Pre: {pre:.4f}, '
        f'Rec: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}'
    )
    print('Confusion matrix')
    print(cf_matrix)
    print(metrics.classification_report(lb_list.numpy(), pred_list.numpy(), digits=4))
    print(metric_report(lb_list.numpy(), pred_list.numpy()))
    return acc


# ===== main =====
if __name__ == '__main__':
    seed = 0
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_function = nn.CrossEntropyLoss().cuda()

    # ===== load submodels =====
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

    # ===== build fusion model =====
    model = Model(net32, net64, net96, freeze_backbones=True)
    model_path = '/root/autodl-tmp/model_trained/proposed_ablation2'
    model.load_state_dict(torch.load(os.path.join(model_path, 'trained_model.pt')))
    model = model.to(device)

    # ===== test =====
    data_path = '/root/autodl-tmp/npy32'
    model_name = 'proposed_V10'
    params = {
        'batch_size': 48,
        'data_path': data_path,
        'loss_function': loss_function,
        'model_name': model_name,
        'device': device
    }

    preds = test(model, params)
    torch.cuda.empty_cache()
