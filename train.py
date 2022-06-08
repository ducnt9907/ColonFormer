import argparse
import logging
import os
import random
import sys
import time
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from collections import OrderedDict
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352)) 

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
    
epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
        else:
            lr_scheduler.step()

        for rate in size_rates: 
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(trainsize_init*rate/32)*32)
            images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            map4, map3, map2, map1 = model(images)
            map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
            # ---- metrics ----
            dice_score = dice_m(map4, gts)
            iou_score = iou_m(map4, gts)
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, args.batchsize)
                dice.update(dice_score.data, args.batchsize)
                iou.update(iou_score.data, args.batchsize)

        # ---- train visualization ----
        if i == total_step:
            print('{} Training Epoch [{:03d}/{:03d}], '
                    '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
                    format(datetime.now(), epoch, num_epochs,\
                            loss_record.show(), dice.show(), iou.show()))

    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)

    log = OrderedDict([
        ('loss', loss_record.show()), ('dice', dice.show()), ('iou', iou.show()),
    ])

    return log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='ConlonFormerB3')
    args = parser.parse_args()

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")


    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'dice', 'iou', 'val_loss', 'val_dice', 'val_iou'
    ])
    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob(f'{args.train_path}/images/*')
    train_mask_paths = glob(f'{args.train_path}/masks/*')
    train_img_paths.sort()
    train_mask_paths.sort()

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    total_step = len(train_loader)

    model = UNet(backbone=dict(
                    type=f'mit_{args.backbone}',
                    style='pytorch'), 
                decode_head=dict(
                    type='UPerHead',
                    in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    align_corners=False,
                    decoder_params=dict(embed_dim=768),
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
                neck=None,
                auxiliary_head=None,
                train_cfg=dict(),
                test_cfg=dict(mode='whole'),
                pretrained=f'pretrained/mit_{args.backbone}.pth').cuda()

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.nit_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=len(train_loader)*args.num_epochs,
                                        eta_min=args.init_lr/1000)

    start_epoch = 1
    ckpt_path = ''
    if ckpt_path != '':
        log = pd.read_csv(ckpt_path.replace('last.pth', 'log.csv'))
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("#"*20, f"Start Training", "#"*20)
    for epoch in range(start_epoch, args.num_epochs+1):
        train_log = train(train_loader, model, optimizer, epoch, lr_scheduler, args)

        log_tmp = pd.Series([epoch, optimizer.param_groups[0]["lr"], 
                train_log['loss'].item(), train_log['dice'].item(), train_log['iou'].item(),  
        ], index=['epoch', 'lr', 'loss', 'dice', 'iou'])
        log = log.append(log_tmp, ignore_index=True)
        log.to_csv(f'snapshots/{train_save}/log.csv', index=False)