import os
import sys
from typing import Any

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.cuda.amp import autocast, GradScaler


from models.loss import RankedWeightedLoss

sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from einops import repeat

from models.loss import DiceLoss, FocalLoss
# from monai.losses import DiceLoss, MaskedDiceLoss
import segmentation_models_pytorch as smp

from monai.networks.nets import UNet
from models.attunet import AttentionU_Net, DiceBCELoss



class UNet(pl.LightningModule):
    def __init__(self, lr=2e-5, ch=128):
        super(UNet, self).__init__()
        self.lr = lr

        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['tumor']
        ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation
        # 使用unet++模型
        self.unet = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            in_channels=1
        )


        self.dice_loss = DiceLoss()
        self.sigmoid = nn.Sigmoid()
        self.weighted_loss_fn = RankedWeightedLoss(self.dice_loss, max_weight=2.0)

    def forward(self, input_image):

        predict_mask = self.unet(input_image)
        predict_mask = self.sigmoid(predict_mask)

        # print("predict mask has shape", predict_mask.shape)
        # pred_loc = localized
        return predict_mask

    def train_loss(self, gt_mask, predict_mask, rank_ratio):
        loss = self.weighted_loss_fn(predict_mask, gt_mask,rank_ratio)
        loss_dict = {
            'train_loss': loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def val_loss(self, gt_mask, predict_mask, rank_ratio):
        loss = self.weighted_loss_fn(predict_mask, gt_mask,rank_ratio)
        loss_dict = {
            'val_loss': loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_mask = self.forward(input_image)
        # gt_secret = gt_secret.reshape(-1, 1)
        loss, loss_dict = self.train_loss(gt_mask, predict_mask, rank_ratio)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # self.log('train_dice_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_mask = self.forward(input_image)
        # gt_secret = gt_secret.reshape(-1, 1)
        loss, loss_dict = self.val_loss(gt_mask, predict_mask, rank_ratio)
        # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        log = dict()
        # print("len batch",len(batch))
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_label = self.forward(input_image)
        # record = torch.cat([input_image,gt_mask, predict_label], dim=-2)
        log['input_image_batch'] = input_image
        log['input_label_batch'] = gt_mask
        log['predict_label_batch'] = predict_label
        # log['val_record'] = record
        # print("predict step complete")
        return log

    def configure_optimizers(self):
        lr = self.lr
        params_list = list(self.unet.parameters())
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=(0.5, 0.9))
        return [optimizer], []

    @torch.no_grad()
    def log_images(self, batch, split, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_mask = self.forward(input_image)
        record = torch.cat([input_image, gt_mask, predict_mask], dim=-2)
        log['record'] = record
        return log


class our_UNetPlusPlus(pl.LightningModule):
    def __init__(self, lr=2e-5, ch=128):
        super(our_UNetPlusPlus, self).__init__()
        self.lr = lr

        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['tumor']
        ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation
        # 使用unet++模型
        self.decoder = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            in_channels=1
        )


        self.dice_loss = DiceLoss()
        self.sigmoid = nn.Sigmoid()
        self.weighted_loss_fn = RankedWeightedLoss(self.dice_loss, max_weight=2.0)

    def forward(self, input_image):

        predict_mask = self.decoder(input_image)
        predict_mask = self.sigmoid(predict_mask)

        # print("predict mask has shape", predict_mask.shape)
        # pred_loc = localized
        return predict_mask

    def train_loss(self, gt_mask, predict_mask, rank_ratio):
        loss = self.weighted_loss_fn(predict_mask, gt_mask, rank_ratio)
        loss_dict = {
            'train_loss': loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def val_loss(self, gt_mask, predict_mask, rank_ratio):
        loss = self.weighted_loss_fn(predict_mask, gt_mask, rank_ratio)
        loss_dict = {
            'val_loss': loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_mask = self.forward(input_image)
        # gt_secret = gt_secret.reshape(-1, 1)
        loss, loss_dict = self.train_loss(gt_mask, predict_mask, rank_ratio)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # self.log('train_dice_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_mask = self.forward(input_image)
        # gt_secret = gt_secret.reshape(-1, 1)
        loss, loss_dict = self.val_loss(gt_mask, predict_mask, rank_ratio)
        # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        log = dict()
        # print("len batch",len(batch))
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_label = self.forward(input_image)
        # record = torch.cat([input_image,gt_mask, predict_label], dim=-2)
        log['input_image_batch'] = input_image
        log['input_label_batch'] = gt_mask
        log['predict_label_batch'] = predict_label
        # log['val_record'] = record
        # print("predict step complete")
        return log

    def configure_optimizers(self):
        lr = self.lr
        params_list = list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=(0.5, 0.9))
        return [optimizer], []

    @torch.no_grad()
    def log_images(self, batch, split, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_mask = self.forward(input_image)
        record = torch.cat([input_image, gt_mask, predict_mask], dim=-2)
        log['record'] = record
        return log


class AttentionUnet_Pipeline(pl.LightningModule):
    def __init__(self,lr=1e-3):
        super(AttentionUnet_Pipeline, self).__init__()
        self.AttentionUnet = AttentionU_Net(
            in_ch= 1,
            out_ch=1
        )
        self.lr = lr
        self.loss_fn = DiceBCELoss()
        self.sigmoid = nn.Sigmoid().eval()
        self.weighted_loss_fn = RankedWeightedLoss(self.loss_fn, max_weight=2.0)

    def forward(self, input_image):
        predict_mask = self.AttentionUnet(input_image)
        return predict_mask

    def train_loss(self, pred_mask, gt_mask,rank_ratio):
        loss = self.weighted_loss_fn(pred_mask, gt_mask,rank_ratio)
        # loss = self.loss_fn(pred_mask, gt_mask)
        loss_dict = {
            'train_loss': loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        pred_mask = self(input_image)
        loss, loss_dict = self.train_loss(pred_mask, gt_mask,rank_ratio)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def val_loss(self, pred_mask, gt_mask,rank_ratio):

        loss = self.weighted_loss_fn(pred_mask, gt_mask, rank_ratio)
        # loss = self.loss_fn(pred_mask, gt_mask)
        loss_dict = {
            'val_loss': loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def validation_step(self, batch, batch_idx):
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        pred_mask = self(input_image)
        loss, loss_dict = self.val_loss(pred_mask, gt_mask, rank_ratio)
        # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    # def test_step(self, batch, batch_idx):
    #     log = dict()
    #     input_image, gt_loc , rank_ratio= batch
    #     predict_label = self(input_image)
    #     record = torch.cat([input_image, gt_loc, predict_label], dim=-2)
    #     # gt_secret = gt_secret.reshape(-1, 1)
    #     loss, loss_dict = self.val_loss(gt_loc, predict_label)
    #     self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
    #     # log['val_training_data_record'] = record
    #     # print("test training data complete")
    #     return loss


    def predict_step(self, batch, batch_idx):
        log = dict()
        # print("len batch",len(batch))
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        predict_label = self(input_image)
        # record = torch.cat([input_image,gt_mask, predict_label], dim=-2)
        log['input_image_batch'] = input_image
        log['input_label_batch'] = gt_mask
        log['predict_label_batch'] = predict_label
        # log['val_record'] = record
        # print("predict step complete")
        return log

    def configure_optimizers(self):
        lr = self.lr
        optimizer = torch.optim.Adam(self.AttentionUnet.parameters(),lr=lr)
        return [optimizer], []

    def log_images(self, batch, split, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        input_image, gt_mask, rank_ratio, slice_index,condition_mask, condition_image = batch
        pred_mask = self(input_image)
        input_image = torch.cat([input_image, gt_mask, pred_mask], dim=-2)
        log['record'] = input_image
        return log


if __name__ == '__main__':
    model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=1,
            in_channels=1
        )

    x = torch.randn((1,1,256,256))
    y = model(x)
    print(y.shape)
