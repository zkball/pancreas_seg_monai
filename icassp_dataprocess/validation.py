import os

import math
import pandas as pd
import torch
import cv2
import numpy as np
import pytorch_lightning as pl
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from data.Logger import  ImageLogger_2D
from data.MedicalDataset import  MedicalDataset_2D_tumor_area_rank
from models.Pipeline import  AttentionUnet_Pipeline, our_UNetPlusPlus, UNet

from torch.utils.data import DataLoader
# import bchlib
from models.loss import f_score

import warnings
warnings.filterwarnings("ignore")


def visualize_prediction(images, save_dir, index, test_sample_name,batch_size):
    sample_save_path = os.path.join(save_dir, test_sample_name)
    os.makedirs(sample_save_path,exist_ok=True)

    input_image = images['input_image_batch']
    gt_mask = images['input_label_batch']
    predict_label = images['predict_label_batch']

    images = torch.cat([input_image, gt_mask, predict_label], dim=-2)

    # print(images[k].shape) #torch.Size([4, 1, 768, 256])
    images = torch.clamp(images, -1., 1.)
    grid = torchvision.utils.make_grid(images, nrow=batch_size)

    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    # print(grid.shape)
    grid = (grid * 255).astype(np.uint8)

    filename = test_sample_name + "_gs-{:06}_batch.png".format(index)
    path = os.path.join(sample_save_path, filename)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)



def dice_score(preds, targets, threshold=0.8, smooth=1e-6):
    preds = (preds > threshold).float()  # Apply threshold
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()



# Function to compute TP, FP, and FN
def compute_confusion_elements(preds, targets):
    TP = (preds * targets).sum().item()  # True Positives
    FP = (preds * (1 - targets)).sum().item()  # False Positives
    FN = ((1 - preds) * targets).sum().item()  # False Negatives
    return TP, FP, FN


def sklearn_f1_score(preds, targets, threshold=0.5, smooth=1e-6):
    from sklearn.metrics import f1_score
    # Flatten the 2D arrays
    preds_flat = preds.flatten().numpy()
    targets_flat = targets.flatten().numpy()

    # Apply threshold to get binary predictions
    binary_preds_flat = (preds_flat > threshold).astype(np.float32)

    # Compute the F1 score
    f1 = f1_score(targets_flat, binary_preds_flat)
    return f1



def binary_iou(predict_mask,gt_mask):
    preds_flat = predict_mask.flatten().numpy()
    targets_flat = gt_mask.flatten().numpy()

    # Apply threshold to get binary predictions
    binary_preds_flat = (preds_flat > 0.5).astype(np.float32)
    assert (len(binary_preds_flat.shape) == len(targets_flat.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(binary_preds_flat, targets_flat)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(binary_preds_flat + targets_flat > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou

def compute_metrics(predictions):
    dice_list = []
    f1_list = []
    dice_list_2 = []
    iou_score_list = []

    for batch_index, batch_sample in enumerate(predictions):

        gt_mask = batch_sample['input_label_batch'].squeeze()
        predict_mask = batch_sample['predict_label_batch'].squeeze()

        if torch.any(gt_mask != 0): #has tumor
            dice = dice_score(predict_mask, gt_mask)
            dice_list_2.append(dice)


        # predict_mask = (predict_mask * 255).astype(np.uint8)

        # Compute Dice coefficient
        # dice = dice_coefficient(gt_mask, predict_mask)
        f1_score = sklearn_f1_score(predict_mask,gt_mask)
        dice = dice_score(predict_mask, gt_mask)
        iou_score = binary_iou(predict_mask, gt_mask)
        iou_score_list.append(iou_score)
        dice_list.append(dice)
        f1_list.append(f1_score)

    print(dice_list_2)

    # print(dice_list)
    return sum(dice_list)/ len(dice_list), sum(dice_list_2)/ len(dice_list_2), sum(f1_list)/ len(f1_list), sum(iou_score_list)/len(iou_score_list)





if __name__ == '__main__':
    # 1. Load checkpoint and model to do the inference on validation data
    # model = sd_seg_model(use_timestep=True)
    # model = AttentionUnet_Pipeline()
    model = our_UNetPlusPlus()
    # model = Locator()
    # model = haar_UNetPlusPlus()
    # model = UNet()
    model_name = 'validadion_model'
    ckpt_path = '/groupshare_1/wenqi_code/screen-watermark/our_unet++_weights_2024-07-28/model-epoch=15-val_loss=0.0399.ckpt'
    weights = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(weights['state_dict'])


    # 2. set the parameters
    batch_size = 1
    logger_freq = 1
    samples_dice_sum = 0
    samples_dice_sum_2 = 0
    samples_f1_sum = 0
    sample_count = 0
    samples_iou_sum = 0
    val_save_dir = f'/groupshare_1/wenqi_code/screen-watermark/{model_name}'
    os.makedirs(val_save_dir, exist_ok=True)



    # 3. do the inference on each validation sample
    f = open("/groupshare_1/cancer_center/cropped_origin_data_256_256_z/val_samples.txt", "r", encoding="UTF-8")
    saved_predict_info = []
    for line in f:
        sample_count += 1
        print(line.strip("\n"))
        val_sample_name = line.strip("\n")

        has_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_val_v2/has_tumor'
        no_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_val_v2/no_tumor'
        traindata_test_dataset = MedicalDataset_2D_tumor_area_rank(has_tumor_dir=has_tumor_dir, no_tumor_dir=no_tumor_dir, type='traindata_test',val_sample_name=val_sample_name)
        traindata_test_dataloader = DataLoader(traindata_test_dataset, num_workers=2, batch_size=batch_size, shuffle=False,
                                      persistent_workers=True, pin_memory=True)

        trainer = pl.Trainer(devices=1, accelerator='gpu', precision=32)
        predictions = trainer.predict(model, traindata_test_dataloader) # has length B, each prediction is a dictionary

        for batch_index, batch_sample in enumerate(predictions):
            # print(batch_sample)
            visualize_prediction(batch_sample, val_save_dir, batch_index, val_sample_name, batch_size)

        # 4. compute dice metric
        sample_dice, sample_dice_2, sample_f1, iou_score = compute_metrics(predictions)
        print(val_sample_name, " dice: ", str(sample_dice))
        print(val_sample_name, " dice2: ", str(sample_dice_2))
        print(val_sample_name, " f1: ", str(sample_f1))
        print(val_sample_name, " iou: ", str(iou_score))
        samples_dice_sum = samples_dice_sum + sample_dice
        samples_dice_sum_2 = samples_dice_sum_2 + sample_dice_2
        samples_f1_sum = samples_f1_sum + sample_f1
        samples_iou_sum = samples_iou_sum + iou_score

        saved_predict_info.append({
            "val_sample_name": val_sample_name,
            "dice": sample_dice,
            "dice only for tumor area": sample_dice_2,
            "f1": sample_f1,
            "iou_score": iou_score,
        })


    val_mean_dice = samples_dice_sum / sample_count
    val_mean_dice_2 = samples_dice_sum_2 / sample_count
    val_mean_f1 = samples_f1_sum / sample_count
    val_mean_iou = samples_iou_sum / sample_count

    print(f"validation set has {sample_count} samples, the mean dice is {val_mean_dice}, mean dice only tumor is {val_mean_dice_2}, mean f1 score is {val_mean_f1}, mean iou is {val_mean_iou}")

    df = pd.DataFrame(saved_predict_info)
    # Save DataFrame to CSV
    df.to_csv(os.path.join(val_save_dir,f"validation_metrics_{model_name}.csv"), index=False)


