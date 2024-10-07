import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
import nibabel as nib
import torch.nn.functional as F


def window_adjust(data, w_width=250, w_center=50):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data.copy()
    data_adjusted[data < val_min] = val_min
    data_adjusted[data > val_max] = val_max

    return data_adjusted



def nii_gz_2_png(source_dir, sample_name, save_dir, type):
    path = os.path.join(source_dir, sample_name)
    obj = nib.load(path)

    data = obj.get_fdata()
    if type == 'image':
        data = window_adjust(data)

    for index in range(data.shape[2]):
        slice = data[:, :, index]

        # Normalize data to 0-255 for image saving
        norm_slice = (
                (slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(
            np.uint8)

        # Save slice as PNG image
        norm_slice = Image.fromarray(norm_slice)
        norm_slice.save(os.path.join(save_dir, f'{sample_name}_{index}.png'))


if __name__ == '__main__':
    # An example for loading nii.gz file
    image_dir = "/groupshare_1/cancer_center/origin/images_A"
    mask_dir = "/groupshare_1/cancer_center/origin/Mask_A"
    image_save_dir = '/groupshare_1/cancer_center/2d_dataset/images_A'
    mask_save_dir = '/groupshare_1/cancer_center/2d_dataset/Mask_A'
    os.makedirs(image_save_dir)
    os.makedirs(mask_save_dir)

    for sample_name in os.listdir(image_dir):

        nii_gz_2_png(image_dir, sample_name, image_save_dir, 'image')
        nii_gz_2_png(mask_dir, sample_name, mask_save_dir, 'mask')





