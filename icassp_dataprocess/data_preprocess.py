import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
# import cv2
# print(cv2.__version__)
import torch.nn as nn
import torchvision
import nibabel as nib
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
## max~min: 3071~-1024
## liquid: -10~10
## air: -1000
## fat: -10~-90
## tissue: 20-50
## bone: >300


def window_adjust(data, w_width=250, w_center=50):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data.copy()
    data_adjusted[data < val_min] = val_min
    data_adjusted[data > val_max] = val_max

    return data_adjusted

def get_Z_location(image_dir, csv_path='/raid/datasets/origin/origin_images_info.csv'):
    remove_set = None
    with open("/raid/datasets/origin/delete_samples.txt", "r") as f:
            files = f.readlines()
            remove_set = set([item.strip()+f"{'A' if 'A' in image_dir else 'V'}.nii.gz" for item in files])

    df = pd.read_csv(csv_path, sep=',')
    Z_location = {}
    for index, row in df.iterrows():
        z_col = row['Mask Z location'][1:-1]
        z_col = z_col.split(",")
        start_idx, end_idx = int(z_col[0]), int(z_col[1])
        interval = end_idx+1-start_idx
        if interval<=0:
            print(f"[Error in mask] {row}. skip..")
        else:
            Z_location[row['Filename']] = {'start': start_idx, 'end': end_idx, 'interval': interval}
    
    return Z_location, remove_set

def nii_gz_2_png_save_negative_slices(source_dir, sample_name, save_dir, type, valid_idx):
    res = []
    path = os.path.join(source_dir, sample_name)
    original_img = nib.load(path)

    original_np = original_img.get_fdata()
    if type == 'image':
        original_np = window_adjust(original_np)

    # import pdb;pdb.set_trace()
    positive_slices= set(valid_idx)

    for index in range(original_np.shape[2]):
        sliced_np = original_np[:, :, index:index+1]

        # Normalize data to 0-255 for image saving
        if index in positive_slices:
            # print(f"max==min in {path}. skip!")
            continue
        else:
            # norm_slice = (
            #         (slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(
            #     np.uint8)

            # # Save slice as PNG image
            # norm_slice = Image.fromarray(norm_slice)
            # norm_slice.save(os.path.join(save_dir, f'{sample_name}_{index}.png'))
           
            ## only get index according to mask
            new_img = nib.Nifti1Image(sliced_np, affine=original_img.affine, header=original_img.header)
            new_img.header['dim'][3] = sliced_np.shape[2]

            nib.save(new_img, os.path.join(save_dir, f'{sample_name}_{index}.nii.gz'))
            res.append(index)
    
    return res

def nii_gz_2_png(source_dir, sample_name, save_dir, type, valid_idx: list=None, no_saving=False):

    res = []
    path = os.path.join(source_dir, sample_name)
    original_img = nib.load(path)

    original_np = original_img.get_fdata()
    if type == 'image':
        original_np = window_adjust(original_np)


    if valid_idx is None:
        valid_idx = range(original_np.shape[2])

    for index in valid_idx:
        sliced_np = original_np[:, :, index:index+1]

        # Normalize data to 0-255 for image saving
        if sliced_np.max()==sliced_np.min():
            # print(f"max==min in {path}. skip!")
            continue
        else:
            # norm_slice = (
            #         (slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(
            #     np.uint8)

            # # Save slice as PNG image
            # norm_slice = Image.fromarray(norm_slice)
            # norm_slice.save(os.path.join(save_dir, f'{sample_name}_{index}.png'))
            if not no_saving:
                ## only get index according to mask
                new_img = nib.Nifti1Image(sliced_np, affine=original_img.affine, header=original_img.header)
                new_img.header['dim'][3] = sliced_np.shape[2]

                nib.save(new_img, os.path.join(save_dir, f'{sample_name}_{index}.nii.gz'))
            res.append(index)
    
    return res


# def nii_gz_crop_slices(source_dir, sample_name, save_dir, Z_location):

#     res = []
#     path = os.path.join(source_dir, sample_name)
#     original_img = nib.load(path)

#     original_np = original_img.get_fdata()

#     if valid_idx is None:
#         valid_idx = range(original_np.shape[2])


#     new_img = nib.Nifti1Image(sliced_np, affine=original_img.affine, header=original_img.header)
#     new_img.header['dim'][3] = sliced_np.shape[2]

#     nib.save(new_img, os.path.join(save_dir, f'{sample_name}'))
          


if __name__ == '__main__':
    import argparse
    # with tempfile.TemporaryDirectory() as tempdir:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--get_stat",
        type=str,
        default="3d",
        required=False,
        help="Get stat",
    )
    parser.add_argument(
        "--gen_2d",
        action="store_true",
        required=False,
        help="Generate 2d slices in .nii.gz format",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        required=False,
        help="No mask",
    )
    parser.add_argument(
        "--crop_slice_dim",
        action="store_true",
        required=False,
        help="Cropping in slice dim to reserve roi only",
    )
    args = parser.parse_args()
    
    # An example for loading nii.gz file
    image_dir = "/raid/datasets/origin/images_A"
    mask_dir = "/raid/datasets/origin/Mask_A"
    
    Z_location, remove_set = get_Z_location(image_dir)

    ## main functions
    if args.gen_2d:
        print("Generating 2d data. please wait...")
        image_save_dir = '/raid/datasets/origin_negative_2d_slices/images_A'
        mask_save_dir = '/raid/datasets/origin_negative_2d_slices/Mask_A'
        if image_save_dir is not None:
            os.makedirs(image_save_dir, exist_ok=True)
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)

        for sample_name in tqdm(os.listdir(image_dir)):
            # import pdb;pdb.set_trace()
            valid_index = None
            if (not sample_name not in remove_set):
                if (mask_dir is not None):
                    valid_index = nii_gz_2_png(mask_dir, sample_name, mask_save_dir, 'mask', no_saving=True)
                nii_gz_2_png_save_negative_slices(image_dir, sample_name, image_save_dir, 'image', valid_idx=valid_index)


    if args.get_stat:
        print("Get stats. please wait...")
        print("do nothing.")


    if args.crop_slice_dim:
        print("Cropping in slice dimension. please wait...")
        print("do nothing.")

        # base_dir = '/raid/datasets/origin_slice_cropped'
        # os.makedirs(base_dir, exist_ok=True)
        # image_save_dir = os.path.join(base_dir, 'images_A')
        # os.makedirs(image_save_dir, exist_ok=True)

        # for sample_name in tqdm(os.listdir(image_dir)):
        #     if sample_name not in remove_set:
        #         nii_gz_crop_slices(mask_dir, sample_name, image_save_dir)




    





