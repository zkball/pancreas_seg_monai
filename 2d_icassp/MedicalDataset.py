import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
import nibabel as nib
import torch.nn.functional as F
from torchvision import transforms

np2tensor = lambda x: torch.from_numpy(np.array(x).astype(np.float32)).permute((2, 0, 1))

class HistogramEqualization:
    """Apply histogram equalization to enhance contrast."""

    def __call__(self, img):

        # Convert image to 8-bit
        img = np.array(img)
        # Apply histogram equalization
        equalized_img = cv2.equalizeHist(img)
        return equalized_img


class CLAHE:
    """Apply CLAHE to enhance contrast of a grayscale image."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        # Apply CLAHE
        clahe_img = clahe.apply(img_np)

        # Convert back to PIL Image
        return clahe_img

class MedicalDataset_3D(Dataset):
    def __init__(self, image_dir, mask_dir, predict_dir):
        super(MedicalDataset_3D, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.predict_dir = predict_dir

        self.image_sample_dirs = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir)]
        self.mask_sample_dirs = [os.path.join(self.mask_dir, filename) for filename in os.listdir(self.mask_dir)]
        self.predict_sample_dirs = [os.path.join(self.predict_dir, filename) for filename in os.listdir(self.predict_dir)]

    def extract_slice_index(self, filename):
        slice_name = filename.split('/')[-1][:-4]
        slice_index = slice_name.split('_')[2]
        # print(filename, slice_index)
        return int(slice_index)

    def load_sample(self, sample_dir):
        slices = []
        for f in sorted(os.listdir(sample_dir), key=self.extract_slice_index, reverse=False): # Assuming slice1.png to slice170.png
            slice_path = os.path.join(sample_dir, f)
            image = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            slices.append(image)

        # Stack slices along the depth dimension (axis=0)
        volume = np.stack(slices, axis=0)
        return volume  # This will have shape (170, 256, 256)

    def __getitem__(self, index):
        image_sample_dir = self.image_sample_dirs[index]
        mask_sample_dir = self.mask_sample_dirs[index]
        predict_sample_dir = self.predict_sample_dirs[index]

        image_volume = self.load_sample(image_sample_dir)
        mask_volume = self.load_sample(mask_sample_dir)
        predict_volume = self.load_sample(predict_sample_dir)

        multi_channel_volume = np.stack([image_volume, predict_volume], axis=0)



class MedicalDataset_2D_tumor_area_rank(Dataset):
    def __init__(self, has_tumor_dir, no_tumor_dir, type=None, val_sample_name=None, condition = False, data_augmentation = False):
        super(MedicalDataset_2D_tumor_area_rank, self).__init__()
        self.has_tumor_dir = has_tumor_dir
        self.no_tumor_dir = no_tumor_dir
        self.type = type
        self.condition = condition
        self.data_augmentation = data_augmentation


        self.tumor_image_path = os.path.join(has_tumor_dir, 'image')
        self.tumor_mask_path = os.path.join(has_tumor_dir, 'mask')
        self.no_tumor_image_path = os.path.join(no_tumor_dir, 'image')
        self.no_tumor_mask_path = os.path.join(no_tumor_dir, 'mask')

        if type == 'train' or type == 'val':

            tumor_images = sorted([f for f in os.listdir(self.tumor_image_path) if f.endswith('.png')])
            tumor_images = [os.path.join(self.tumor_image_path, filename) for filename in tumor_images]

            tumor_masks = sorted([f for f in os.listdir(self.tumor_mask_path) if f.endswith('.png')])
            tumor_masks = [os.path.join(self.tumor_mask_path, filename) for filename in tumor_masks]

            no_tumor_images = sorted([f for f in os.listdir(self.no_tumor_image_path) if f.endswith('.png')])
            no_tumor_images = [os.path.join(self.no_tumor_image_path, filename) for filename in no_tumor_images]

            no_tumor_masks = sorted([f for f in os.listdir(self.no_tumor_mask_path) if f.endswith('.png')])
            no_tumor_masks = [os.path.join(self.no_tumor_mask_path, filename) for filename in no_tumor_masks]

            print(len(tumor_images), len(no_tumor_images[:len(tumor_images)]))
            self.image_list = tumor_images + no_tumor_images[:len(tumor_images)]
            self.mask_list = tumor_masks + no_tumor_masks[:len(tumor_masks)]
            self.tumor_image_mount = len(tumor_images)

        elif type == 'traindata_test':

            tumor_images = sorted([f for f in os.listdir(self.tumor_image_path) if
                                   f.endswith('.png') and f.split('_')[0] == val_sample_name])
            tumor_images = [os.path.join(self.tumor_image_path, filename) for filename in tumor_images]

            tumor_masks = sorted([f for f in os.listdir(self.tumor_mask_path) if
                                  f.endswith('.png') and f.split('_')[0] == val_sample_name])
            tumor_masks = [os.path.join(self.tumor_mask_path, filename) for filename in tumor_masks]

            no_tumor_images = sorted([f for f in os.listdir(self.no_tumor_image_path) if
                                      f.endswith('.png') and f.split('_')[0] == val_sample_name])
            no_tumor_images = [os.path.join(self.no_tumor_image_path, filename) for filename in no_tumor_images]

            no_tumor_masks = sorted([f for f in os.listdir(self.no_tumor_mask_path) if
                                     f.endswith('.png') and f.split('_')[0] == val_sample_name])
            no_tumor_masks = [os.path.join(self.no_tumor_mask_path, filename) for filename in no_tumor_masks]

            images = tumor_images + no_tumor_images
            masks = tumor_masks + no_tumor_masks

            self.image_list = sorted(images, key=self.extract_slice_index, reverse=False)
            self.mask_list = sorted(masks, key=self.extract_slice_index, reverse=False)

            # print("val images")
            print(self.image_list)


    def __getitem__(self, index):
        image_file_name = self.image_list[index]
        # image = cv2.imread(image_file_name,-1)
        image = Image.open(image_file_name).convert("L")
        if self.data_augmentation == True and index < self.tumor_image_mount:
            image = self.data_augmentation_operation(image, "image")

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
        image = image / 255
        image = np2tensor(image)


        mask_file_name = self.mask_list[index]
        if image_file_name.split('/')[-1] != mask_file_name.split('/')[-1]:
            print("ERROR!!!!!!!!")
            print(image_file_name)
            print(mask_file_name)
        # mask = cv2.imread(mask_file_name, -1)
        mask = Image.open(mask_file_name).convert("L")
        if self.data_augmentation == True and index < self.tumor_image_mount:
            mask = self.data_augmentation_operation(mask, "mask")
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask / 255
        mask = np2tensor(mask)

        sample_name = image_file_name.split('/')[-1][:-4]

        tumor_area_rank = sample_name.split('_')[4]
        slice_sum = sample_name.split('_')[5]
        slice_index = sample_name.split('_')[2]
        # print(tumor_area_rank)

        slice_index_ratio = self.interpolate_timesteps(int(slice_index), int(slice_sum))
        # print(slice_index_ratio)

        if self.condition == False:
            # print(image.shape)
            return image, mask, round(int(tumor_area_rank) / int(slice_sum), 4), slice_index_ratio, 0, 0

        else:
            # print(self.tumor_mask_path)
            largest_tumor_masks_name = [f for f in os.listdir(self.tumor_mask_path) if f.endswith('.png') and f.split('_')[0] == sample_name.split('_')[0]  and f.split('_')[-2] == '0'][0]
            # print(largest_tumor_masks_name)
            largest_tumor_mask = cv2.imread(os.path.join(self.tumor_mask_path, largest_tumor_masks_name))
            largest_tumor_mask = np2tensor(largest_tumor_mask).squeeze()


            largest_tumor_image = cv2.imread(os.path.join(self.tumor_image_path, largest_tumor_masks_name))
            largest_tumor_image = np2tensor(largest_tumor_image).squeeze()


            return image, mask, round(int(tumor_area_rank) / int(slice_sum), 4), slice_index_ratio, largest_tumor_mask, largest_tumor_image

    def __len__(self):
        return len(self.image_list)

    def data_augmentation_operation(self, x, type):
        augmentations = {
            "horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            "vertical_flip": transforms.RandomVerticalFlip(p=1.0),  # Always flip
            "rotation_90": transforms.RandomRotation((90, 90)),  # Rotate by 90 degrees
            "rotation_180": transforms.RandomRotation((180, 180)),  # Rotate by 180 degrees
            "rotation_270": transforms.RandomRotation((270, 270)),  # Rotate by 270 degrees
            "brightness": transforms.ColorJitter(brightness=0.5),  # Adjust brightness
            "contrast": transforms.ColorJitter(contrast=0.5),  # Adjust contrast
            "HE": transforms.Compose([
                HistogramEqualization(),
                transforms.ToPILImage(),
            ]),
            "CLACHE": transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
                CLAHE(),
                transforms.ToPILImage(),
            ]),
            "origin":transforms.RandomHorizontalFlip(p=0),
        }

        aug_name, augmentation = random.choice(list(augmentations.items()))

        if type == 'image':
            output = augmentation(x)
        else:
            if aug_name in ['brightness', 'contrast', 'HE', 'CLACHE']:
                output = x
            else:
                output = augmentation(x)

        return output

    def extract_slice_index(self, filename):
        slice_name = filename.split('/')[-1][:-4]
        slice_index = slice_name.split('_')[2]
        # print(filename, slice_index)
        return int(slice_index)

    def interpolate_timesteps(self, slice_indices, total_slices, T=1000):
        # Normalize slice indices to range [0, 1]
        normalized_indices = slice_indices / total_slices

        # Map to timestep range [1, T]
        mapped_timesteps = 1 + normalized_indices * (T - 1)

        # Ensure contiguous timesteps by interpolating mapped timesteps
        interpolated_timesteps = np.round(mapped_timesteps).astype(int)

        return interpolated_timesteps



if __name__ == '__main__':
    train_has_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_train_v2_aug/has_tumor'
    train_no_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_train_v2_aug/no_tumor'
    val_has_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_val_v2_aug/has_tumor'
    val_no_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_val_v2_aug/no_tumor'
    train_dataset = MedicalDataset_2D_tumor_area_rank(train_has_tumor_dir, train_no_tumor_dir, type='train')
    # val_dataset = MedicalDataset_2D_tumor_area_rank(val_has_tumor_dir, val_no_tumor_dir, type='val')
    input_image, gt_mask, rank_ratio, slice_index, largest_tumor_mask, largest_tumor_image = train_dataset.__getitem__(1800)
    print(input_image, largest_tumor_image)

    # Check if all channels are the same
    # channels_are_equal = torch.equal(largest_tumor_mask[0], largest_tumor_mask[1]) and torch.equal(largest_tumor_mask[1], largest_tumor_mask[2])
    #
    # if channels_are_equal:
    #     print("All three channels are identical.")
    # else:
    #     print("The channels are not identical.")





