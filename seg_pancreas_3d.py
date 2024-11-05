# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import monai.losses
import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import monai
from monai.data import (
    # ImageDataset, 
    ArrayDataset, create_test_image_3d, decollate_batch, DataLoader
)
from monai.inferers import sliding_window_inference
from monai.metrics import (
    DiceMetric, MeanIoU
)
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    # RandSpatialCrop,
    ScaleIntensity,
)
from monai.data.image_reader import (
    ImageReader,
    ITKReader,
    NibabelReader,
    NrrdReader,
    NumpyReader,
    PILReader,
    PydicomReader,
)
from monai.visualize import plot_2d_or_3d_image

from tqdm import tqdm
import torch.distributed as dist
import ignite.distributed as idist
from ignite.metrics import Accuracy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils import stitch_images, postprocess
import pandas as pd
from icassp_dataprocess.data_preprocess import get_Z_location

from modified_monai.dataset.image_dataset import (
    ImageDataset
)

from modified_monai.metrics.dce import DCE
from seg_pancreas import seg_pancreas_base
class seg_pancreas_3d(seg_pancreas_base):

    def __init__(
        self, args
    ) -> None:
        super().__init__(args=args)

        """
        task-related variables
        """
        self.batch_size = 4
        self.class_labels = None
        self.learning_rate = 5e-3

        self.num_epoch  = 500
        self.num_epoch = self.num_epoch if self.args.eval is False else 1

        self.image_dir='/raid/datasets/origin/images_A'
        self.mask_dir='/raid/datasets/origin/Mask_A'
        # self.negative_dir='/raid/datasets/origin_negative_2d_slices/images_A'
        self.file_format=".nii.gz"
        self.roi_size = (512, 512, 128)

        print("dir paths set")

        self.image_test_dir='/raid/datasets/207file/images_A'

        """
        prepare dataset and model
        """
        self.prepare_everything()

    def prepare_data(self):
        test_loader, test_ds = None, None
        negative_train_ds, negative_train_sampler, negative_train_loader = None, None, None
        
        from modified_monai.transforms.rand_slice_crop import (
            RandSpatialCrop
        )

        Z_location, remove_set = get_Z_location(self.image_dir)

        # import pdb;pdb.set_trace()
        filenames = sorted([os.path.basename(f) for f in glob(os.path.join(self.mask_dir, f"*{self.file_format}"))])
        filenames = list(filter(lambda x: x in set(Z_location.keys()) and x not in remove_set, filenames))

        images = [os.path.join(self.image_dir, filename) for filename in filenames]
        segs = [os.path.join(self.mask_dir, filename) for filename in filenames]
        images_train, images_val, segs_train, segs_val = images[:-40], images[-40:], segs[:-40], segs[-40:]

        train_imtrans = Compose(
            [
                # ScaleIntensity(),
                EnsureChannelFirst(),
                # RandSpatialCrop((512, 512, 64), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=2),
            ]
        )
        train_segtrans = Compose(
            [
                EnsureChannelFirst(),
                # RandSpatialCrop((512, 512, 64), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=2),
            ]
        )
        val_imtrans = Compose(
            [
                # ScaleIntensity(), 
                EnsureChannelFirst(),
                RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=2),
            ]
        )
        val_segtrans = Compose(
            [
                EnsureChannelFirst(),
                RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=2),
            ]
        )

        if not self.args.eval:
            train_ds = ImageDataset(images_train, segs_train, transform=train_imtrans, seg_transform=train_imtrans)
            train_sampler = DistributedSampler(train_ds)
            # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.batch_size,
                pin_memory=True,
                sampler=train_sampler,
            )
            # create a validation data loader
            val_ds = ImageDataset(images_val, segs_val, transform=val_imtrans, seg_transform=val_imtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
        else:
            train_ds = None
            train_sampler = None
            train_loader = None
            val_ds = ImageDataset(images, segs, transform=val_imtrans, seg_transform=val_imtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        if self.image_test_dir is not None:
            ###### testing
            filenames_test = sorted([os.path.basename(f) for f in glob(os.path.join(self.image_test_dir, f"*{self.file_format}"))])

            images_test = [os.path.join(self.image_test_dir, filename) for filename in filenames_test]
            test_ds = ImageDataset(images_test, transform=val_imtrans)
            test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())


        print(f"found training/val/test {len(images_train)}/{len(images_val)}/{len(images_test)} images and training/val {len(segs_train)}/{len(segs_val)} masks. mask can be zero, so len(masks) <= len(images).")
        return (train_ds, train_sampler, train_loader), (val_ds, val_loader), (test_ds, test_loader), (negative_train_ds, negative_train_loader), (None, None)


    def prepare_model(self, pretrain=None):
       
        # from modified_monai.nets.unet import UNet
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            kernel_size = 5,
        )
      
        model = DistributedDataParallel(model.to(self.device), device_ids=[self.device])
        if pretrain is not None:
            print(f"loading pretrain from: ./{pretrain}")
            data = torch.load(f"./{pretrain}")
            model.load_state_dict(data)
        return model

    def prepare_losses(self):
        loss = monai.losses.DiceLoss(sigmoid=True)

        return loss


    def inference_for_validation(self, inputs, labels, **aux_inputs):
        val_outputs = self.model(inputs) #sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        # import pdb;pdb.set_trace()
        val_outputs_digit = [i for i in decollate_batch(val_outputs)]
        val_outputs = [self.post_trans(i) for i in val_outputs_digit]
        # compute metric for current iteration
        self.dice_metric(y_pred=val_outputs, y=labels)

        return val_outputs
    
    
    def inference_for_testing(self, inputs, **aux_inputs):
        val_outputs = self.model(inputs, test=True) #sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        # import pdb;pdb.set_trace()
        val_outputs_digit = [i for i in decollate_batch(val_outputs)]
        val_outputs = [self.post_trans(i) for i in val_outputs_digit]
        
        return val_outputs
    

    def inference_for_training(self, inputs, labels, **aux_inputs):

        outputs = self.model(inputs)
        

        loss_seg = self.seg_loss(outputs, labels)
        loss = 0
        loss += loss_seg

        info = f"train_loss: {loss_seg.item():.4f}"

        return loss, outputs, info
    

    def preprocessing(self, inputs, **aux_inputs):
        (inputs,) = self.window_adjust(inputs=[inputs])
        # (inputs,) = self.nii_to_2d(inputs=[inputs])

        return inputs


    def preprocessing_with_labels(self, inputs, labels, **aux_inputs):
        # other_inputs = {}
        # if 'with_neg' in aux_inputs.keys():
        #     negative_samples = self.next_negative_sample()
        #     aggregate_batch_size = self.batch_size+negative_samples.shape[0]
        #     if self.class_labels is None or self.class_labels.shape[0]!=aggregate_batch_size:
        #         self.class_labels = torch.tensor([1.]*self.batch_size+[0.]*negative_samples.shape[0])[..., None].to(self.device)

        #     other_inputs["negative_samples"] = negative_samples

        (inputs,) = self.window_adjust(inputs=[inputs])
        # inputs, labels = self.nii_to_2d(inputs=[inputs, labels])

        # for k, v in other_inputs.items():
        #     (v,) = self.window_adjust(inputs=[v])
        #     (v,) = self.nii_to_2d(inputs=[v])
        #     other_inputs[k] = v

        return inputs, labels, None

    
    def plot_results(self, idx, epoch, inputs, outputs, suffix="", **kwargs):
        name = os.path.join(os.getcwd(), "runs", f"{self.args.mode}{suffix}_output", f"{epoch}_{idx}_{self.args.mode}_{self.args.model}_{self.args.signature}_.png")
        if idx % 100 == 0:
            random_slices_idx = [np.random.randint(0,outputs[0].shape[-1]) for _ in range(4)]
            # import pdb;pdb.set_trace()
            images = stitch_images(
                postprocess(inputs[0, :, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                *[postprocess(v[0, :, :, :, random_slices_idx].permute(3, 0, 1, 2)) for k,v in kwargs.items()],
                postprocess(outputs[0][:, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                img_per_row=1
            )
            print('\nsaving sample: ' + name)
            images.save(name)
