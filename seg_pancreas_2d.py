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
class seg_pancreas_2d(seg_pancreas_base):

    def __init__(
        self, args
    ) -> None:
        super().__init__(args=args)

        """
        task-related variables
        """
        self.batch_size = 4
        self.class_labels = None

        self.num_epoch  = 100
        self.num_epoch = self.num_epoch if self.args.eval is False else 1

        self.image_dir='/raid/datasets/origin_generated_2d_slices/images_A'
        self.mask_dir='/raid/datasets/origin_generated_2d_slices/Mask_A'
        self.negative_dir='/raid/datasets/origin_negative_2d_slices/images_A'
        self.file_format=".nii.gz"
        self.roi_size = (512, 512)

        print("dir paths set")

        self.image_test_dir='/raid/datasets/207file_generated_2d_slices/images_A'

        """
        prepare dataset and model
        """
        self.prepare_everything()
    

    def preprocessing(self, inputs, **aux_inputs):
        (inputs,) = self.window_adjust(inputs=[inputs])
        (inputs,) = self.nii_to_2d(inputs=[inputs])

        return inputs


    def preprocessing_with_labels(self, inputs, labels, **aux_inputs):
        other_inputs = {}
        if 'with_neg' in aux_inputs.keys():
            negative_samples = self.next_negative_sample()
            aggregate_batch_size = self.batch_size+negative_samples.shape[0]
            if self.class_labels is None or self.class_labels.shape[0]!=aggregate_batch_size:
                self.class_labels = torch.tensor([1.]*self.batch_size+[0.]*negative_samples.shape[0])[..., None].to(self.device)

            other_inputs["negative_samples"] = negative_samples

        (inputs,) = self.window_adjust(inputs=[inputs])
        inputs, labels = self.nii_to_2d(inputs=[inputs, labels])

        for k, v in other_inputs.items():
            (v,) = self.window_adjust(inputs=[v])
            (v,) = self.nii_to_2d(inputs=[v])
            other_inputs[k] = v

        return inputs, labels, other_inputs


    def prepare_data(self):
        test_loader, test_ds = None, None
        negative_train_ds, negative_train_sampler, negative_train_loader = None, None, None
        
        
        import math
        from monai.transforms import (
            RandSpatialCrop
        )
        reader = None if ".nii.gz" in self.file_format else PILReader

        filenames = sorted([os.path.basename(f) for f in glob(os.path.join(self.mask_dir, f"*{self.file_format}"))])


        print("train/val split according to patient id...")
        batch_names = set()
        for file in filenames:
            patient_id = file.split(".nii.gz")[0]
            batch_names.add(patient_id)

        sorted_batches = sorted(list(batch_names))
        split_index = math.ceil(len(sorted_batches) * 0.85)
        training_batches = set(sorted_batches[:split_index])
        test_batches = set(sorted_batches[split_index:])
        
        images_train, images_val, segs_train, segs_val = [], [], [], []
        for file in filenames:
            patient_id = file.split(".nii.gz")[0]
            if patient_id in training_batches:
                images_train.append(os.path.join(self.image_dir, file))
                segs_train.append(os.path.join(self.mask_dir, file))
            else:
                images_val.append(os.path.join(self.image_dir, file))
                segs_val.append(os.path.join(self.mask_dir, file))


        # images = [os.path.join(image_dir, filename) for filename in filenames]
        # segs = [os.path.join(mask_dir, filename) for filename in filenames]
        train_imtrans = Compose(
            [
                LoadImage(reader=reader, image_only=True, ensure_channel_first=True),
                # ScaleIntensity(),
                RandSpatialCrop((512, 512, 1), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            ]
        )
        train_segtrans = Compose(
            [
                LoadImage(reader=reader, image_only=True, ensure_channel_first=True),
                # ScaleIntensity(),
                RandSpatialCrop((512, 512, 1), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            ]
        )
        val_imtrans = Compose([
            LoadImage(reader=reader,image_only=True, ensure_channel_first=True), 
            # ScaleIntensity()
            ])
        val_segtrans = Compose([
            LoadImage(reader=reader,image_only=True, ensure_channel_first=True), 
            # ScaleIntensity()
            ])

        if not self.args.eval:
            train_ds = ArrayDataset(images_train, train_imtrans, segs_train, train_segtrans)
            train_sampler = DistributedSampler(train_ds)
            train_loader = DataLoader(
                train_ds, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.batch_size, 
                pin_memory=True,
                sampler=train_sampler)
            val_ds = ArrayDataset(images_val, val_imtrans, segs_val, val_segtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
        else:
            train_ds = None
            train_sampler = None
            train_loader = None
            val_ds = ArrayDataset(images_train+images_val, val_imtrans, segs_train+segs_val, val_segtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        if self.negative_dir is not None:
            nagative_filenames = sorted([os.path.basename(f) for f in glob(os.path.join(self.negative_dir, f"*{self.file_format}"))])
            # nagative_filenames = list(set(nagative_filenames) - set(filenames))
            # import pdb;pdb.set_trace()
            negative_train, negative_val = [], []

            for file in nagative_filenames:
                patient_id = file.split(".nii.gz")[0]
                if patient_id in test_batches:
                    negative_val.append(os.path.join(self.negative_dir, file))
                else:
                    negative_train.append(os.path.join(self.negative_dir, file))

            negative_train_ds = ArrayDataset(negative_train, train_imtrans)
            negative_train_loader = DataLoader(
                negative_train_ds, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.batch_size, 
                pin_memory=True,
                )
            negative_val_ds = ArrayDataset(negative_val, val_imtrans)
            negative_val_loader = DataLoader(negative_val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        if self.image_test_dir is not None:
            ###### testing
            filenames_test = sorted([os.path.basename(f) for f in glob(os.path.join(self.image_test_dir, f"*{self.file_format}"))])

            images_test = [os.path.join(self.image_test_dir, filename) for filename in filenames_test]
            test_ds = ArrayDataset(images_test, val_imtrans)
            test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        print(f"found training/val/test {len(images_train)}/{len(images_val)}/{len(images_test)} images and training/val {len(segs_train)}/{len(segs_val)} masks. mask can be zero, so len(masks) <= len(images).")
        return (train_ds, train_sampler, train_loader), (val_ds, val_loader), (test_ds, test_loader), (negative_train_ds, negative_train_loader), (negative_val_ds, negative_val_loader)

    def prepare_model(self, pretrain=None):
        
        if self.args.model=="multiscale_unet":
            from modified_monai.nets.multiscale_unet import Unet
            model = Unet(channels=1, use_bayar=False, use_fft=False, use_SRM=False, dim=16,
                        resnet_block_groups=4,
                        use_hierarchical_segment=[],
                        use_classification=[], 
                        use_hierarchical_class=[],
                        use_normal_output=[1]
                        )


        else:
            from modified_monai.nets.Unet import UNet
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=16,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units=4,
                kernel_size=5,
                classification=[1]
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
        val_outputs, _ = self.model(inputs) #sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        # import pdb;pdb.set_trace()
        val_outputs_digit = [i for i in decollate_batch(val_outputs)]
        val_outputs = [self.post_trans(i) for i in val_outputs_digit]
        # compute metric for current iteration
        self.dice_metric(y_pred=val_outputs, y=labels)

        return val_outputs
    
    def inference_for_testing(self, inputs, **aux_inputs):
        val_outputs, _ = self.model(inputs, test=True) #sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        # import pdb;pdb.set_trace()
        val_outputs_digit = [i for i in decollate_batch(val_outputs)]
        val_outputs = [self.post_trans(i) for i in val_outputs_digit]
        
        return val_outputs
    

    def inference_for_training(self, inputs, labels, **aux_inputs):
        negative_samples = aux_inputs["negative_samples"]
        outputs, classification = self.model(torch.cat([inputs,negative_samples],dim=0), test=False)
        

        loss_seg = self.seg_loss(outputs[:self.batch_size], labels)
        loss_class = self.class_loss(classification[0], self.class_labels)
        loss = 0
        loss += loss_seg
        loss += 0.1*loss_class

        info = f"train_loss: {loss_seg.item():.4f}, class_loss: {loss_class.item():.4f}"

        return loss, outputs, info
    
    def plot_results(self, idx, epoch, inputs, outputs, **kwargs):
        name = os.path.join(os.getcwd(), "runs", f"{self.args.mode}_output", f"{epoch}_{idx}_{self.args.mode}_{self.args.model}_{self.args.signature}_.png")
        if idx % 100 == 0:
            # import pdb;pdb.set_trace()
            images = stitch_images(
                postprocess(inputs),
                *[postprocess(v) for k,v in kwargs.items()],
                postprocess(outputs[0][None]),
                img_per_row=1
            )
            print('\nsaving sample: ' + name)
            images.save(name)

