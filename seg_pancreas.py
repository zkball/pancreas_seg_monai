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
class seg_pancreas_base():

    def __init__(
        self, args
    ) -> None:
        """
        relatively-fixed variables
        """
        self.args = args
        self.device = torch.device(f"cuda:{idist.get_local_rank()}")
        torch.cuda.set_device(self.device)
        self.val_interval = 1
        self.file_format=".nii.gz"

        if not self.args.disable_tb:
            self.writer = SummaryWriter()
        self.bce_loss = nn.BCELoss().to(self.device)
        self.l2_loss = nn.MSELoss().to(self.device)
        self.is_master_process = idist.get_local_rank() == 0
        """
        task-related variables
        """
        self.batch_size = 4
        self.num_epoch  = 100
        self.num_epoch = self.num_epoch if self.args.eval is False else 1

        self.image_dir = None
        self.mask_dir = None
        self.image_test_dir = None
        self.negative_dir = None
        self.learning_rate = None


    def prepare_data(self):
        raise NotImplementedError("please implement prepare_data")

    def prepare_model(self, pretrain=None):
        raise NotImplementedError("please implement prepare_model")

    def prepare_losses(self):
        raise NotImplementedError("please implement prepare_losses")
    
    def prepare_everything(self):
        positive_training_stuff, positive_val_stuff, test_stuff, negative_training_stuff, negative_val_stuff = self.prepare_data()
        self.train_ds, self.train_sampler, self.train_loader = positive_training_stuff
        self.val_ds, self.val_loader = positive_val_stuff
        self.test_ds, self.test_loader = test_stuff
        self.negative_train_ds, self.negative_train_loader = negative_training_stuff
        self.negative_val_ds, self.negative_val_loader = negative_val_stuff

        if self.negative_train_loader is not None:
            print("init negative iterator")
            self.negative_iter = iter(self.negative_train_loader)
        else:
            self.negative_iter = None

        self.dice_metric = DCE(include_background=False, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        self.model = self.prepare_model(pretrain=self.args.pretrain)
        self.best_metric = -1
        if self.args.pretrain is not None:
            self.best_metric = float(self.args.pretrain.split("_")[4])
            print(f"best metric set as {self.best_metric}")
        self.best_metric_epoch = -1

        self.seg_loss = self.prepare_losses()     
        self.class_loss = self.bce_loss 
        self.optimizer = torch.optim.Adam(self.model.parameters(), )


    def window_adjust(self, inputs: list, w_width=250, w_min=-75, w_max=175):
        
        for idx, item in enumerate(inputs):
            inputs[idx] = (inputs[idx]-w_min)/w_width
        
        return tuple(inputs)

    
    def nii_to_2d(self, inputs: list):
        for idx, item in enumerate(inputs):
            inputs[idx] = inputs[idx][..., 0]
        return tuple(inputs)
    
    def preprocessing(self, inputs, **aux_inputs):
        raise NotImplementedError("please implement preprocessing")
    
    def preprocessing_with_labels(self, inputs, labels, **aux_inputs):
        raise NotImplementedError("please implement preprocessing")
    
    def inference_for_training(self, inputs, labels, **aux_inputs):
        raise NotImplementedError("please implement inference_for_training")
    
    def inference_for_validation(self, inputs, labels, **aux_inputs):
        raise NotImplementedError("please implement inference_for_validation")
    
    def inference_for_testing(self, inputs, **aux_inputs):
        raise NotImplementedError("please implement inference_for_testing")
    
    def plot_results(self, idx, epoch, inputs, outputs, **kwargs):
        raise NotImplementedError("please implement plot_results")

    def next_negative_sample(self):
        try:
            negative_samples = next(self.negative_iter).to(self.device)
            if negative_samples.shape[0] != self.batch_size:
                print("throw stop iteration")
                raise StopIteration("end of negative samples")
        except StopIteration as e:
            print("refresh negative loader")
            self.negative_iter = iter(self.negative_train_loader)
            negative_samples = next(self.negative_iter).to(self.device)
            print(f"refresh:{negative_samples.shape[0]}")
        
        return negative_samples
    
    def test_loop(self, epoch="infer"):
        for idx_test, test_data in enumerate(self.test_loader):
            # import pdb;pdb.set_trace()
            test_images = test_data.to(self.device)

            test_images = self.preprocessing(test_images)

            test_outputs = self.inference_for_testing(test_images)

            if idist.get_local_rank() == 0:
                self.plot_results(idx_test, epoch, inputs=test_images, outputs=test_outputs)
            idist.barrier()

    def eval_loop(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            ###### validation
            val_images = None
            val_labels = None
            val_outputs = None
            for idx_val, val_data in enumerate(self.val_loader):
                val_images, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                val_images, val_labels, _ = self.preprocessing_with_labels(val_images, val_labels)

                val_outputs = self.inference_for_validation(val_images, val_labels)

                if idist.get_local_rank() == 0:
                    self.plot_results(idx_val, epoch, inputs=val_images, outputs=val_outputs, labels=val_labels)
                    
                idist.barrier()

            # aggregate the final mean dice result
            metric = self.dice_metric.aggregate().item()
            # reset the status for next validation round
            self.dice_metric.reset()
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_metric_epoch = epoch + 1
                torch.save(self.model.state_dict(), f"best_{mode}_{self.args.model}_{self.args.signature}_{self.best_metric}_{self.best_metric_epoch}.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, self.best_metric, self.best_metric_epoch
                )
            )
            if not self.args.disable_tb:
                self.writer.add_scalar("val_mean_dice", metric, epoch + 1)

            ######### test - if have 
            if self.best_metric>0.5 and metric == self.best_metric:
                self.test_loop(epoch=epoch)
        
        return metric

    def train_loop(self):
        
        epoch_loss_values = list()
        metric_values = list()
    

        for epoch in range(self.num_epoch):
            
            ###### training code
            self.train_sampler.set_epoch(epoch)
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.num_epoch}")
            self.model.train()
            with torch.enable_grad():
                epoch_loss = 0
                step = 0
                for batch_data in tqdm(self.train_loader, disable=not self.is_master_process):
                    step += 1
                    inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

                    inputs, labels, aux_inputs = self.preprocessing_with_labels(inputs, labels, with_neg=True)

                    self.optimizer.zero_grad()
                    
                    loss, outputs, info = self.inference_for_training(inputs, labels, **aux_inputs)
                    
                    loss.backward()
                    ###  gradient_clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_len = len(self.train_ds) // self.train_loader.batch_size
                    tqdm.write(f"{step}/{epoch_len}, {info}")
                    if not self.args.disable_tb:
                        self.writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            ###### inference code
            if self.args.eval or (not self.args.eval and (epoch + 1) % self.val_interval == 0):
                metric = self.eval_loop()
                metric_values.append(metric)


                
        print(f"train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
        if not self.args.disable_tb:
            self.writer.close()

        dist.destroy_process_group()

        return epoch_loss_values, metric_values


if __name__ == "__main__":
    import argparse
    # with tempfile.TemporaryDirectory() as tempdir:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        required=False,
        help="Model definition",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        required=False,
        help="Model definition",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        required=False,
        help="Model definition",
    )
    parser.add_argument(
        "--signature",
        type=str,
        default="default",
        required=False,
        help="signature",
    )
    parser.add_argument(
        "--disable_tb",
        action="store_true",
        required=False,
        help="Force disable tensorboard for debugging",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        required=False,
        help="Eval Mode",
    )
    args = parser.parse_args()
    print(f"mode is {args.mode}")

    mode=args.mode
    disable_tb=args.disable_tb

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    if args.mode=="2d":
        from seg_pancreas_2d import seg_pancreas_2d
        pancreas_module = seg_pancreas_2d(args=args)
        print("seg_pancreas_2d created")
    elif args.mode=="3d":
        from seg_pancreas_3d import seg_pancreas_3d
        pancreas_module = seg_pancreas_3d(args=args)
        print("seg_pancreas_3d created")
    else:
        pancreas_module = seg_pancreas_base(args=args)

    if not args.eval:
        epoch_loss_values, metric_values = pancreas_module.train_loop()
        print(f"epoch_loss_values: {epoch_loss_values}")
        print(f"metric_values: {metric_values}")
    else:
        pancreas_module.test_loop()
