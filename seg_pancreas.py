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

import monai
from monai.data import (
    ImageDataset, ArrayDataset, create_test_image_3d, decollate_batch, DataLoader
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
    RandSpatialCrop,
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

def prepare_data(mode="3d", image_dir=None, mask_dir=None, file_format=None):
    if mode == "3d":
        filenames = sorted([os.path.basename(f) for f in glob(os.path.join(mask_dir, f"*{file_format}"))])
        images = [os.path.join(image_dir, filename) for filename in filenames]
        segs = [os.path.join(mask_dir, filename) for filename in filenames]

        train_imtrans = Compose(
            [
                ScaleIntensity(),
                EnsureChannelFirst(),
                RandSpatialCrop((512, 512, 128), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            ]
        )
        train_segtrans = Compose(
            [
                EnsureChannelFirst(),
                RandSpatialCrop((512, 512, 128), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            ]
        )
        val_imtrans = Compose([ScaleIntensity(), EnsureChannelFirst()])
        val_segtrans = Compose([EnsureChannelFirst()])

        train_ds = ImageDataset(images[:-200], segs[:-200], transform=train_imtrans, seg_transform=train_segtrans)
        train_sampler = DistributedSampler(train_ds)
        # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
        train_loader = DataLoader(
            train_ds,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
        )
        # create a validation data loader
        val_ds = ImageDataset(images[-200:], segs[-200:], transform=val_imtrans, seg_transform=val_segtrans)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    else:
        filenames = sorted([os.path.basename(f) for f in glob(os.path.join(mask_dir, f"*{file_format}"))])
        # filenames = [filename.split(".")[0]+filename.split(".")[-1] for filename in filenames]
        images = [os.path.join(image_dir, filename) for filename in filenames]
        segs = [os.path.join(mask_dir, filename) for filename in filenames]
        train_imtrans = Compose(
            [
                LoadImage(reader=PILReader, image_only=True, ensure_channel_first=True),
                ScaleIntensity(),
                RandSpatialCrop((512, 512), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            ]
        )
        train_segtrans = Compose(
            [
                LoadImage(reader=PILReader, image_only=True, ensure_channel_first=True),
                ScaleIntensity(),
                RandSpatialCrop((512, 512), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            ]
        )
        val_imtrans = Compose([
            LoadImage(reader=PILReader,image_only=True, ensure_channel_first=True), 
            ScaleIntensity()])
        val_segtrans = Compose([
            LoadImage(reader=PILReader,image_only=True, ensure_channel_first=True), 
            ScaleIntensity()])

        train_ds = ArrayDataset(images[:-20], train_imtrans, segs[:-20], train_segtrans)
        train_sampler = DistributedSampler(train_ds)
        train_loader = DataLoader(
            train_ds, 
            batch_size=8, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True,
            sampler=train_sampler)
        val_ds = ArrayDataset(images[-20:], val_imtrans, segs[-20:], val_segtrans)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    print(f"found {len(images)} images and {len(segs)} masks. mask can be zero, so len(masks) <= len(images).")
    return train_ds, train_sampler, train_loader, val_ds, val_loader

def prepare_model(mode):
    if mode == "3d":
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    return model

def prepare_losses(mode):
    if mode == "3d":
        loss = monai.losses.TverskyLoss(sigmoid=True, alpha=0.8, beta=0.2) #monai.losses.DiceLoss(sigmoid=True)

    else:
        loss = monai.losses.DiceLoss(sigmoid=True)

    return loss


def main(mode="3d"):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    if mode=="3d":
        image_dir='/raid/datasets/origin/images_A'
        mask_dir='/raid/datasets/origin/Mask_A'
        file_format=".nii.gz"
        roi_size = (512, 512, 128)
    else:
        image_dir='/raid/datasets/origin_generated_2d_slices/images_A'
        mask_dir='/raid/datasets/origin_generated_2d_slices/Mask_A'
        file_format=".png"
        roi_size = (512, 512)

    ## previously we iterate two folders independently
    # images = sorted(glob(os.path.join(image_dir, f"*{file_format}")))
    # segs = sorted(glob(os.path.join(mask_dir, f"*{file_format}")))
    ## now we only iterate mask folder
    
    master_process = idist.get_local_rank() == 0
    # define transforms for image and segmentation
    train_ds, train_sampler, train_loader, val_ds, val_loader = prepare_data(mode=mode, 
                                                                             image_dir=image_dir, 
                                                                             mask_dir=mask_dir,
                                                                             file_format=file_format)
    # import pdb;pdb.set_trace()

    # define image dataset, data loader
    # check_ds = ImageDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
    # check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, seg = monai.utils.misc.first(check_loader)
    # print(im.shape, seg.shape)

    dice_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{idist.get_local_rank()}")
    torch.cuda.set_device(device)
    model = prepare_model(mode=mode).to(device)
    loss_function = prepare_losses(mode=mode)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    model = DistributedDataParallel(model, device_ids=[device])

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    num_epoch  =20
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epoch}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # import pdb;pdb.set_trace();
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                if idist.get_local_rank() == 0:
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                    plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                    plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
                idist.barrier()
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

    dist.destroy_process_group()


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
    args = parser.parse_args()
    print(f"mode is {args.mode}")
    main(args.mode)
