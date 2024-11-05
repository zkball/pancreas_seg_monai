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

def define_batch_size(mode="3d"):
    if mode == "3d":
        batch_size = 4
    else:
        batch_size = 8

    return batch_size

def prepare_data(mode="3d", image_dir=None, mask_dir=None, file_format=None, only_eval=False, image_test_dir=None, negative_dir=None, batch_size=None):
    test_loader, test_ds = None, None
    negative_train_ds, negative_train_sampler, negative_train_loader = None, None, None
    
    if mode == "3d":
        from modified_monai.transforms.rand_slice_crop import (
            RandSpatialCrop
        )

        Z_location, remove_set = get_Z_location(image_dir)

        # import pdb;pdb.set_trace()
        filenames = sorted([os.path.basename(f) for f in glob(os.path.join(mask_dir, f"*{file_format}"))])
        filenames = list(filter(lambda x: x in set(Z_location.keys()) and x not in remove_set, filenames))

        images = [os.path.join(image_dir, filename) for filename in filenames]
        segs = [os.path.join(mask_dir, filename) for filename in filenames]

        train_imtrans = Compose(
            [
                # ScaleIntensity(),
                EnsureChannelFirst(),
                # RandSpatialCrop((512, 512, 64), random_size=False),
                # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=2),
            ]
        )
        # train_segtrans = Compose(
        #     [
        #         EnsureChannelFirst(),
        #         # RandSpatialCrop((512, 512, 64), random_size=False),
        #         # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        #         RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=4),
        #     ]
        # )
        val_imtrans = Compose(
            [
                # ScaleIntensity(), 
                EnsureChannelFirst(),
                RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=2),
            ]
        )
        # val_segtrans = Compose(
        #     [
        #         EnsureChannelFirst(),
        #         RandSpatialCrop(roi_size=(512, 512, 64), slice_dict=Z_location, stride_slice=3),
        #     ]
        # )

        if not only_eval:
            train_ds = ImageDataset(images[:-40], segs[:-40], transform=train_imtrans, seg_transform=train_imtrans)
            train_sampler = DistributedSampler(train_ds)
            # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=batch_size,
                pin_memory=True,
                sampler=train_sampler,
            )
            # create a validation data loader
            val_ds = ImageDataset(images[-40:], segs[-40:], transform=val_imtrans, seg_transform=val_imtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
        else:
            train_ds = None
            train_sampler = None
            train_loader = None
            val_ds = ImageDataset(images, segs, transform=val_imtrans, seg_transform=val_imtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        if image_test_dir is not None:
            ###### testing
            filenames_test = sorted([os.path.basename(f) for f in glob(os.path.join(image_test_dir, f"*{file_format}"))])

            images_test = [os.path.join(image_test_dir, filename) for filename in filenames_test]
            test_ds = ImageDataset(images_test, transform=val_imtrans)
            test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    else:
        import math
        from monai.transforms import (
            RandSpatialCrop
        )
        reader = None if ".nii.gz" in file_format else PILReader

        filenames = sorted([os.path.basename(f) for f in glob(os.path.join(mask_dir, f"*{file_format}"))])


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
                images_train.append(os.path.join(image_dir, file))
                segs_train.append(os.path.join(mask_dir, file))
            else:
                images_val.append(os.path.join(image_dir, file))
                segs_val.append(os.path.join(mask_dir, file))


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

        if not only_eval:
            train_ds = ArrayDataset(images_train, train_imtrans, segs_train, train_segtrans)
            train_sampler = DistributedSampler(train_ds)
            train_loader = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=batch_size, 
                pin_memory=True,
                sampler=train_sampler)
            val_ds = ArrayDataset(images_val, val_imtrans, segs_val, val_segtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
        else:
            train_ds = None
            train_sampler = None
            train_loader = None
            val_ds = ArrayDataset(images, val_imtrans, segs, val_segtrans)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        if negative_dir is not None:
            nagative_filenames = sorted([os.path.basename(f) for f in glob(os.path.join(negative_dir, f"*{file_format}"))])
            # nagative_filenames = list(set(nagative_filenames) - set(filenames))
            # import pdb;pdb.set_trace()
            negative_train, negative_val = [], []

            for file in nagative_filenames:
                patient_id = file.split(".nii.gz")[0]
                if patient_id in test_batches:
                    negative_val.append(os.path.join(negative_dir, file))
                else:
                    negative_train.append(os.path.join(negative_dir, file))

            negative_train_ds = ArrayDataset(negative_train, train_imtrans)
            negative_train_loader = DataLoader(
                negative_train_ds, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=batch_size, 
                pin_memory=True,
                )
            negative_val_ds = ArrayDataset(negative_val, val_imtrans)
            negative_val_loader = DataLoader(negative_val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        if image_test_dir is not None:
            ###### testing
            filenames_test = sorted([os.path.basename(f) for f in glob(os.path.join(image_test_dir, f"*{file_format}"))])

            images_test = [os.path.join(image_test_dir, filename) for filename in filenames_test]
            test_ds = ArrayDataset(images_test, val_imtrans)
            test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    print(f"found training/val/test {len(images_train)}/{len(images_val)}/{len(images_test)} images and training/val {len(segs_train)}/{len(segs_val)} masks. mask can be zero, so len(masks) <= len(images).")
    return (train_ds, train_sampler, train_loader), (val_ds, val_loader), (test_ds, test_loader), (negative_train_ds, negative_train_loader), (negative_val_ds, negative_val_loader)

def prepare_model(mode, pretrain=None, device=None):
    if mode == "3d":
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
    else:
        if args.model=="multiscale_unet":
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

    model = model = DistributedDataParallel(model.to(device), device_ids=[device])
    if pretrain is not None:
        print(f"loading pretrain from: ./{pretrain}")
        data = torch.load(f"./{pretrain}")
        model.load_state_dict(data)
    return model

def prepare_losses(mode):
    if mode == "3d":
        loss = monai.losses.DiceLoss(sigmoid=True)
        # loss = monai.losses.TverskyLoss(sigmoid=True, alpha=0.8, beta=0.2) #monai.losses.DiceLoss(sigmoid=True)

    else:
        loss = monai.losses.DiceLoss(sigmoid=True)

    return loss

def window_adjust(data, w_width=250, w_min=-75, w_max=175):

    data_adjusted = (data-w_min)/w_width
    
    return data_adjusted

def main(args):
    mode=args.mode
    disable_tb=args.disable_tb

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
        negative_dir='/raid/datasets/origin_negative_2d_slices/images_A'
        file_format=".nii.gz"
        roi_size = (512, 512)

        image_test_dir='/raid/datasets/207file_generated_2d_slices/images_A'

    ## previously we iterate two folders independently
    # images = sorted(glob(os.path.join(image_dir, f"*{file_format}")))
    # segs = sorted(glob(os.path.join(mask_dir, f"*{file_format}")))
    ## now we only iterate mask folder
    
    master_process = idist.get_local_rank() == 0

    batch_size = define_batch_size(mode=mode)
    positive_training_stuff, positive_val_stuff, test_stuff, negative_training_stuff, negative_val_stuff = prepare_data(mode=mode, 
                                                                             image_dir=image_dir, 
                                                                             mask_dir=mask_dir,
                                                                             file_format=file_format,
                                                                             image_test_dir=image_test_dir,
                                                                             negative_dir=negative_dir,
                                                                             batch_size=batch_size)
    
    train_ds, train_sampler, train_loader = positive_training_stuff
    val_ds, val_loader = positive_val_stuff
    test_ds, test_loader = test_stuff
    negative_train_ds, negative_train_loader = negative_training_stuff
    negative_val_ds, negative_val_loader = negative_val_stuff

    # import pdb;pdb.set_trace()

    # define image dataset, data loader
    # check_ds = ImageDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
    # check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, seg = monai.utils.misc.first(check_loader)
    # print(im.shape, seg.shape)

    dice_metric = DCE(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{idist.get_local_rank()}")
    torch.cuda.set_device(device)
    model = prepare_model(mode=mode, pretrain=args.pretrain, device=device)
    loss_function = prepare_losses(mode=mode)
    bce_loss = nn.BCELoss().to(device)
    l2_loss = nn.MSELoss().to(device)
    learning_rate = 1e-3 if mode=="2d" else 5e-3 
    optimizer = torch.optim.Adam(model.parameters(), )

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    if args.pretrain is not None:
        best_metric = float(args.pretrain.split("_")[4])
        print(f"best metric set as {best_metric}")
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    if not disable_tb:
        writer = SummaryWriter()
    num_epoch  = 100 if mode=="2d" else 1000
    num_epoch = num_epoch if args.eval is False else 1
    print("refresh negative loader")
    negative_iter = iter(negative_train_loader)

    for epoch in range(num_epoch):
        
        ###### training code
        if not args.eval:
            train_sampler.set_epoch(epoch)
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epoch}")
            model.train()
            with torch.enable_grad():
                epoch_loss = 0
                step = 0
                for batch_data in tqdm(train_loader, disable=not master_process):
                    step += 1
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    batch_size = inputs.shape[0]
                    ######### load negative batch
                    try:
                        negative_samples = next(negative_iter).to(device)
                        if negative_samples.shape[0] != batch_size:
                            print("throw stop iteration")
                            raise StopIteration("end of negative samples")
                    except StopIteration as e:
                        print("refresh negative loader")
                        negative_iter = iter(negative_train_loader)
                        negative_samples = next(negative_iter).to(device)
                        print(f"refresh:{negative_samples.shape[0]}")

                    class_labels = torch.tensor([1.]*batch_size+[0.]*negative_samples.shape[0])[..., None].to(device)

                    inputs = window_adjust(inputs)
                    negative_samples = window_adjust(negative_samples)

                    optimizer.zero_grad()

                    ## handle cases 2d images might be read from 3d nib
                    if mode=="2d" and len(inputs.shape)>4:
                        inputs, labels, negative_samples = inputs[...,0], labels[..., 0], negative_samples[..., 0]

                    
                    """
                    Multiscale output
                    (Pdb) self.output_encoder[0].shape
                    torch.Size([8, 16, 256, 256])
                    (Pdb) self.output_encoder[1].shape
                    torch.Size([8, 32, 128, 128])
                    (Pdb) self.output_encoder[2].shape
                    torch.Size([8, 64, 64, 64])
                    (Pdb) self.output_encoder[3].shape
                    torch.Size([8, 128, 32, 32])
                    (Pdb) self.output_encoder[4].shape
                    torch.Size([8, 256, 16, 16])

                    (Pdb) self.output_decoder[0].shape
                    torch.Size([8, 512, 16, 16])
                    (Pdb) self.output_decoder[1].shape
                    torch.Size([8, 128, 32, 32])
                    (Pdb) self.output_decoder[2].shape
                    torch.Size([8, 64, 64, 64])
                    (Pdb) self.output_decoder[3].shape
                    torch.Size([8, 32, 128, 128])
                    (Pdb) self.output_decoder[4].shape ## use this level
                    torch.Size([8, 16, 256, 256])
                    (Pdb) self.output_decoder[5].shape
                    torch.Size([8, 1, 512, 512])
                    """
                    outputs, classification = model(torch.cat([inputs,negative_samples],dim=0), train=True)
                    # import pdb;pdb.set_trace()
                    loss_seg = loss_function(outputs[:batch_size], labels)
                    
                    loss_class = l2_loss(classification[0], class_labels)
                    loss = 0
                    loss += loss_seg
                    loss += 0.1*loss_class
                    # if loss_class>20:
                        # import pdb;pdb.set_trace()
                    loss.backward()
                    ###  gradient_clipping
                    nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_len = len(train_ds) // train_loader.batch_size
                    tqdm.write(f"{step}/{epoch_len}, train_loss: {loss_seg.item():.4f}, class_loss: {loss_class.item():.4f}")
                    if not disable_tb:
                        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        ###### inference code
        if args.eval or (not args.eval and (epoch + 1) % val_interval == 0):
            model.eval()
            with torch.no_grad():
                ###### validation
                val_images = None
                val_labels = None
                val_outputs = None
                for idx_val, val_data in enumerate(val_loader):
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

                    val_images = window_adjust(val_images)

                    sw_batch_size = 4

                    ## handle cases 2d images might be read from 3d nib
                    if mode=="2d" and len(val_images.shape)>4:
                        val_images, val_labels = val_images[...,0], val_labels[...,0]

                    val_outputs = model(val_images) #sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    # import pdb;pdb.set_trace()
                    val_outputs_digit = [i for i in decollate_batch(val_outputs)]
                    val_outputs = [post_trans(i) for i in val_outputs_digit]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                    if idist.get_local_rank() == 0:
                        # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                        # print(val_images.shape)
                        # print(val_labels.shape)
                        # print(val_outputs.shape)
                        # import pdb;pdb.set_trace()
                        # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                        # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                        # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
                        name = os.path.join(os.getcwd(), "runs", f"{mode}_output", f"{epoch}_{idx_val}_{mode}_{args.model}_{args.signature}_.png")
                        if mode=="2d" and idx_val % 100 == 0:
                            images = stitch_images(
                                postprocess(val_images),
                                postprocess(val_labels),
                                postprocess(val_outputs[0][None]),
                                img_per_row=1
                            )
                            print('\nsaving sample: ' + name)
                            images.save(name)
                        elif mode=="3d" and idx_val % 2 == 0:
                            random_slices_idx = [np.random.randint(0,val_outputs[0].shape[-1]) for _ in range(4)]
                            # import pdb;pdb.set_trace()
                            images = stitch_images(
                                postprocess(val_images[0, :, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                                postprocess(val_labels[0, :, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                                postprocess(val_outputs[0][:, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                                img_per_row=1
                            )
                            print('\nsaving sample: ' + name)
                            images.save(name)
                    idist.barrier()




                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"best_{mode}_{args.model}_{args.signature}_{best_metric}_{best_metric_epoch}.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                if not disable_tb:
                    writer.add_scalar("val_mean_dice", metric, epoch + 1)

                ######### test - if have 
                if best_metric>0.4 and metric == best_metric:
                    for idx_test, test_data in enumerate(test_loader):
                        # import pdb;pdb.set_trace()
                        test_images = test_data.to(device)

                        test_images = window_adjust(test_images)

                        sw_batch_size = 4

                        ## handle cases 2d images might be read from 3d nib
                        if mode=="2d" and len(test_images.shape)>4:
                            test_images = test_images[...,0]

                        test_outputs = model(test_images) #sliding_window_inference(test_images, roi_size, sw_batch_size, model)
                        # import pdb;pdb.set_trace()
                        test_outputs_digit = [i for i in decollate_batch(test_outputs)]
                        test_outputs = [post_trans(i) for i in test_outputs_digit]

                        if idist.get_local_rank() == 0:
                            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                            # print(test_images.shape)
                            # print(test_labels.shape)
                            # print(test_outputs.shape)
                            # import pdb;pdb.set_trace()
                            # plot_2d_or_3d_image(test_images, epoch + 1, writer, index=0, tag="image")
                            # plot_2d_or_3d_image(test_labels, epoch + 1, writer, index=0, tag="label")
                            # plot_2d_or_3d_image(test_outputs, epoch + 1, writer, index=0, tag="output")
                            name = os.path.join(os.getcwd(), "runs", f"{mode}_test_output", f"{epoch}_{idx_test}_{mode}_{args.model}_{args.signature}_.png")
                            if mode=="2d" and idx_test % 1000 == 0:
                                images = stitch_images(
                                    postprocess(test_images),
                                    postprocess(test_outputs[0][None]),
                                    img_per_row=1
                                )
                                print('\nsaving sample: ' + name)
                                images.save(name)
                            elif mode=="3d" and idx_test % 2 == 0:
                                random_slices_idx = [np.random.randint(0,test_outputs[0].shape[-1]) for _ in range(4)]
                                # import pdb;pdb.set_trace()
                                images = stitch_images(
                                    postprocess(test_images[0, :, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                                    postprocess(test_outputs[0][:, :, :, random_slices_idx].permute(3, 0, 1, 2)),
                                    img_per_row=1
                                )
                                print('\nsaving sample: ' + name)
                                images.save(name)
                        idist.barrier()


            
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    if not disable_tb:
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
    main(args)
