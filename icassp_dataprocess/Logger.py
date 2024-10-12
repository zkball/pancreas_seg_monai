import os
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


class LossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics['train_loss'].item())
        print("train loss", trainer.callback_metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics['val_loss'].item())
        print("val loss", trainer.callback_metrics['val_loss'].item())


class ImageLogger_2D(Callback):
    def __init__(self, image_save_dir, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.save_dir = image_save_dir
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):

        root = os.path.join(self.save_dir, "image_log", split)

        for k in images:
            # print(images[k].shape)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            # print(grid.shape)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # print(grid.shape)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            # print(grid.shape)
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)


    def log_img(self, pl_module, batch, batch_idx, split, val_sample_name = None):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.train()

            else:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images) #N=4
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)


            self.log_local(split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)



    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            # self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_img(pl_module, batch, batch_idx, split="train")


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            # self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            # self.log_img(pl_module, batch, batch_idx, split="train")
            print("on_test_batch_end")
            self.log_img(pl_module, batch, batch_idx, split="test")

class VolumeLogger(Callback):
    def __init__(self, image_save_dir, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.save_dir = image_save_dir
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def CT_volume_to_image(self, images,global_step, current_epoch, batch_idx, split):
        root = os.path.join(self.save_dir, "image_log", split)
        os.makedirs(root, exist_ok=True)
        input_image = images['input_image_batch'].detach().cpu()
        input_label = images['input_label_batch'].detach().cpu()
        predict_label = images['predict_label_batch'].detach().cpu()

        predict_label = torch.clamp(predict_label, -1., 1.)
        predict_label = (predict_label * 255).astype(np.uint8)


        sample_slice_list = np.arange(1, 96, 2).tolist()


        plt.figure("check", (10, 50))
        i = 1
        for row, slice in enumerate(sample_slice_list):
            slice_index = slice - 1
            plt.subplot(50, 3, i)

            if i == 1:
                plt.title("input image")
            plt.imshow(input_image[0, 0, :, :, slice_index], cmap="gray")
            plt.axis('off')
            plt.subplot(50, 3, i + 1)
            if i == 1:
                plt.title("input label")
            plt.imshow(input_label[0, 0, :, :, slice_index], cmap="gray")
            plt.axis('off')
            plt.subplot(50, 3, i + 2)
            if i == 1:
                plt.title("predict label")
            plt.imshow(predict_label[0, 0, :, :, slice_index], cmap="gray")
            plt.axis('off')
            plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=1, wspace=0.0001, hspace=0.0001)  # 调整子图间距

            i = i + 3

        filename = split + "-{:06}_e-{:06}_b-{:06}.png".format( global_step, current_epoch, batch_idx)
        plt.savefig(os.path.join(root, filename))

    def log_CT_volume(self,pl_module, batch, batch_idx, split):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training

            if is_train:
                pl_module.train()

            else:
                pl_module.eval()


            with torch.no_grad():
                images = pl_module.log_ct_images(batch, **self.log_images_kwargs)

            self.CT_volume_to_image(images, pl_module.global_step, pl_module.current_epoch, batch_idx, split)



    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            # self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_CT_volume(pl_module, batch, batch_idx, split="train")


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            # self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_CT_volume(pl_module, batch, batch_idx, split="val")



