import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import time
import matplotlib.pyplot as plt

from data.Logger import ImageLogger_2D, LossLogger
from data.MedicalDataset import  MedicalDataset_2D_tumor_area_rank
from models.Pipeline import our_UNetPlusPlus, UNet, AttentionUnet_Pipeline
from torch.utils.data import DataLoader
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':

    seed_everything(42)

    batch_size = 4
    logger_freq = 5
    learning_rate = 1e-5
    timestamp = time.strftime("%Y-%m-%d")
    print(timestamp)


    epoch = 3

    model_name = f'/groupshare_1/wenqi_code/screen-watermark/unetpp_epoch_{epoch}'
    image_save_dir = model_name + f'_image_save_{timestamp}'
    ckpt_save_dir = model_name + f'_weights_{timestamp}'

    if os.path.exists(image_save_dir):
        shutil.rmtree(image_save_dir)

    if os.path.exists(ckpt_save_dir):
        shutil.rmtree(ckpt_save_dir)


    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # model = haar_UNetPlusPlus()
    # model = UNet()
    # model = AttentionUnet_Pipeline()
    model = our_UNetPlusPlus()



    # 设置保存路径和文件名模板
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpt_save_dir,
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=5,  # 保存最好的3个模型
        mode='min',  # 模式可以是 'min' 或 'max'，取决于监控的指标
    )

    #TODO: Train on the whole image data, half tumor and half no tumor

    train_has_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_train_v3/has_tumor'
    train_no_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_train_v3/no_tumor'
    val_has_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_val_v3/has_tumor'
    val_no_tumor_dir = '/groupshare_1/cancer_center/2d_origin_data_with_tumor_area_rank_val_v3/no_tumor'
    train_dataset = MedicalDataset_2D_tumor_area_rank(train_has_tumor_dir, train_no_tumor_dir, type='train', data_augmentation=False)
    val_dataset = MedicalDataset_2D_tumor_area_rank(val_has_tumor_dir, val_no_tumor_dir, type='val', data_augmentation=False)

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True,
                                  persistent_workers=True, pin_memory=True)

    val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=batch_size, shuffle=False,
                                  persistent_workers=True, pin_memory=True)

    logger = ImageLogger_2D(batch_frequency=logger_freq, rescale=False, image_save_dir=image_save_dir)
    loss_logger = LossLogger()


    trainer = pl.Trainer(devices='5, 6, 7', callbacks=[checkpoint_callback, logger, loss_logger], accelerator='gpu', strategy='ddp_find_unused_parameters_true',precision=16,default_root_dir=ckpt_save_dir, max_epochs=epoch, log_every_n_steps=30)
    trainer.fit(model, train_dataloader, val_dataloader)



    train_losses = loss_logger.train_losses
    val_losses = loss_logger.val_losses

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(f'{model_name}.png')
    plt.show()
