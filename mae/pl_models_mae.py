"""Pytorch Lightning module for MAE model."""

import lightning as L
from mae.models_mae_robot_lang import MAERobotLang
import torch
import torchvision
from torch import distributed
import os
import math
import sys
from typing import Any, Tuple, Union


class MAEPLRobotLang(MAERobotLang, L.LightningModule):
    """Pytorch Lightning module for MAE model."""
    def __init__(self, mask_ratio, save_imgs_every, kwargs):  
        super(MAEPLRobotLang, self).__init__(**kwargs)
        L.LightningModule.__init__(self)
        
        self.mask_ratio = mask_ratio 


    def training_step(self, batch: Any, batch_idx: int):
        img1, img2, lang, pick, place = batch
        loss, _, _ = self(img1, img2, pick, place, lang, mask_ratio=self.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        self.log('train_loss', loss)
        self.log('lr', self.lr)
        
        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int):
        img1, img2, lang, pick, place = batch
        loss, pred, _ = self(img1, img2, pick, place, lang, mask_ratio=self.mask_ratio)
        
        self.log('val_loss', loss, sync_dist=True)

        if self.save_imgs_every:
            p = int(self.save_imgs_every)
            if self.trainer.current_epoch % p == 0:
                nb = self.trainer.num_val_batches[0]
                ns = self.num_save_imgs
                per_batch = math.ceil(ns / nb)
                self.saved_imgs_list.append(pred[:per_batch])

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        
        batch_size = self.trainer.train_dataloader.batch_size
        
        lr_scale = devices * nodes * batch_size / self.base_batch_size
        lr = self.lr * lr_scale

        optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=(.9, .95), weight_decay=0.05)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            cycle_momentum=False,
        )

        return {
                    'optimizer': optim, 
                    'lr_scheduler': {'scheduler': schedule, 'interval': 'step'}
                }

    def on_train_batch_end(self, *args, **kwargs):
        if self.trainer.global_step == 2 and self.trainer.is_global_zero:
            # print GPU memory usage once at beginning of training
            avail, total = torch.cuda.mem_get_info()
            mem_used = 100 * (1 - (avail / total))
            gb = 1024**3
            self.print(f'GPU memory used: {(total-avail)/gb:.2f} of {total/gb:.2f} GB ({mem_used:.2f}%)')
        if self.trainer.num_nodes > 1 or self.trainer.num_devices > 1:
            distributed.barrier()
 
    def on_validation_epoch_end(self):
        if self.save_imgs_every:
            if self.trainer.is_global_zero:
                imgs = torch.cat(self.saved_imgs_list, 0)
                self.saved_imgs_list.clear()
                self.save_imgs(imgs[:self.num_save_imgs])
            if self.trainer.num_nodes > 1 or self.trainer.num_devices > 1:
                distributed.barrier()

    @L.utilities.rank_zero_only
    def save_imgs(self, imgs: torch.Tensor):
        with torch.no_grad():
            r = int(imgs.shape[0]**0.5)
            imgs = self.mae.tokens_as_image(imgs.detach())
            imgs = imgs.add_(1).mul_(127.5).clamp_(0, 255).byte()
            imgs = torchvision.utils.make_grid(imgs, r).cpu()
            epoch = self.trainer.current_epoch
            dir = os.path.join(self.trainer.log_dir, 'imgs')
            os.makedirs(dir, exist_ok=True)
            torchvision.io.write_png(imgs, os.path.join(dir, f'epoch_{epoch}_imgs.png'))