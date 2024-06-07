import math
import sys
from tkinter import N
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import wandb
import torchvision.utils as vutils
import os
TESTSET_IDX = [0,716,1216,1716,2205,2694,2928,3205,3305,3535,3757,3924,4091,5015,5933,6570,7207,7920,8633]

def generate_token(text_processor, lang, device):
    if type(lang) is str:
        decoded_strings = [lang]
    else:
        decoded_strings = [s.decode('ascii') for s in lang]
    processed_lang = text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
    processed_lang = processed_lang.to(device)
    return processed_lang

def train_one_epoch_ours(model: torch.nn.Module,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, loss_scaler,
                         log_writer=None,
                         args=None,
                         text_processor=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        img1, img2, lang, pick, place = batch
        img1 = img1.to(device, non_blocking=True).half()
        img2 = img2.to(device, non_blocking=True).half()
        pick = pick.to(device, non_blocking=True).half()
        place = place.to(device, non_blocking=True).half()

        # put the tokenizer here to avoid deadlock caused by the fork of the tokenizer
        processed_lang = generate_token(text_processor, lang, device)       

        with torch.cuda.amp.autocast():
            loss, _, _ = model(img1, img2, pick, place, processed_lang, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.log({
                'train_loss': loss_value_reduce,
                'lr': lr,
                }, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_vis_img2(model: torch.nn.Module,
                      data_loader: Iterable,
                      device: torch.device,
                      epoch: int,
                      log_writer=None,
                      args=None,
                      text_processor=None):
    model.eval()
    loss = 0

    for data_iter_step, batch in enumerate(data_loader):
        img1, img2, lang, pick, place = batch
        img1 = img1.to(device, non_blocking=True).float()
        img2 = img2.to(device, non_blocking=True).float()
        pick = pick.to(device, non_blocking=True).float()
        place = place.to(device, non_blocking=True).float()

        with torch.no_grad():
            
            processed_lang = generate_token(text_processor, lang, device)       
            loss, predict, mask = model(img1, img2, pick, place, processed_lang, mask_ratio=args.mask_ratio)
            loss += loss.item()

            if data_iter_step in TESTSET_IDX or data_iter_step-1 in TESTSET_IDX:
                img1 = img1.detach().cpu()
                img1 = img1[0]
                # normalize to [0, 1]
                img1 = (img1 - img1.min()) / (img1.max() - img1.min())

                # original image
                img2 = img2.detach().cpu()
                img2 = img2[0]
                img2 = (img2 - img2.min()) / (img2.max() - img2.min())

                # masked image
                mask = mask.detach().cpu()
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
                mask = mask[0]
                mask = mask.permute(2, 0, 1)
                im_masked = img2 * (1 - mask)

                # MAE reconstruction
                predict = model.unpatchify(predict)
                predict = predict.detach().cpu()
                predict = predict[0]
                predict = (predict - predict.min()) / (predict.max() - predict.min())
                im_paste = img2 * (1 - mask) + predict * mask

                combined_image = torch.cat((img1, img2, im_masked, predict, im_paste), dim=2)
                    
                if log_writer is not None:
                    combined_image = combined_image.permute(1, 2, 0).numpy()
                    image = wandb.Image(combined_image, caption=lang)
                    log_writer.log({f'validation_vis_{data_iter_step}': [image]}, epoch)
                
                else:
                    os.makedirs('vis_tmp', exist_ok=True)
                    vutils.save_image(combined_image, f'vis_tmp/vis_{args.model}_{data_iter_step}.png', normalize=True, range=(0, 1))
                    print("saved image to vis_tmp/vis_{}.png".format(data_iter_step))

    loss /= len(data_loader)
    if log_writer is not None:
        log_writer.log({'validation_loss': loss}, epoch)

