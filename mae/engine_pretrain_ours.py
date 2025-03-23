import math
import sys
from typing import Iterable
from unittest.util import _MAX_LENGTH

from numpy import isin
import torch

import util.misc as misc
import util.lr_sched as lr_sched
import wandb
import torchvision.utils as vutils
import os
TESTSET_IDX = [0,716,1216,1716,2205,2694,2928,3205,3305,3535,3757,3924,4091,5015,5933,6570,7207,7920,8633]

def generate_token(text_processor, lang, device, max_length):
    if type(lang) is str:
        decoded_strings = [lang]
    else:
        decoded_strings = [s.decode('ascii', errors='replace') for s in lang]
    processed_lang = text_processor(text=decoded_strings, padding="max_length", return_tensors='pt', max_length=max_length, truncation=True)
    processed_lang = processed_lang.to(device)
    return processed_lang

def train_one_epoch_ours(model: torch.nn.Module,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, loss_scaler,
                         log_writer=None,
                         args=None,
                         text_processor=None,
                         momentum_scheduler=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if 'tt' in args.model:
        metric_logger.add_meter('loss_pred', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_complete', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # training loop (steps)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # get batch
        img1, img2, lang, pick, place = batch
        if isinstance(img1, list):
            img1 = list(i.to(device, non_blocking=True).half() for i in img1)
        else:
            img1 = img1.to(device, non_blocking=True).half()
        if isinstance(img2, list):
            img2 = list(i.to(device, non_blocking=True).half() for i in img2)
        else:
            img2 = img2.to(device, non_blocking=True).half()
        pick = pick.to(device, non_blocking=True).half()
        place = place.to(device, non_blocking=True).half()

        # tokenizer
        # put the tokenizer here to avoid deadlock caused by the fork of the tokenizer
        max_length = 20 if 'voltron' in args.model else 77
        processed_lang = generate_token(text_processor, lang, device, max_length)       

        # model forward
        with torch.cuda.amp.autocast():
            loss, _, _ = model(img1, img2, pick, place, processed_lang, mask_ratio=args.mask_ratio)

        if isinstance(loss, tuple):
            loss_pred, loss_complete = loss
            loss = loss_pred + 0.1 * loss_complete
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # update momentum encoder2
        if momentum_scheduler is not None:            
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(model.module.blocks.parameters(), model.module.blocks2.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                for param_q, param_k in zip(model.module.patch_embed.parameters(), model.module.patch_embed2.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                for param_q, param_k in zip(model.module.norm.parameters(), model.module.norm2.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_pred=loss_pred.item())
        metric_logger.update(loss_complete=loss_complete.item())

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
            
            max_length = 20 if 'voltron' in args.model else 77
            processed_lang = generate_token(text_processor, lang, device, max_length)    
            if 'cv' in args.model: img2_input = list([img2,img2])
            else: img2_input = img2
            loss, predict, mask = model(img1, img2_input, pick, place, processed_lang, mask_ratio=args.mask_ratio)
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

                if mask is not None:
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
                elif predict is not None: # no mask but prediction
                    predict = model.unpatchify(predict)
                    predict = predict.detach().cpu()
                    predict = predict[0]
                    combined_image = torch.cat((img1, img2, predict), dim=2)
                else: # no mask and no prediction
                    continue

                if log_writer is not None:
                    combined_image = combined_image.permute(1, 2, 0).numpy()
                    image = wandb.Image(combined_image, caption=lang)
                    log_writer.log({f'validation_vis_{data_iter_step}': [image]}, epoch)               
                else:
                    os.makedirs('vis_tmp', exist_ok=True)
                    vutils.save_image(combined_image, f'vis_tmp/vis_{args.model}_{data_iter_step}.png')
                    print("saved image to vis_tmp/vis_{}.png".format(data_iter_step))

    loss /= len(data_loader)
    if log_writer is not None:
        log_writer.log({'validation_loss': loss}, epoch)

