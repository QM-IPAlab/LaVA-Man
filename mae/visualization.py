"""
visulize cross attentions
"""

import argparse
import enum
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import timm

import util.misc as misc
from dataset_mae import MAEDataset

from visualizer import get_local
get_local.activate() # 激活装饰器
import models_lib
from mae.engine_pretrain_ours import generate_token
from transformers import AutoTokenizer
import torchvision.utils as vutils
import os
import cv2
import matplotlib.pyplot as plt
assert timm.__version__ == "0.3.2"  # version check
MEAN = [0.1867, 0.1683, 0.1569]
STD = [0.1758, 0.1402, 0.1236]
MEAN_CLIPORT = [0.48145466, 0.4578275, 0.40821073]
STD_CLIPORT = [0.26862954, 0.26130258, 0.27577711]
PATH = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset_no_aug.hdf5'
TEST_PATH = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset_no_aug_all_test.hdf5'
TESTSET_IDX = [0,716,1216,1716,2205,2694,2928,3205,3305,3535,3757,3924,4091,5015,5933,6570,7207,7920,8633]

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_robot_lang', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default=PATH, type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./debug',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log', action='store_true',
                        help='log training process')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # ours parameters
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--save_ca', action='store_true', help='save cross attention maps')

    return parser


def get_fix_transform():
    trasform_fix = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])
    return trasform_fix


def visualize_cross_attentions(cross_attentions, 
                               tokenized_prompt, 
                               image, 
                               idx=None, 
                               folder=None):
    """
    cross_attentions: (b, query_h, query_w, key),e.g., (1, 20, 10, 77)
    """    
    assert cross_attentions.shape[0] == 1, 'only support batch size 1'
    idx = idx if idx is not None else 0
    folder = folder if folder is not None else 'vis_cross'
    # fig
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))    
    for j, ax in enumerate(axes.ravel()):
        
        # last map, show and then break
        if j >= len(tokenized_prompt):
            ax.imshow(image)
            break  # 

        a = cross_attentions[0, :, :, j]
        a = cv2.resize(a, (image.shape[1], image.shape[0]))

        heatmap = ax.imshow(a, cmap='jet', alpha=0.6)
        ax.imshow(image, alpha=0.4)  # 
        ax.set_title(tokenized_prompt[j])
        ax.axis('off')
        plt.colorbar(heatmap, ax=ax)  # 
    
    plt.tight_layout()  
    os.makedirs(folder, exist_ok=True)
    saved_name = os.path.join(folder, f'vis_ca{idx:06d}.png')
    plt.savefig(saved_name)
    print(f'save to {saved_name}')
    plt.close()

def encode_and_decode(prompt,processor):
    with torch.no_grad():
        if type(prompt) is str:
            decoded_strings = [prompt]
        else:
            decoded_strings = [s.decode('ascii') for s in prompt]
        inputs = processor(text=decoded_strings, return_tensors="pt", padding=True)
        token_ids = inputs['input_ids'][0].tolist()
        tokenized_prompt = [processor.decode([token_id]) for token_id in token_ids]
    return tokenized_prompt

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = get_fix_transform()
    dataset_vis = MAEDataset(transform=transform_train, data_path=TEST_PATH)

    data_loader_vis = torch.utils.data.DataLoader(
        dataset_vis, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=args.pin_mem
    )
    text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # define the model
    model = models_lib.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    print("Imported model: %s" % args.model)
    model.to(device)

    if args.pretrain:
        misc.dynamic_load_pretrain(model, args.pretrain)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
   
    for data_iter_step, batch in enumerate(data_loader_vis):
        
        if data_iter_step in TESTSET_IDX or data_iter_step-1 in TESTSET_IDX:
            img1, img2, lang, pick, place = batch
            img1 = img1.to(device, non_blocking=True).float()
            img2 = img2.to(device, non_blocking=True).float()
            pick = pick.to(device, non_blocking=True).float()
            place = place.to(device, non_blocking=True).float()

            processed_lang = generate_token(text_processor, lang, device)
            with torch.no_grad():
                loss, predict, mask = model(img1, img2, pick, place, lang, mask_ratio=args.mask_ratio)

                img1 = img1.detach().cpu()
                img1 = img1[0]
                # normalize to [0, 1]
                img1 = (img1 - img1.min()) / (img1.max() - img1.min())    

                cache = get_local.cache
                cross_attns = cache['CrossAttention.forward.attn']

                cross_attns = cross_attns[-1]
                # average the multi head
                cross_attns = cross_attns.mean(axis=1) # (b,201,77)
                cls_map = cross_attns[:, :1, :]
                attns = cross_attns[:, 1:, :]
                attns = attns.reshape(-1, 20, 10, 77)

                processor = model.text_processor
                tokens = encode_and_decode(lang,processor)

                # visualize cross attentions
                img1 = img1.permute(1, 2, 0).numpy()
                visualize_cross_attentions(attns, tokens, img1, idx=data_iter_step, folder='vis_cross')

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
