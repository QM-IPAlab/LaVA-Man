
# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Heads for downstream tasks
# --------------------------------------------------------

"""
A head is a module where the __init__ defines only the head hyperparameters.
A method setup(croconet) takes a CroCoNet and set all layers according to the head and croconet attributes.
The forward takes the features as well as a dictionary img_info containing the keys 'width' and 'height'
"""

import torch
import torch.nn as nn
from cliport.models.core.dpt_block import DPTOutputAdapter, DPTOutputAdapterOurs


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for CroCo.
    by default, hooks_idx will be equal to:
    * for encoder-only: 4 equally spread layers
    * for encoder+decoder: last encoder + 3 equally spread layers of the decoder 
    """

    def __init__(self, *, hooks_idx=None, layer_dims=[96,192,384,768],
                 output_width_ratio=1, num_channels=1, postprocess=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_blocks = True # backbone needs to return all layers 
        self.postprocess = postprocess
        self.output_width_ratio = output_width_ratio
        self.num_channels = num_channels
        self.hooks_idx = hooks_idx # set in setup
        self.layer_dims = layer_dims
    
    def setup(self):
        dpt_args = {'output_width_ratio': self.output_width_ratio, 'num_channels': self.num_channels}
        if self.hooks_idx is None:
            step = 3
            hooks_idx = [8+12-1-i*step for i in range(3,-1,-1)]
            # in my case, enc_depth=12 and dec_depth=8
            # so hooks_idx = [11, 14, 17, 20]
            self.hooks_idx = hooks_idx
            print(f'  PixelwiseTaskWithDPT: automatically setting hook_idxs={self.hooks_idx}')
        dpt_args['hooks'] = self.hooks_idx
        dpt_args['layer_dims'] = self.layer_dims
        self.dpt = DPTOutputAdapter(**dpt_args)
        dim_tokens = [768 if hook<12 else 512 for hook in self.hooks_idx]
        dpt_init_args = {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)


    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info['height'],img_info['width']))
        if self.postprocess: out = self.postprocess(out)
        return out


class PixelwiseTaskWithDPTOurs(PixelwiseTaskWithDPT):
    """ DPT module for CroCo.
    by default, hooks_idx will be equal to:
    * for encoder-only: 4 equally spread layers
    * for encoder+decoder: last encoder + 3 equally spread layers of the decoder 
    """
    def setup(self):
        dpt_args = {'output_width_ratio': self.output_width_ratio, 'num_channels': self.num_channels}
        if self.hooks_idx is None:
            step = 3
            hooks_idx = [8+12-1-i*step for i in range(3,-1,-1)]
            # in my case, enc_depth=12 and dec_depth=8
            # so hooks_idx = [11, 14, 17, 20]
            self.hooks_idx = hooks_idx
            print(f'  PixelwiseTaskWithDPT: automatically setting hook_idxs={self.hooks_idx}')
        dpt_args['hooks'] = self.hooks_idx
        dpt_args['layer_dims'] = self.layer_dims
        self.dpt = DPTOutputAdapterOurs(**dpt_args)
        dim_tokens = [768 if hook<12 else 512 for hook in self.hooks_idx]
        dpt_init_args = {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info, rgb=None):
        out = self.dpt(x, image_size=(img_info['height'],img_info['width']), rgb=rgb)
        if self.postprocess: out = self.postprocess(out)
        return out