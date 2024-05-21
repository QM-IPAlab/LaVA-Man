"""Attention module."""
import numpy as np
import torch
import cliport.models as models
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion
import torch.nn.functional as F


class OneStreamAttentionLangFusion(TwoStreamAttentionLangFusion):
    """Attention (a.k.a Pick) module with language features fused at the bottleneck."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, l):
        x = self.attn_stream_one(x, l)
        return x


class OneStreamAttentionMAE(TwoStreamAttentionLangFusion):
    """Attention (a.k.a Pick) module with language features fused at the bottleneck."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.pretrain = cfg['pretrain_path']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.fusion_type = cfg['train']['attn_stream_fusion_type']

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg,
                                                self.device, self.preprocess,
                                                pretrain_path=self.pretrain)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, lang):
        x = self.attn_stream_one(x, lang)
        return x


class OneStreamAttentionMAEFixSize(TwoStreamAttentionLangFusion):
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.pretrain = cfg['pretrain_path']
        self.ori_inshape = in_shape
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.fusion_type = cfg['train']['attn_stream_fusion_type']


    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        self.attn_stream_one = stream_one_model(self.ori_inshape, 1, self.cfg,
                                                self.device, self.preprocess,
                                                pretrain_path=self.pretrain)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, lang):
        x = self.attn_stream_one(x, lang)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""

        in_data = inp_img
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)  # same as unqueeze(0)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output

