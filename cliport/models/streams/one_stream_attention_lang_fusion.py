"""Attention module."""
import numpy as np

import cliport.models as models
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion


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