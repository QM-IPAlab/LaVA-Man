import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
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
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x


class TwoStreamAttnlangFusionLatBatch(TwoStreamAttentionLangFusionLat):
    """ Batch version of TwoStreamAttentionLangFusionLat. """
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        inp_img = inp_img.permute(0, 3, 1, 2)
        #FIXME: hard coded padding and then cropping
        pad = int(self.padding[1][0]), int(self.padding[1][1])
        in_data = F.pad(inp_img, pad, mode='constant')

        #FIXME: no rotation here now
        logits = self.attend(in_data, lang_goal)
       
        logits = logits[:, :, :, pad[0]:-pad[1]]
        logits = logits.permute(0, 2, 3, 1)  # [B W H 1]
        output = logits.reshape(inp_img.shape[0], -1)
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(inp_img.shape[0], *logits.shape[1:])
        return output


class TwoStreamAttentionMAEFixSize(TwoStreamAttentionLangFusion):
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.pretrain = cfg['pretrain_path']
        self.ori_inshape = in_shape
        self.mae_model = cfg['mae_model']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.fusion_type = cfg['train']['attn_stream_fusion_type']

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.ori_inshape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.ori_inshape, 1, self.cfg, self.device, self.preprocess,
                                                model_name=self.mae_model, pretrain_path=self.pretrain)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, lang):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, lang)
        x = self.fusion(x1, x2)
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