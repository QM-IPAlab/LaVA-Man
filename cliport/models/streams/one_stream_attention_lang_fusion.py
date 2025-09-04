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
        self.mae_model = cfg['mae_model']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.fusion_type = cfg['train']['attn_stream_fusion_type']


    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        self.attn_stream_one = stream_one_model(self.ori_inshape, 1, self.cfg,
                                                self.device, self.preprocess,
                                                model_name=self.mae_model,
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


class OneStreamAttentionMAEFixSize2Loss(OneStreamAttentionMAEFixSize):

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
            out = self.attend(x, lang_goal)
            lgts = out['out']
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
        return {'out': output, 'rgb_loss': out['rgb_loss']}
    

class OneStreamAttentionMAEBatch(OneStreamAttentionMAEFixSize):
    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # Padding
        inp_img = inp_img.permute(0, 3, 1, 2)
        in_data = inp_img
        # No padding for MAE input
        # pad_left_right = int(self.padding[1][0]), int(self.padding[1][1])
        # pad_top_bottom = int(self.padding[0][0]), int(self.padding[0][1])
        # pad_all = pad_left_right + pad_top_bottom
        # in_data = F.pad(inp_img, pad_all, mode='constant')

        #FIXME: no rotation here now
        logits = self.attend(in_data, lang_goal)

        # Crop the padding
        # h_start = pad_top_bottom[0]
        # h_end = -pad_top_bottom[1] if pad_top_bottom[1] != 0 else None
        # w_start = pad_left_right[0]
        # w_end = -pad_left_right[1] if pad_left_right[1] != 0 else None
        # logits = logits[:, :, h_start:h_end, w_start:w_end]
        
        logits = logits.permute(0, 2, 3, 1)  # [B W H 1]
        output = logits.reshape(inp_img.shape[0], -1)
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(inp_img.shape[0], *logits.shape[1:])
        return output

    def predict(self, inp_img, lang_goal):
        """Forward pass."""
        
        # Padding
        inp_img = inp_img.permute(0, 3, 1, 2)
        in_data = inp_img

        #FIXME: no rotation here now
        prediction = self.attn_stream_one.predict(in_data, lang_goal)
        return prediction


class OneStreamAttentionMAEBatchRecon(OneStreamAttentionMAEBatch):
    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # Padding
        inp_img = inp_img.permute(0, 3, 1, 2)
        in_data = inp_img
        # No padding for MAE input
        # pad_left_right = int(self.padding[1][0]), int(self.padding[1][1])
        # pad_top_bottom = int(self.padding[0][0]), int(self.padding[0][1])
        # pad_all = pad_left_right + pad_top_bottom
        # in_data = F.pad(inp_img, pad_all, mode='constant')

        #FIXME: no rotation here now
        logits, recon_loss = self.attend(in_data, lang_goal)

        # Crop the padding
        # h_start = pad_top_bottom[0]
        # h_end = -pad_top_bottom[1] if pad_top_bottom[1] != 0 else None
        # w_start = pad_left_right[0]
        # w_end = -pad_left_right[1] if pad_left_right[1] != 0 else None
        # logits = logits[:, :, h_start:h_end, w_start:w_end]
        
        logits = logits.permute(0, 2, 3, 1)  # [B W H 1]
        output = logits.reshape(inp_img.shape[0], -1)
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(inp_img.shape[0], *logits.shape[1:])
        return output, recon_loss
