import torch
import numpy as np
import torch.nn.functional as F
import cliport.models as models
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion


class OneStreamTransportLangFusion(TwoStreamTransportLangFusion):
    """Transport (a.k.a) Place module with language features fused at the bottleneck"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device,
                                                 self.preprocess)

        print(f"Transport FCN: {stream_one_fcn}")

    def transport(self, in_tensor, crop, l):
        logits = self.key_stream_one(in_tensor, l)
        kernel = self.query_stream_one(crop, l)
        return logits, kernel


class OneStreamTransportMAE(TwoStreamTransportLangFusion):
    """Transport (a.k.a) Place module with language features fused at the bottleneck"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.pretrain = cfg['pretrain_path']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        self.fusion_type = cfg['train']['trans_stream_fusion_type']


    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        self.key_stream_one = stream_one_model((384, 224), self.output_dim, self.cfg,
                                               self.device, self.preprocess, pretrain_path=self.pretrain)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg,
                                                 self.device, self.preprocess, pretrain_path=self.pretrain)

        print(f"Transport FCN: {stream_one_fcn}")

    def transport(self, in_tensor, crop, lang):
        logits = self.key_stream_one(in_tensor, lang)
        kernel = self.query_stream_one(crop, lang)
        return logits, kernel


class OneStreamTransportMAEFixSize(TwoStreamTransportLangFusion):

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.pretrain = cfg['pretrain_path']
        self.mae_model = cfg['mae_model']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        self.fusion_type = cfg['train']['trans_stream_fusion_type']

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg,
                                               self.device, self.preprocess, model_name=self.mae_model,
                                               pretrain_path=self.pretrain)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg,
                                                 self.device, self.preprocess, model_name=self.mae_model,
                                                 pretrain_path=self.pretrain)

        print(f"Transport FCN: {stream_one_fcn}")

    def transport(self, in_tensor, crop, lang):
        logits = self.key_stream_one(in_tensor, lang)
        kernel = self.query_stream_one(crop, lang)
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        in_tensor_ori = in_tensor[:, :, hcrop:-hcrop, hcrop:-hcrop]
        logits, kernel = self.transport(in_tensor_ori, crop, lang_goal)

        return self.correlate(logits, kernel, softmax)


    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output


class OneStreamTransportMAEFixSize2Loss(OneStreamTransportMAEFixSize):

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        in_tensor_ori = in_tensor[:, :, hcrop:-hcrop, hcrop:-hcrop]
        logits, kernel = self.transport(in_tensor_ori, crop, lang_goal)

        logits_loss = logits['rgb_loss']
        logits = logits['out']
        kernel = kernel['out']

        out = self.correlate(logits, kernel, softmax)
        return {'out': out, 'rgb_loss': logits_loss}

