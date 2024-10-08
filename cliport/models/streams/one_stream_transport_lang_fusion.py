import torch
import numpy as np
import torch.nn.functional as F
import cliport.models as models
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion
from cliport.utils import utils

from visualizer import get_local

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
        # construct input tensor (padding for crop, but not for model input)
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

        # obtain the original image tensor
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


class OneStreamTransportMAEBatch(OneStreamTransportMAEFixSize):
    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        self.rotator = utils.ImageRotatorBatch(self.n_rotations)

    #@get_local('inp_img','out','logits','kernel')
    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass. Batch size version.
        Args:
            inp_img: Input image tensor [B H W C]
            lang_goal: Goal language tensor [B G]
            softmax: Apply softmax to output
        """
        # Pad input image.
        inp_img = inp_img.permute(0, 3, 1, 2) #[b, 6, 320, 160]
        pad_left_right = int(self.padding[1][0]), int(self.padding[1][1])
        pad_top_bottom = int(self.padding[0][0]), int(self.padding[0][1])
        pad_all = pad_left_right + pad_top_bottom
        in_data = F.pad(inp_img, pad_all, mode='constant') #[b, 6, 384, 224]

        # Find rotation pivot.
        if isinstance(p, torch.Tensor):
            p = [p[:, 0].long(), p[:, 1].long()]
        pv = [p[0] + self.pad_size, p[1] + self.pad_size]
        
        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        batch_indices = torch.arange(in_data.size(0)).view(-1, 1, 1).to('cuda')
        h_offsets_start = (pv[0]-hcrop).int()
        w_offsets_start = (pv[1]-hcrop).int()

        h_indices = h_offsets_start.view(-1, 1) + torch.arange(2*hcrop).view(1, -1).to('cuda')
        w_indices = w_offsets_start.view(-1, 1) + torch.arange(2*hcrop).view(1, -1).to('cuda')

        h_indices = h_indices.unsqueeze(2)  # Shape (batch_size, 64, 1)
        w_indices = w_indices.unsqueeze(1)  # Shape (batch_size, 1, 64)

        cropped_tensor = in_data[batch_indices, :, h_indices, w_indices] # Shape (batch_size, 64, 64, 6)
        cropped_tensor = cropped_tensor.permute(0, 3, 1, 2) # Shape (batch_size, 6, 64, 64)
        cropped_tensor = cropped_tensor.unsqueeze(1) # Shape (batch_size, 1, 6, 64, 64)
        cropped_tensor = cropped_tensor.repeat(1, self.n_rotations, 1, 1, 1) # Shape (batch_size, 36, 6, 64, 64)
        cropped_rotated = self.rotator(cropped_tensor, pivot=pv) # Shape (batch_size, 6, 64, 64)
        cropped_rotated = cropped_rotated.view(-1, 6, 64, 64)
        
        # image = cropped_rotated[:36,:3,:,:]
        # image = image/255
        # import torchvision
         
        # torchvision.utils.save_image(image, '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/batch_image2.png', nrow=8)
        # import pdb; pdb.set_trace()

        logits, kernel = self.transport(inp_img, cropped_rotated, lang_goal)
        out = self.correlate(logits, kernel, softmax)
        return out
    
    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        batch_size = in0.shape[0]
        in0 = in0.reshape(1, -1, in0.shape[-2], in0.shape[-1])
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size), groups=batch_size)
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output.reshape(batch_size, self.n_rotations, output.shape[2], output.shape[3])
        output = output.permute(0, 2, 3, 1)  # [B W H 1]
        output_shape = output.shape
        output = output.reshape(batch_size, -1)
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(batch_size, *output_shape[1:])
        return output

