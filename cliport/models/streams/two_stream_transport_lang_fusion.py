import torch
import numpy as np
import torch.nn.functional as F
import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.transport import Transport
from torchvision.utils import save_image
from cliport.utils import utils

class TwoStreamTransportLangFusion(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
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

        logits, kernel = self.transport(in_tensor, crop, lang_goal)

        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        return self.correlate(logits, kernel, softmax)


class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel


class TwoStreamTranslangFusionLatBatch(TwoStreamTransportLangFusionLat):
    """ Batch version of TwoStreamAttentionLangFusionLat. """
    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        self.rotator = utils.ImageRotatorBatch(self.n_rotations)

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
        logits, kernel = self.transport(in_data, cropped_rotated, lang_goal)
        return self.correlate(logits, kernel, softmax)
    
    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        batch_size = in0.shape[0]
        in0 = in0.view(1, -1, 384, 224)
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size), groups=batch_size)
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        output = output.view(batch_size, 36, output.shape[2], output.shape[3])
        output = output.permute(0, 2, 3, 1)  # [B W H 1]
        output_shape = output.shape
        output = output.reshape(batch_size, -1)
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(batch_size, *output_shape[1:])
        return output


class TwoStreamTransportMAEFixSize(TwoStreamTransportLangFusion):
    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.pretrain = cfg['pretrain_path']
        self.mae_model = cfg['mae_model']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        self.fusion_type = cfg['train']['trans_stream_fusion_type']

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg,
                                               self.device, self.preprocess, model_name=self.mae_model,
                                               pretrain_path=self.pretrain)
        self.query_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg,
                                                 self.device, self.preprocess, model_name=self.mae_model,
                                                 pretrain_path=self.pretrain)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        print(f"Transport FCN: {stream_one_fcn}")

    def transport(self, in_tensor, crop, lang):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, lang)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, lang)
        kernel = self.fusion_query(query_out_one, query_out_two)

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