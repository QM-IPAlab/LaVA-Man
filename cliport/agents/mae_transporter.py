"""
Agents for MAE model
"""
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionMAE, OneStreamAttentionMAEFixSize
from cliport.models.streams.one_stream_transport_lang_fusion import OneStreamTransportMAE, OneStreamTransportMAEFixSize
from cliport.utils import utils

import numpy as np
import torch


class MAETransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae'
        self.attention = OneStreamAttentionMAE(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAE(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAESegTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAESeg2TransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg2'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAESeg2DepthTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg2_depth'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAESeg2TransporterAgentRenor(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg2'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess_norm,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess_norm,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAESeg2ModelFullMaskAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg2_fm'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess_norm,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess_norm,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAEFixTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAEFixGloss(MAEFixTransporterAgent):

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1

        # #new 
        # import math
        # from skimage.draw import polygon

        # dimension = 4
        # half_size = dimension // 2
        # vertices = np.array([
        #     [-half_size, -half_size],
        #     [half_size, -half_size],
        #     [half_size, half_size],
        #     [-half_size, half_size]
        # ])
        # # rotation_matrix = np.array([
        # #     [math.cos(theta), -math.sin(theta)],
        # #     [math.sin(theta), math.cos(theta)]
        # # ])
        # # rotated_vertices = np.dot(vertices, rotation_matrix)
        # centered_vertices = vertices + np.array(p)
        # rr, cc = polygon(centered_vertices[:, 0], centered_vertices[:, 1], label.shape)
        # label[rr, cc] = 1

        # Define the center of the Gaussian circle (peak point)
        y, x = p[0], p[1]
        
        # Apply a Gaussian filter to create a circle around the point with a peak at the center
        # Define the standard deviations for y and x dimensions
        sigma_y, sigma_x = 4, 4  # You may want to adjust these values
        
        # Generate a grid of (x,y) coordinates
        yv, xv = np.meshgrid(np.arange(label_size[0]), np.arange(label_size[1]), indexing='ij')
        
        # Calculate the Gaussian distribution
        label[:, :, theta_i] = np.exp(-((yv - y)**2 / (2 * sigma_y**2) + (xv - x)**2 / (2 * sigma_x**2)))
        
        # Normalize the distribution so that the peak is at 1
        label[:, :, theta_i] /= np.max(label[:, :, theta_i])

        # import pdb; pdb.set_trace()
        # from cliport.utils.visual_utils import save_tensor_with_heatmap        
        # save = save_tensor_with_heatmap(inp_img[:,:,:3].astype(np.uint8), label[:,:,0],'vis_loss.png')

        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            
            # the prediced location is here
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=2),
                'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err
    

class MAEFixBloss(MAEFixTransporterAgent):
    """Blokcy loss like MAE"""

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1

        # Gaussian loss
        y, x = p[0], p[1]
        sigma_y, sigma_x = 4, 4  # You may want to adjust these values
        yv, xv = np.meshgrid(np.arange(label_size[0]), np.arange(label_size[1]), indexing='ij')
        label[:, :, theta_i] = np.exp(-((yv - y)**2 / (2 * sigma_y**2) + (xv - x)**2 / (2 * sigma_x**2)))
        label[:, :, theta_i] /= np.max(label[:, :, theta_i])

        label = label.transpose((2, 0, 1))  # n_rotations, h, w,
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)
        out = out.reshape(label.shape)

        # Get loss.
        loss = (label - out) ** 2
        loss = loss.mean()

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            
            # the prediced location is here
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=2),
                'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err
    

class MAESegBaseAgent(TwoStreamClipLingUNetTransporterAgent):
    """ The agent that can use different mae models"""

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg_base'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        ) 


class MAESeg3TransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg3'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAEFeatUpTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_featup'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

class MAESegCLIPModel(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_clip'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )