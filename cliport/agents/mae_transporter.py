"""
Agents for MAE model
"""
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionMAE, OneStreamAttentionMAEFixSize, OneStreamAttentionMAEFixSize2Loss
from cliport.models.streams.one_stream_transport_lang_fusion import OneStreamTransportMAE, OneStreamTransportMAEFixSize, OneStreamTransportMAEFixSize2Loss
from cliport.utils import utils
import cliport.utils.visual_utils as vu

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


class MAESegDPTTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg_dpt'
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


class MAESegDPT3TransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    """
    DPT for picking and seg2 head for placing
    """
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn1 = 'mae_seg_dpt'
        stream_fcn2 = 'mae_seg2'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn1, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn2, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class MAESegDPT2LossTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'mae_seg_dpt_2loss'
        self.attention = OneStreamAttentionMAEFixSize2Loss(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize2Loss(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        loss_rgb = out['rgb_loss']
        out = out['out']

        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        loss = loss + 0.00005 * loss_rgb.sum()

        # Backpropagate.
        if backprop:
            attn_optim, _ = self.optimizers()
            if self.sch:
                s_att, _ = self.lr_schedulers()
                s_att.step(epoch=self.current_epoch)
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)['out']
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err
    
    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        
        loss_rgb = output['rgb_loss']
        output = output['out']

        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)

        loss = loss + 0.00005 * loss_rgb.sum()

        if backprop:
            _, transport_optim = self.optimizers()
            if self.sch:
                _, s_trans = self.lr_schedulers()
                s_trans.step(epoch=self.current_epoch)
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()
 
        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf = self.trans_forward(inp)['out']
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
                'theta': np.absolute((theta - p1_theta) % np.pi)
            }
        self.transport.iters += 1
        return err, loss
    
    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)

        # save attention map in validation
        if backprop is False and compute_err is True and self.logger is not None and self.save_visuals == 0:
            image = inp_img[:, :, :3]
            image = vu.tensor_to_cv2_img(image, to_rgb=False)
            heatmap = out['out'].reshape(image.shape[0], image.shape[1]).detach().cpu().numpy()
            combined = vu.save_tensor_with_heatmap(image, heatmap,
                                                   filename=None, return_img=True)
            combined = combined[:, :, ::-1]
            self.logger.log_image(key='heatmap', images=[combined], caption=[lang_goal])
            self.save_visuals += 1
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)['out']
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)['out']
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }
    

