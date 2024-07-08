import os
import numpy as np

from prometheus_client import instance_ip_grouping_key
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttnlangFusionLatBatch
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.utils import utils
import cliport.utils.visual_utils as vu

class PickAgent(TwoStreamClipLingUNetTransporterAgent):
    """ Train the attention module only.
    """
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _set_optimizers(self):
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
        }                                       

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttnlangFusionLatBatch(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)

        # save attention map in validation
        # if backprop is False and compute_err is True and self.logger is not None and self.save_visuals == 0:
        #     image = inp_img[:, :, :3]
        #     image = vu.tensor_to_cv2_img(image, to_rgb=False)
        #     heatmap = out.reshape(image.shape[0], image.shape[1]).detach().cpu().numpy()
        #     combined = vu.save_tensor_with_heatmap(image, heatmap,
        #                                            filename=None, return_img=True)
        #     combined = combined[:, :, ::-1]
        #     self.logger.log_image(key='heatmap', images=[combined], caption=[lang_goal])
        #     self.save_visuals += 1
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        inp_img = inp['inp_img']
        batch_size = inp_img.shape[0]
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = torch.round(theta_i).long() % self.attention.n_rotations
        label_size = inp_img.shape[1:3] + (self.attention.n_rotations,)
        # rotation as last dimenstion (h, w, rotation), not channel !
        label = torch.zeros((batch_size,) + label_size, dtype=torch.float, device=out.device)
        batch_indices = torch.arange(batch_size, device=out.device)
        
        if isinstance(p, torch.Tensor):
            p = [p[:, 0].long(), p[:, 1].long()]
        label[batch_indices, p[0], p[1], theta_i] = 1
        label = label.permute(0, 3, 1, 2).reshape(batch_size, -1)

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
        dist = []
        theta_dist = []
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()

            for i in range(batch_size):
                argmax = np.argmax(pick_conf[i])
                argmax = np.unravel_index(argmax, shape=pick_conf[i].shape)
                p0_pix = argmax[:2]
                p0_theta = argmax[2] * (2 * np.pi / pick_conf[i].shape[2])
                p_numpy = [p[0][i].cpu().numpy(), p[1][i].cpu().numpy()]
                dist.append(np.linalg.norm(np.array(p_numpy) - p0_pix, ord=1))
                theta_dist.append(np.absolute((theta[i].cpu().numpy() - p0_theta) % np.pi))
            
            dist = np.sum(np.array(dist))
            theta = np.sum(np.array(theta_dist))
            err = {
                'dist': dist,
                'theta': theta_dist
            }

        return loss, err

    def training_step(self, batch, batch_idx):
        self.attention.train()
        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        total_loss = loss0
        self.log('tr/attn/loss', loss0)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)
        self.check_save_iteration()

        return dict(        
            loss=total_loss,
        )
    
    def validation_step(self, batch, batch_idx):
        self.attention.eval()

        loss0 = 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
        loss0 /= self.val_repeats

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(loss0)

        return dict(
            val_loss=loss0,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
        )
    
    def validation_epoch_end(self, all_outputs):
        mean_val_loss0 = np.mean([v['val_loss'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))

        return dict(
            val_loss=mean_val_loss0,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
        )


class PlaceAgent(TwoStreamClipLingUNetTransporterAgent):
    """ Train the transporter module only.
    """
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def training_step(self, batch, batch_idx):
        self.transport.train()

        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss1, err1 = self.transport_training_step(frame)
        total_loss = loss1
        self.log('tr/trans/loss', loss1)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )
    
    def validation_step(self, batch, batch_idx):
        self.transport.eval()

        loss1 = 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
            loss1 += l1
        loss1 /= self.val_repeats
        val_total_loss = loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss1=loss1,
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )
    
    def validation_epoch_end(self, all_outputs):
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            mean_val_loss1=mean_val_loss1,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )