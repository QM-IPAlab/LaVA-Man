import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.utils import utils
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTranslangFusionLatBatch
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat


class AttentionTrainer(LightningModule, TwoStreamAttentionLangFusionLat):
    """ Train the attention module only.
    """
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)                               

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # Padding
        inp_img = inp_img.permute(0, 3, 1, 2)
        pad_left_right = int(self.padding[1][0]), int(self.padding[1][1])
        pad_top_bottom = int(self.padding[0][0]), int(self.padding[0][1])
        pad_all = pad_left_right + pad_top_bottom
        in_data = F.pad(inp_img, pad_all, mode='constant')

        #FIXME: no rotation here now
        logits = self.attend(in_data, lang_goal)

        # Crop the padding
        h_start = pad_top_bottom[0]
        h_end = -pad_top_bottom[1] if pad_top_bottom[1] != 0 else None
        w_start = pad_left_right[0]
        w_end = -pad_left_right[1] if pad_left_right[1] != 0 else None
        logits = logits[:, :, h_start:h_end, w_start:w_end]
        
        logits = logits.permute(0, 2, 3, 1)  # [B W H 1]
        output = logits.reshape(inp_img.shape[0], -1)
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(inp_img.shape[0], *logits.shape[1:])
        return output

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.forward(inp_img, lang_goal, softmax=softmax)
        return out

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
        # Get the rotation index.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = torch.round(theta_i).long() % self.attention.n_rotations
        
        # Get label.
        inp_img = inp['inp_img']
        batch_size = inp_img.shape[0]
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
            theta_dist = np.sum(np.array(theta_dist))
            err = {
                'dist': dist,
                'theta': theta_dist
            }

        return loss, err

    def training_step(self, batch, batch_idx):
        self.train()
        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        total_loss = loss0
        self.log('tr/attn/loss', loss0)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)
        self.check_save_iteration(suffix='pick')

        return total_loss
    
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
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg['train']['lr'])
        return opt


class TransporterTrainer(LightningModule, TwoStreamTranslangFusionLatBatch):
    """ Train the transporter module only.
    """
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
    
    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}
        out = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err
    
    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        
        # Get the rotation index.
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta =(torch.round(itheta)).long() % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        batch_size = inp_img.shape[0]
        label_size = inp_img.shape[1:3] + (self.transport.n_rotations,)
        label = torch.zeros((batch_size,) + label_size, dtype=torch.float, device=output.device)
        batch_indices = torch.arange(batch_size, device=output.device)
        
        if isinstance(p, torch.Tensor):
            p = [p[:, 0].long(), p[:, 1].long()]
        label[batch_indices, p[0], p[1], itheta] = 1
        label = label.reshape(batch_size, -1)
        
        # Get loss.
        loss = self.cross_entropy_with_logits(output, label)
 
        # Pixel and Rotation error (not used anywhere).
        err = {}
        dist = []
        theta_dist = []
        if compute_err:
            place_conf = self.trans_forward(inp)
            place_conf = place_conf.detach().cpu().numpy()
            
            for i in range(batch_size):
                argmax = np.argmax(place_conf[i])
                argmax = np.unravel_index(argmax, shape=place_conf[i].shape)
                p1_pix = argmax[:2]
                p1_theta = argmax[2] * (2 * np.pi / place_conf[i].shape[2])
                q_numpy = [q[0][i].cpu().numpy(), q[1][i].cpu().numpy()]
                dist.append(np.linalg.norm(np.array(q_numpy) - p1_pix, ord=1))
                theta_dist.append(np.absolute((theta[i].cpu().numpy() - p1_theta) % np.pi))

            dist = np.sum(np.array(dist))
            theta_dist = np.sum(np.array(theta_dist))
            err = {
                'dist': dist,
                'theta': theta_dist
            }

        self.transport.iters += 1
        return err, loss

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = inp['lang_goal']
        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out

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
        self.check_save_iteration(suffix='place')

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self.transport.eval()

        loss1 = 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
            loss1 += l1
        loss1 /= self.val_repeats

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(loss1)

        return dict(
            val_loss=loss1,
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )
    
    def validation_epoch_end(self, all_outputs):
        mean_val_loss1 = np.mean([v['val_loss'].item() for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_loss1,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        return opt
