import os
import numpy as np
import math

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
import cliport.utils.visual_utils as vu
from cliport.models.core.attention import Attention

class TransporterAgentSep(LightningModule):
    """
        Our trasnporter agent.
        Train the attention and transport model separately,
        And support for batch size training + learning scheduler
    """
    def __init__(self, name, cfg, train_ds, test_ds, mode):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']
        self.save_visuals = 0
        self._build_model()
        self.automatic_optimization = False
        #self._set_optimizers()
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

        self.total_steps = self.cfg['train']['n_steps']
        self.warmup_epochs = self.cfg['train']['warmup_epochs']
        self.sch = cfg['train']['lr_scheduler']
        self.lr = cfg['train']['lr']
        self.lr_min = cfg['train']['lr_min']

        self.mode = mode # train or test
        assert self.mode in ['pick', 'place', 'both'], "Mode should be either pick, place, or both"

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()
    
    def configure_optimizers(self):
        
        opt_attn = torch.optim.AdamW(self.attention.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.95))
        opt_trans = torch.optim.AdamW(self.transport.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.95)) 
        self.max_epochs = self.trainer.max_epochs
        
        # configure learning rate schedulr
        if self.sch:
            print('Using cosine annealing learning rate scheduler with warm up !')
        
            def sch_foo(epoch):
                """Decay the learning rate with half-cycle cosine after warmup"""
                if epoch < self.warmup_epochs:
                    lr = self.lr * (epoch+1) / self.warmup_epochs 
                else:
                    lr = self.lr_min + (self.lr - self.lr_min) * 0.5 * \
                        (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
                return lr/self.lr
            
            lrs_attn = torch.optim.lr_scheduler.LambdaLR(opt_attn, lr_lambda=sch_foo)
            lrs_trans = torch.optim.lr_scheduler.LambdaLR(opt_trans, lr_lambda=sch_foo)
         
            pick_optim = {"optimizer": opt_attn, "lr_scheduler": lrs_attn}
            place_optim = {"optimizer": opt_trans, "lr_scheduler": lrs_trans}
        else:
            pick_optim = opt_attn
            place_optim = opt_trans

        # return pick and place
        if self.mode == 'pick':
            return pick_optim
        elif self.mode == 'place':
            return place_optim
        else:
            return pick_optim, place_optim
        
    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)

        # save attention map in validation
        if backprop is False and compute_err is True and self.logger is not None and self.save_visuals == 0:
            import pdb; pdb.set_trace()
            image = inp_img[0, :, :, :3]
            image = vu.tensor_to_cv2_img(image, to_rgb=False)
            heatmap = out[0].reshape(image.shape[0], image.shape[1]).detach().cpu().numpy()
            combined = vu.save_tensor_with_heatmap(image, heatmap,
                                                   filename=None, return_img=True)
            combined = combined[:, :, ::-1]
            self.logger.log_image(key='heatmap', images=[combined], caption=[lang_goal])
            self.save_visuals += 1

        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get the rotation index
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

        # Choose optimizer and learning rate scheduler.
        if backprop:
            if self.mode == 'pick':
                attn_optim = self.optimizers()
                if self.sch: s_att  = self.lr_schedulers()
            elif self.mode == 'both':
                attn_optim, _ = self.optimizers()
                if self.sch: s_att, _ = self.lr_schedulers()
            else:
                raise NotImplementedError()
            
            # Back prop and step
            s_att.step(epoch=self.current_epoch)
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
            theta_dist = np.sum(np.array(theta_dist))
            err = {
                'dist': dist,
                'theta': theta_dist
            }
        return loss, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = inp['lang_goal']
        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out
    
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
        
        # Choose optimizer and learning rate scheduler.
        if backprop:
            if self.mode == 'place': 
                transport_optim = self.optimizers()
                if self.sch: s_trans = self.lr_schedulers()
            elif self.mode == 'both':
                _, transport_optim = self.optimizers()
                if self.sch: _, s_trans = self.lr_schedulers()
            else:
                raise NotImplementedError()
            
            # Back prop and step
            s_trans.step(epoch=self.current_epoch)    
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()
 
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
        return err, loss

    def training_step(self, batch, batch_idx, optimizer_idx):        
        if self.attention is not None: self.attention.train()  
        if self.attention is not None: self.transport.train()
        frame, _ = batch

        # Get training losses
        if self.mode == 'both':
            loss0, err0 = self.attn_training_step(frame)
            if isinstance(self.transport, Attention):
                loss1, err1 = self.attn_training_step(frame)
            else:
                loss1, err1 = self.transport_training_step(frame)
            total_loss = loss0 + loss1
            self.log('tr/attn/loss', loss0)
            self.log('tr/trans/loss', loss1)
            self.log('tr/loss', total_loss)
        
        elif self.mode == 'pick':
            loss0, err0 = self.attn_training_step(frame)
            total_loss = loss0
            self.log('tr/attn/loss', loss0)

        elif self.mode == 'place':
            loss1, err1 = self.transport_training_step(frame)
            total_loss = loss1
            self.log('tr/trans/loss', loss1)
            
        # final loss and checkpoint
        self.trainer.train_loop.running_loss.append(total_loss)
        self.check_save_iteration(suffix=self.mode)
        self.total_steps += 1 
        return total_loss

    def check_save_iteration(self, suffix='None'):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            filename = f"{suffix}-{filename}"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def on_validation_epoch_start(self) -> None:
        self.save_visuals = 0

    def validation_step(self, batch, batch_idx):
        if self.attention is not None: self.attention.eval() 
        if self.transport is not None: self.transport.eval()
        frame, _ = batch

        #XXX: Not support val_repeat > 1
        assert self.val_repeats ==1, "Not support val_repeat > 1 currently"

        # Init recordings
        loss0, loss1 = 0, 0
        err0 = {'dist': 0, 'theta': 0}
        err1 = {'dist': 0, 'theta': 0}
        
        if self.mode == 'both':        
            loss0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            if isinstance(self.transport, Attention):
                loss1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
            else:
                loss1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
        elif self.mode == 'pick':
            loss0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
        elif self.mode == 'place':
            loss1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
        
        # totoal loss and return
        val_total_loss = loss0 + loss1
        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)
        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch+1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
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

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)

    def test_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)
        

        # #import pdb; pdb.set_trace()
        # img = frame['img'][:,:,:3]
        # pick_place = frame['p0']
        # place_place = frame['p1']
        # pick_radius = frame['pick_radius']
        # place_radius = frame['place_radius']
        # text = frame['lang_goal']

        # img = img.astype(np.uint8)

        # brg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.circle(brg, (pick_place[1], pick_place[0]), int(pick_radius), (0, 255, 0), 2)
        # cv2.circle(brg, (place_place[1], place_place[0]), int(place_radius), (0, 0, 255), 2)
        # cv2.putText(brg, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        # foler = 'data_gt_real_images'
        # idx = len(os.listdir(foler))
        # cv2.imwrite(f'data_gt_real_images/real{idx}.png', brg)
        

        # whether successful pick and place ?
        if err0['dist'] < frame['pick_radius']:
            success_pick = 1
        else:
            success_pick = 0
        
        if err1['dist'] < frame['place_radius']:
            success_place = 1
        else:
            success_place = 0
        
        if err0['dist'] < frame['pick_radius'] and err1['dist'] < frame['place_radius']:
            success = 1
        else:
            success = 0
                
        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
            success=success,
            success_pick=success_pick,
            success_place=success_place
        )

    def test_epoch_end(self, all_outputs):

        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])
        success_rate = np.sum([v['success'] for v in all_outputs]) / len(all_outputs)
        success_pick_rate = np.sum([v['success_pick'] for v in all_outputs]) / len(all_outputs)
        success_place_rate = np.sum([v['success_place'] for v in all_outputs]) / len(all_outputs)

        file_name = os.path.join(self.cfg['train']['train_dir'],'checkpoints', 'pick_n_place_loss.txt') 
        saved_file = open(file_name, 'a')
        print('=============================', file=saved_file)
        print('vl/attn/loss', mean_val_loss0, file=saved_file)
        print('vl/trans/loss', mean_val_loss1,file=saved_file)
        print('vl/loss', mean_val_total_loss,file=saved_file)
        print('vl/total_attn_dist_err', total_attn_dist_err,file=saved_file)
        print('vl/total_attn_theta_err', total_attn_theta_err,file=saved_file)
        print('vl/total_trans_dist_err', total_trans_dist_err,file=saved_file)
        print('vl/total_trans_theta_err', total_trans_theta_err,file=saved_file)
        print('success_rate', success_rate,file=saved_file)
        print('success_pick_rate', success_pick_rate,file=saved_file)
        print('success_place_rate', success_place_rate,file=saved_file)
        saved_file.close()

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
            success_rate=success_rate,
            success_pick_rate=success_pick_rate,
            success_place_rate=success_place_rate
        )