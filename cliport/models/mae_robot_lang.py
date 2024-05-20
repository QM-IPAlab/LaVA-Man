from mae import models_lib
from mae.util import misc

import torch.nn.functional as F
import torch.nn as nn
import torch

from visualizer import get_local
from einops import rearrange


class MAEModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, model_name='mae_robot_lang',
                 pretrain_path='/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang/checkpoint-399.pth'):
        super(MAEModel, self).__init__()
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        # load pretrain model
        #if pretrain_path is not None:
        #    misc.dynamic_load_pretrain(self.model, pretrain_path)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.decoder_affordance = nn.Linear(512, 16 ** 2 * output_dim, bias=True)

    def unpatchify(self, x):
        p = self.model.patch_embed.patch_size[0]
        h = self.model.img_size[0] // p
        w = self.model.img_size[1] // p
        assert h * w == x.shape[1]
        ch = self.output_dim
        x = x.reshape(shape=(x.shape[0], h, w, p, p, ch))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], ch, h * p, w * p))
        return imgs

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
        lang_emb = self.model.get_lang_embed(lang)

        fea = self.model.decoder_embed(latent)
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0], 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)

        out = self.model.decoder_norm(out1)
        out = self.decoder_affordance(out)
        out = out[:, 1:, :]

        predict = self.unpatchify(out)
        assert predict.shape[-2:] == in_shape[-2:]
        # x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return predict


class MAESegModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, model_name='mae_robot_lang',
                 pretrain_path='/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang/checkpoint-399.pth'):
        super(MAESegModel, self).__init__()
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        # load pretrain model
        if pretrain_path is not None:
            misc.dynamic_load_pretrain(self.model, pretrain_path)

        self.preprocess = preprocess
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 20x20 -> 40x40
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 40x40 -> 80x80
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, self.output_dim, kernel_size=4, stride=2, padding=1)
        )

    def unpatchify(self, x):
        p = self.model.patch_embed.patch_size[0]
        h = self.model.img_size[0] // p
        w = self.model.img_size[1] // p
        x = rearrange(x, 'b (nh nw) c -> b c nh nw', nh=h, nw=w)
        return x

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
        lang_emb = self.model.get_lang_embed(lang)

        fea = self.model.decoder_embed(latent)
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0], 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)

        out = self.model.decoder_norm(out1)
        out = out[:, 1:, :]  # 1, 400 512
        out = self.unpatchify(out)
        predict = self.head(out)
        return predict
