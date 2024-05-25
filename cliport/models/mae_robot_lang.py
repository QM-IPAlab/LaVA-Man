from mae import models_lib
from mae.util import misc

import torch.nn.functional as F
import torch.nn as nn
import torch
from cliport.models.core.unet import Cat
from visualizer import get_local
from einops import rearrange
from cliport.models.resnet import IdentityBlock, ConvBlock

class MAEModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, model_name='mae_robot_lang',
                 pretrain_path='/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang/checkpoint-399.pth'):
        super(MAEModel, self).__init__()
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

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
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False

        self.layer1 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(64, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
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
        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        predict = self.conv(out)
        return predict


class MAESeg2Model(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, model_name='mae_robot_lang',
                 pretrain_path='/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang/checkpoint-399.pth'):
        super(MAESeg2Model, self).__init__()
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False

        self.layer1 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.cat1 = Cat(256, 256)

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.cat2 = Cat(128, 128)
        self.layer3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.cat3 = Cat(64, 64)
        self.layer4 = nn.Sequential(
            ConvBlock(64, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.cat4 = Cat(16, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
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
        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify(out)

        out = self.layer1(out)
        out = self.cat1(out, rgb)
        out = self.layer2(out)
        out = self.cat2(out, rgb)
        out = self.layer3(out)
        out = self.cat3(out, rgb)
        out = self.layer4(out)
        out = self.cat4(out, rgb)

        predict = self.conv(out)
        return predict


class MAEFeatUpModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, model_name='mae_robot_lang',
                 pretrain_path='/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang/checkpoint-399.pth'):
        super(MAESegModel, self).__init__()
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False

        self.layer1 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(64, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
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
        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        predict = self.conv(out)
        return predict