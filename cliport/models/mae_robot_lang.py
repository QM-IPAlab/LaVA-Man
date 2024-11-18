from mae import models_lib
from mae.util import misc

import torch.nn.functional as F
import torch.nn as nn
import torch
from cliport.models.core.unet import Cat, CatPredic, CatPredicVision
from visualizer import get_local
from einops import rearrange
from cliport.models.resnet import IdentityBlock, ConvBlock
from transformers import AutoTokenizer
from cliport.models.featup_pretrained import FeatUp
from cliport.models.dpt_head import PixelwiseTaskWithDPT, PixelwiseTaskWithDPTOurs
CACHE_PATH = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/cache/hf-cache"
from cliport.models.core.clip import build_model, load_clip, tokenize
from torchvision import transforms
from cliport.models.core.fusion import FusionMultOurs
from cliport.models.core.unet import Up
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

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg2Model, self).__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        text_model=cfg['text_model'] if 'text_model' in cfg else 'openai/clip-vit-base-patch32'
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False,
            text_model=text_model)
        
        # linear probe
        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)
            print("Linear probing")

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = cfg['train']['batchnorm']
        text_model = cfg['text_model'] if 'text_model' in cfg else 'openai/clip-vit-base-patch32'
        self.text_processor = AutoTokenizer.from_pretrained(text_model)
        print(f"batchnorm: {self.batchnorm}")
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

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        elif type(lang) is list: # if batch size
            if type(lang[0]) is str:
                decoded_strings = [s for s in lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB, b, c, h, w
        latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
        lang_emb = self.get_lang_embed(lang, device)

        fea = self.model.decoder_embed(latent)
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None
        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)
        out = out[:, 1:, :]  # 1, 200, 512
        out = self.unpatchify(out) # 1, 512, 20, 10

        out = self.layer1(out)    # 1, 256, 40, 20
        out = self.cat1(out, rgb) # 1, 256, 40, 20
        out = self.layer2(out)    # 1, 128, 80, 40
        out = self.cat2(out, rgb) # 1, 128, 80, 40
        out = self.layer3(out)    # 1, 64, 160, 80
        out = self.cat3(out, rgb) # 1, 64, 160, 80
        out = self.layer4(out)    # 1, 16, 320, 160
        out = self.cat4(out, rgb) # 1, 16, 320, 160

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out) # 1, 1, 320, 160
        return predict


class MAESeg2ModelFullMask(MAESeg2Model):
    """MAESeg2 model, but feed the 100 percent masked image to the decoder"""
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent1, mask1, ids_restore1 = self.model.forward_encoder(rgb, mask_ratio=0)
        latent2, mask2, ids_restore2 = self.model.forward_encoder(rgb, mask_ratio=1.0)
        lang_emb = self.get_lang_embed(lang,device)

        fea1 = self.model.decoder_embed(latent1)
        fea2 = self.model.decoder_embed(latent2)
        
        masked_tokens = self.model.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        fea1 = fea1 + self.model.decoder_pos_embed
        fea2 = fea2 + self.model.decoder_pos_embed

        out1 = fea1
        out2 = fea2

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)
        
        # from torchvision.utils import save_image
        # #(C, H, W)
        # import os;
        # os.makedirs('recons', exist_ok=True)
        # recon = self.model.decoder_pred(out)
        # recon = recon[:, 1:, :]  # 1, 200, 768
        # recon = self.model.unpatchify(recon)
        # recon = recon[0]
        # recon = (recon- recon.min()) / (recon.max() - recon.min())
        # i = len(os.listdir('recons'))
        # save_image(recon, f'recons/recon_pick{i}.png')       

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

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict


class MAEFeatUpModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, model_name='mae_robot_lang',
                 pretrain_path='/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang/checkpoint-399.pth'):
        super(MAEFeatUpModel, self).__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.upsampler = FeatUp()

        self.layer1 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(64, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
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

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
        lang_emb = self.get_lang_embed(lang, device)

        fea = self.model.decoder_embed(latent)
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0], 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)

        out = self.model.decoder_norm(out1)
        out = out[:, 1:, :]  # 1, 200, 512
        out = self.unpatchify(out)
        out = self.upsampler(out, rgb)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv(out)

        return out


class MAESeg3Model(nn.Module):
    """Use the output of mae after the prediction head"""

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg3Model, self).__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)
        
        # linear probe
        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.layer1 = nn.Sequential(
            ConvBlock(768, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
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

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    @get_local('predict', 'x')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
        lang_emb = self.get_lang_embed(lang,device)

        fea = self.model.decoder_embed(latent)
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0], 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)
        out = self.model.decoder_pred(out)
        out = out[:, 1:, :]  # 1, 200, 768
        out = self.unpatchify(out) # 1, 768, 20, 10

        out = self.layer1(out)    # 1, 256, 40, 20
        out = self.cat1(out, rgb) # 1, 256, 40, 20
        out = self.layer2(out)    # 1, 128, 80, 40
        out = self.cat2(out, rgb) # 1, 128, 80, 40
        out = self.layer3(out)    # 1, 64, 160, 80
        out = self.cat3(out, rgb) # 1, 64, 160, 80
        out = self.layer4(out)    # 1, 16, 320, 160
        out = self.cat4(out, rgb) # 1, 16, 320, 160

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict


class MAESegBaseModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super().__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)

        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)

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

        if 'voltron' in model_name: 
            self.text_processor = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_PATH)
            self.layer1 = nn.Sequential(
                ConvBlock(384, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
                IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
                nn.UpsamplingBilinear2d(scale_factor=2),
            )
        else:
            self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    def unpatchify(self, x):
        p = self.model.patch_embed.patch_size[0]
        h = self.model.img_size[0] // p
        w = self.model.img_size[1] // p
        x = rearrange(x, 'b (nh nw) c -> b c nh nw', nh=h, nw=w)
        return x

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip_text(**processed_lang, return_dict=False)
        return lang_emb
    
    def get_lang_processed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        elif type(lang) is list: # if batch size
            if type(lang[0]) is str:
                decoded_strings = [s for s in lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        return processed_lang

    @get_local('predict', 'rgb', 'relevance')
    def forward(self, x, lang):
        img_processed = self.preprocess(x, dist='clip')
        lang_processed = self.get_lang_processed(lang, img_processed.device) 

        in_shape = img_processed.shape
        rgb = img_processed[:, :3]  # select RGB

        mae_output = self.model.cliport_forward(rgb, lang_processed)
        #relevance = self.model.show_relevance_map(rgb, lang_processed)
        mae_feamap = self.unpatchify(mae_output)

        out = self.layer1(mae_feamap)
        out = self.cat1(out, rgb)
        out = self.layer2(out)
        out = self.cat2(out, rgb)
        out = self.layer3(out)
        out = self.cat3(out, rgb)
        out = self.layer4(out)
        out = self.cat4(out, rgb)

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict


class MAESegCLIPModel(MAESeg2Model):
    
    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESegCLIPModel, self).__init__(input_shape, output_dim, cfg, 
                 device, preprocess, model_name,
                 pretrain_path)
        
        self.cat1 = CatPredic(256, 256)
        self.cat2 = CatPredic(128, 128)
        self.cat3 = CatPredic(64, 64)
        self.cat4 = CatPredic(16, 16)

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        elif type(lang) is list: # if batch size
            if type(lang[0]) is str:
                decoded_strings = [s for s in lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip.text_model(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent1, mask1, ids_restore1 = self.model.clip.vision_model(mask_ratio=0.0, pixel_values=rgb,  output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True)
        latent2, mask2, ids_restore2 = self.model.clip.vision_model(mask_ratio=1.0, pixel_values=rgb,  output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True)
        lang_emb = self.get_lang_embed(lang,device)

        fea1 = self.model.decoder_embed(latent1)
        fea2 = self.model.decoder_embed(latent2)
        
        masked_tokens = self.model.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        fea1 = fea1 + self.model.decoder_pos_embed
        fea2 = fea2 + self.model.decoder_pos_embed

        out1 = fea1
        out2 = fea2

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)

        recon = self.model.decoder_pred(out)
        recon = recon[:, 1:, :]  # 1, 200, 768
        recon = self.model.unpatchify(recon)
        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify(out)

        out = self.layer1(out)
        out = self.cat1(out, rgb, recon)
        out = self.layer2(out)
        out = self.cat2(out, rgb, recon)
        out = self.layer3(out)
        out = self.cat3(out, rgb, recon)
        out = self.layer4(out)
        out = self.cat4(out, rgb, recon)

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict
    

class MAESegDPTModel(nn.Module):
    """
    Use DPT model as the segmentation head as Croco_v2
    """

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESegDPTModel, self).__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)
        
        # linear probe
        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        #FIXME: hard code the head now
        self.head = PixelwiseTaskWithDPT(
            output_width_ratio=0.5,
            num_channels=self.output_dim
        )

        self.head.setup()

    def unpatchify(self, x):
        p = self.model.patch_embed.patch_size[0]
        h = self.model.img_size[0] // p
        w = self.model.img_size[1] // p
        x = rearrange(x, 'b (nh nw) c -> b c nh nw', nh=h, nw=w)
        return x

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        elif type(lang) is list: # if batch size
            if type(lang[0]) is str:
                decoded_strings = [s for s in lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip_text(**processed_lang, return_dict=False)
        return lang_emb
    
    def forward_encoder_dpt(self, x, mask_ratio=0.0):
        # embed patches
        x = self.model.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.model.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.model.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        out_list = []
        for blk in self.model.blocks:
            x = blk(x)
            out_list.append(x)
        out_list[-1] = self.model.norm(out_list[-1])

        return out_list, mask, ids_restore
       

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')
        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.forward_encoder_dpt(rgb, mask_ratio=0.0)
        lang_emb = self.get_lang_embed(lang, device)

        fea = self.model.decoder_embed(latent[-1])
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0]//lang_emb[0].shape[0], 1, 1])

        out_list = []
        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
            out_list.append(out1)
        out_list[-1] = self.model.decoder_norm(out_list[-1])

        all_feats = latent + out_list
        all_feats = [x[:,1:,:] for x in all_feats]

        img_info = {'height': in_shape[2], 'width': in_shape[3]}
        predict = self.head(all_feats, img_info)

        return predict


class MAESegDPT2LossModel(nn.Module):
    """
    Use DPT model as the segmentation head as Croco_v2,
    Add the image reconstruction loss
    """

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESegDPT2LossModel, self).__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)
        
        # linear probe
        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        #FIXME: hard code the head now
        self.head = PixelwiseTaskWithDPT(
            output_width_ratio=0.5,
            num_channels=self.output_dim
        )

        self.head.setup()

    def unpatchify(self, x):
        p = self.model.patch_embed.patch_size[0]
        h = self.model.img_size[0] // p
        w = self.model.img_size[1] // p
        x = rearrange(x, 'b (nh nw) c -> b c nh nw', nh=h, nw=w)
        return x

    def get_lang_embed(self, lang, device):
        if type(lang) is str:
            decoded_strings = [lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to(device)
        lang_emb = self.model.clip_text(**processed_lang, return_dict=False)
        return lang_emb
    
    def forward_encoder_dpt(self, x, mask_ratio=0.0):
        # embed patches
        x = self.model.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.model.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.model.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        out_list = []
        for blk in self.model.blocks:
            x = blk(x)
            out_list.append(x)
        out_list[-1] = self.model.norm(out_list[-1])

        return out_list, mask, ids_restore
       
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.model.patchify(imgs)
        if self.model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        return loss

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')
        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.forward_encoder_dpt(rgb, mask_ratio=0.0)
        lang_emb = self.get_lang_embed(lang, device)

        fea = self.model.decoder_embed(latent[-1])
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0], 1, 1])

        out_list = []
        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
            out_list.append(out1)
        out_list[-1] = self.model.decoder_norm(out_list[-1])

        rgb_predict = self.model.decoder_pred(out_list[-1])
        rgb_predict = rgb_predict[:, 1:, :]
        rgb_loss = self.forward_loss(rgb, rgb_predict)

        all_feats = latent + out_list
        all_feats = [x[:,1:,:] for x in all_feats]

        img_info = {'height': in_shape[2], 'width': in_shape[3]}
        predict = self.head(all_feats, img_info)
        
        return {'out': predict, 'rgb_loss': rgb_loss}
    
    
class MAESegDPTSKModel(MAESegDPTModel):
    """
    Use DPT model as the segmentation head as Croco_v2
    and then use skip connection to combine input image
    """

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super().__init__(input_shape, output_dim, cfg, device, preprocess, model_name, pretrain_path)
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False)
        
        # linear probe
        self.linear_probe = False if 'linear_probe' not in cfg['train'] else cfg['train']['linear_probe']
        if self.linear_probe:
            self.model.requires_grad_(False)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        #FIXME: hard code the head now
        self.head = PixelwiseTaskWithDPTOurs(
            output_width_ratio=0.5,
            num_channels=self.output_dim
        )

        self.head.setup()

    @get_local('predict', 'rgb')
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')
        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent, mask, ids_restore = self.forward_encoder_dpt(rgb, mask_ratio=0.0)
        lang_emb = self.get_lang_embed(lang, device)

        fea = self.model.decoder_embed(latent[-1])
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = None
        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([out1.shape[0]//lang_emb[0].shape[0], 1, 1])

        out_list = []
        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
            out_list.append(out1)
        out_list[-1] = self.model.decoder_norm(out_list[-1])

        all_feats = latent + out_list
        all_feats = [x[:,1:,:] for x in all_feats]

        img_info = {'height': in_shape[2], 'width': in_shape[3]}
        predict = self.head(all_feats, img_info, rgb)

        return predict


class MAESeg2ModelDual(MAESeg2Model):
    """MAESeg2Model, but do not drop one cross-attention.
    Instead, copy the input as the masked image"""

    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB, b, c, h, w
        latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
        lang_emb = self.get_lang_embed(lang, device)

        fea = self.model.decoder_embed(latent)
        fea = fea + self.model.decoder_pos_embed

        out1 = fea
        out2 = fea
        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)
        out = out[:, 1:, :]  # 1, 200, 512
        out = self.unpatchify(out) # 1, 512, 20, 10

        out = self.layer1(out)    # 1, 256, 40, 20
        out = self.cat1(out, rgb) # 1, 256, 40, 20
        out = self.layer2(out)    # 1, 128, 80, 40
        out = self.cat2(out, rgb) # 1, 128, 80, 40
        out = self.layer3(out)    # 1, 64, 160, 80
        out = self.cat3(out, rgb) # 1, 64, 160, 80
        out = self.layer4(out)    # 1, 16, 320, 160
        out = self.cat4(out, rgb) # 1, 16, 320, 160

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out) # 1, 1, 320, 160
        return predict
    

class MAESeg2ModelFozenE(MAESeg2Model):
   
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB, b, c, h, w

        with torch.no_grad():
            latent, mask, ids_restore = self.model.forward_encoder(rgb, mask_ratio=0)
            lang_emb = self.get_lang_embed(lang, device)

            fea = self.model.decoder_embed(latent)
            fea = fea + self.model.decoder_pos_embed

            out1 = fea
            out2 = None

            if out1.shape[0] != lang_emb[0].shape[0]:
                lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)
        out = out[:, 1:, :]  # 1, 200, 512
        out = self.unpatchify(out) # 1, 512, 20, 10

        out = self.layer1(out)    # 1, 256, 40, 20
        out = self.cat1(out, rgb) # 1, 256, 40, 20
        out = self.layer2(out)    # 1, 128, 80, 40
        out = self.cat2(out, rgb) # 1, 128, 80, 40
        out = self.layer3(out)    # 1, 64, 160, 80
        out = self.cat3(out, rgb) # 1, 64, 160, 80
        out = self.layer4(out)    # 1, 16, 320, 160
        out = self.cat4(out, rgb) # 1, 16, 320, 160

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out) # 1, 1, 320, 160
        return predict


class MAESeg2ModelRecon(MAESeg2Model):
    """MAESeg2 model, but feed the 100 percent masked image to the decoder"""     
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent1, mask1, ids_restore1 = self.model.forward_encoder(rgb, mask_ratio=0)
        latent2, mask2, ids_restore2 = self.model.forward_encoder(rgb, mask_ratio=1.0)
        lang_emb = self.get_lang_embed(lang,device)

        fea1 = self.model.decoder_embed(latent1)
        fea2 = self.model.decoder_embed(latent2)
        
        masked_tokens = self.model.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        fea1 = fea1 + self.model.decoder_pos_embed
        fea2 = fea2 + self.model.decoder_pos_embed

        out1 = fea1
        out2 = fea2

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)
        
        recon = self.model.decoder_pred(out)
        recon = recon[:, 1:, :]  # 1, 200, 768
        recon_loss = self.recon_loss(rgb, recon)

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

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return (predict, recon_loss)

    def recon_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.model.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean() # mean loss on removed patches
        
        return loss


class MAESeg2ModelAdd(MAESeg2Model):
    """MAESeg2 model, add the predicted image to the feature map"""
    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg2ModelAdd, self).__init__(input_shape, output_dim, cfg, 
                 device, preprocess, model_name,
                 pretrain_path)
        
        self.cat1 = CatPredic(256, 256)
        self.cat2 = CatPredic(128, 128)
        self.cat3 = CatPredic(64, 64)
        self.cat4 = CatPredic(16, 16)
    
    
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent1, mask1, ids_restore1 = self.model.forward_encoder(rgb, mask_ratio=0)
        latent2, mask2, ids_restore2 = self.model.forward_encoder(rgb, mask_ratio=1.0)
        lang_emb = self.get_lang_embed(lang,device)

        fea1 = self.model.decoder_embed(latent1)
        fea2 = self.model.decoder_embed(latent2)
        
        masked_tokens = self.model.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        fea1 = fea1 + self.model.decoder_pos_embed
        fea2 = fea2 + self.model.decoder_pos_embed

        out1 = fea1
        out2 = fea2

        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)

        recon = self.model.decoder_pred(out)
        recon = recon[:, 1:, :]  # 1, 200, 768
        recon = self.model.unpatchify(recon)

        # from torchvision.utils import save_image
        # import os
        # #(C, H, W)
        # recon_save = recon[0]
        # recon_save = (recon_save- recon.min()) / (recon_save.max() - recon_save.min())
        # folder = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/recons_new"
        # i = len(os.listdir(folder))
        # save_image(recon_save, f'{folder}/recon{i}.png')


        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify(out)

        out = self.layer1(out)
        out = self.cat1(out, rgb, recon)
        out = self.layer2(out)
        out = self.cat2(out, rgb, recon)
        out = self.layer3(out)
        out = self.cat3(out, rgb, recon)
        out = self.layer4(out)
        out = self.cat4(out, rgb, recon)

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict


class MAESeg2ModelCLIPVision(MAESeg2Model):
    """MAESeg2 model, add the predicted image to the feature map"""
    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg2ModelCLIPVision, self).__init__(input_shape, output_dim, cfg, 
                 device, preprocess, model_name,
                 pretrain_path)
        
        self.cat1 = CatPredicVision(256, 256)
        self.cat2 = CatPredicVision(128, 128)
        self.cat3 = CatPredicVision(64, 64)
        self.cat4 = CatPredicVision(16, 16)

        model, _ = load_clip("RN50", device='cuda')
        self.clip = build_model(model.state_dict(), torch.float32).to('cuda')
        self.clip.requires_grad_(False)
        self.resize_transform = transforms.Resize((224, 224))

        self.lang_fuser1 = FusionMultOurs(input_dim=1024)
        self.lang_fuser2 = FusionMultOurs(input_dim=512)
        self.lang_fuser3 = FusionMultOurs(input_dim=256)

        self.lang_proj1 = nn.Linear(1024, 1024)
        self.lang_proj2 = nn.Linear(1024, 512)
        self.lang_proj3 = nn.Linear(1024, 256)

        self.up1 = Up(2048, 1024 // 2)
        self.up2 = Up(1024, 512 // 2)
        self.up3 = Up(512, 256 // 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

    def encode_image(self, img):
        img = F.pad(img, (80, 80, 0, 0), value=0)
        img = self.resize_transform(img)
        with torch.no_grad():
            img_encoding, img_im = self.clip.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, lang):
        if type(lang) is str:
            decoded_strings = [lang]
        elif type(lang) is list: # if batch size
            if type(lang[0]) is str:
                decoded_strings = [s for s in lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
        else:
            decoded_strings = [s.decode('ascii') for s in lang]
        
        processed_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
        processed_lang = processed_lang.to('cuda')
        with torch.no_grad():
            tokens = processed_lang['input_ids']
            text_feat, text_emb = self.clip.encode_text_with_embeddings(tokens)
        return text_feat, text_emb
    
    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""
        
        lang_enc, _ = self.encode_text(processed_lang) # 64, 1024; 64, 77, 512
        
        _, img_im = self.encode_image(processed_img) #64, 2048, 7, 7
    
        if lang_enc.shape[0] != img_im[0].shape[0]:
            lang_enc = lang_enc.repeat([int(img_im[0].shape[0]//lang_enc.shape[0]), 1, 1])
        
        out = self.cliport_lang_fusion(lang_enc, img_im)
        return out
    
    def cliport_lang_fusion(self, l_input, im):
        x = self.conv1(im[-1]) # 64 1024 7 7
        x = self.lang_fuser1(x, l_input, x2_mask=None, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2]) # # 64 512 14 14
        x = self.lang_fuser2(x, l_input, x2_mask=None, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3]) # #64 512 28 28
        x = self.lang_fuser3(x, l_input, x2_mask=None, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4]) #  64 512 56 56
        return x
    
    def forward(self, x, lang):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        
        ref = self.get_hidden_embeds(lang, rgb)
        ref = F.adaptive_avg_pool2d(ref, (320,320))
        ref = ref[:,:,:,80:-80]

        latent1, mask1, ids_restore1 = self.model.forward_encoder(rgb, mask_ratio=0)
        latent2, mask2, ids_restore2 = self.model.forward_encoder(rgb, mask_ratio=1.0)
        

        fea1 = self.model.decoder_embed(latent1)
        fea2 = self.model.decoder_embed(latent2)
        
        masked_tokens = self.model.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        fea1 = fea1 + self.model.decoder_pos_embed
        fea2 = fea2 + self.model.decoder_pos_embed

        out1 = fea1
        out2 = fea2

        lang_emb = self.get_lang_embed(lang,device)
        if out1.shape[0] != lang_emb[0].shape[0]:
            lang_emb = lang_emb[0].repeat([int(out1.shape[0]//lang_emb[0].shape[0]), 1, 1])

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.model.decoder_norm(out1)

        recon = self.model.decoder_pred(out)
        recon = recon[:, 1:, :]  # 1, 200, 768
        recon = self.model.unpatchify(recon)
        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify(out)

        out = self.layer1(out)
        out = self.cat1(out, rgb, recon, ref)
        out = self.layer2(out)
        
        out = self.cat2(out, rgb, recon, ref)
        out = self.layer3(out)
        
        out = self.cat3(out, rgb, recon, ref)
        out = self.layer4(out)
        
        out = self.cat4(out, rgb, recon, ref)

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict


