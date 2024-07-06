from mae import models_lib
from mae.util import misc

import torch.nn.functional as F
import torch.nn as nn
import torch
from cliport.models.core.unet import CatDepth, Cat

from visualizer import get_local
from einops import rearrange
from cliport.models.resnet import IdentityBlock, ConvBlock
from transformers import AutoTokenizer
from cliport.models.core.fusion import FusionConvLat
from visualizer import get_local

class MAESeg2DepthModel(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg2DepthModel, self).__init__()
        model_name = 'mae_robot_lang' if model_name is None else model_name
        self.model = models_lib.__dict__[model_name](
            img_size=input_shape[:2],
            norm_pix_loss=False,
            in_chans=4)

        # load pretrain model
        if pretrain_path:
            misc.dynamic_load_pretrain(self.model, pretrain_path, interpolate=True)

        self.preprocess = preprocess
        self.output_dim = output_dim
        self.batchnorm = False
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.layer1 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.cat1 = CatDepth(256, 256)

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.cat2 = CatDepth(128, 128)
        self.layer3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.cat3 = CatDepth(64, 64)
        self.layer4 = nn.Sequential(
            ConvBlock(64, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.cat4 = CatDepth(16, 16)
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
        rgb = x[:, :4]  # select RGB
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


class MAESeg2LatModel(nn.Module):
    """
    MAE model with lateral connections(depth) same as CLIPort
    """

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg2LatModel, self).__init__()
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

        self.lat_fusion1 = FusionConvLat(input_dim=256+512, output_dim=256)
        self.lat_fusion2 = FusionConvLat(input_dim=128+256, output_dim=128)
        self.lat_fusion3 = FusionConvLat(input_dim=64+128, output_dim=64)
        self.lat_fusion4 = FusionConvLat(input_dim=16+64, output_dim=16)

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
    def forward(self, x, lat, lang):
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
        out = out[:, 1:, :]  # 1, 200, 512
        out = self.unpatchify(out) # 1, 20, 10, 512
        out = self.layer1(out)
        out = self.cat1(out, rgb)
        out = self.lat_fusion1(out, lat[-5])

        out = self.layer2(out)
        out = self.cat2(out, rgb)
        out = self.lat_fusion2(out, lat[-4])

        out = self.layer3(out)
        out = self.cat3(out, rgb)
        out = self.lat_fusion3(out, lat[-3])

        out = self.layer4(out)
        out = self.cat4(out, rgb)
        out = self.lat_fusion4(out, lat[-2])

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict


class MAESeg2LatModelPlus(nn.Module):
    """
    MAE model with lateral connections(depth) same as CLIPort.
    Modify the forward function to avoid possible aliasing.
    """

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESeg2LatModelPlus, self).__init__()
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer5 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer6 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

        self.lat_fusion1 = FusionConvLat(input_dim=1024+512, output_dim=512)
        self.lat_fusion2 = FusionConvLat(input_dim=512+256, output_dim=256)
        self.lat_fusion3 = FusionConvLat(input_dim=256+128, output_dim=128)
        self.lat_fusion4 = FusionConvLat(input_dim=128+64, output_dim=64)
        self.lat_fusion5 = FusionConvLat(input_dim=64+32, output_dim=32)
        self.lat_fusion6 = FusionConvLat(input_dim=32+16, output_dim=16)

        self.cat1 = Cat(512, 512)
        self.cat2 = Cat(256, 256)
        self.cat3 = Cat(128, 128)
        self.cat4 = Cat(64, 64)
        self.cat5 = Cat(32, 32)
        self.cat6 = Cat(16, 16)


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
    def forward(self, x, lat, lang):
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
        out = out[:, 1:, :]  # 1, 200, 512
        out = self.unpatchify(out) # 1, 512, 20, 10
        
        out = self.conv1(out) # 1, 512, 20, 10
        out = self.cat1(out, rgb) # 1, 512, 20, 10
        out = self.lat_fusion1(out, lat[-6]) # 1, 512, 20, 10

        out = self.layer2(out) # 1, 256, 40, 20
        out = self.cat2(out, rgb)
        out = self.lat_fusion2(out, lat[-5]) # 1, 256, 40, 20

        out = self.layer3(out) # 1, 128, 80, 40
        out = self.cat3(out, rgb)
        out = self.lat_fusion3(out, lat[-4]) # 1, 128, 80, 40

        out = self.layer4(out) # 1, 64, 160, 80
        out = self.cat4(out, rgb)
        out = self.lat_fusion4(out, lat[-3]) # 1, 64, 160, 80

        out = self.layer5(out) # 1, 32, 320, 160
        out = self.cat5(out, rgb)
        out = self.lat_fusion5(out, lat[-2]) # 1, 32, 320, 160
 
        out = self.layer6(out) # 1, 16, 640, 320
        out = self.cat6(out, rgb)
        out = self.lat_fusion6(out, lat[-1]) # 1, 16, 640, 320
    
        predict = self.conv(out)
        predict = F.interpolate(predict, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        return predict