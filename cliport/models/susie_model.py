import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Cat, CatPredic, CatPredicVision
from cliport.models.mae_robot_lang import MAESeg2ModelAdd


class SusieSeg2ModelAdd(nn.Module):

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name,
                 pretrain_path):
        super(SusieSeg2ModelAdd, self).__init__()
       
        self.output_dim = output_dim
        self.batchnorm = cfg['train']['batchnorm']
      
        print(f"batchnorm: {self.batchnorm}")
        self.layer1 = nn.Sequential(
            ConvBlock(3, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.cat1 = Cat(256, 256)

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.cat2 = Cat(128, 128)
        self.layer3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.cat3 = Cat(64, 64)
        self.layer4 = nn.Sequential(
            ConvBlock(64, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )
        self.cat4 = Cat(16, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )
           
    
    def preprocess(self, img):
        """Pre-process input (subtract mean, divide by std)."""

        clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_color_std = [0.26862954, 0.26130258, 0.27577711]

        color_mean = clip_color_mean
        color_std = clip_color_std

        # convert to pytorch tensor (if required)
        def cast_shape(stat, img):
            tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
            tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
            return tensor

        color_mean = cast_shape(color_mean, img)
        color_std = cast_shape(color_std, img)

        # normalize
        img = img.clone()
        img[:, :3, :, :] = ((img[:, :3, :, :] / 255 - color_mean) / color_std)
        img[:, 3:, :, :] = ((img[:, 3:, :, :] / 255 - color_mean) / color_std)

        return img

    def forward(self, x, lang):
        x = self.preprocess(x)

        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        recon = x[:, 3:]  # select recon
        
        out = self.layer1(rgb)
        out = self.cat1(out, recon)
        out = self.layer2(out)
        out = self.cat2(out, recon)
        out = self.layer3(out)
        out = self.cat3(out, recon)
        out = self.layer4(out)
        out = self.cat4(out, recon)

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict