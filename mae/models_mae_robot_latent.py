"""
Such a big change again...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTextModel
from blocks import DecoderCABlockLang
from models_mae_robot import MAERobot
from mae.blocks import DropPath
from mae.models_mae_robot_fuse import BiAttentionBlock
from transformers import AutoImageProcessor, AutoModel
from mae.util.pos_embed import get_2d_varsize_sincos_pos_embed


class MAERobotLangFuseDino(nn.Module):
    """ Use fusion module but no siamese encoder, just one encoder
    """

    def __init__(self, img_size=(320, 160), patch_size=14, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False,
                 text_model="openai/clip-vit-base-patch32"):
        super().__init__()

        # parameters
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained(text_model)
        self.clip_text.requires_grad_(False)
        print(f"Loaded CLIP text model: {text_model}")

        # The DINO model
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        self.dino.requires_grad_(False)

        self.decoder_norm = norm_layer(embed_dim)
        self.fuse_blocks = nn.ModuleList([
            BiAttentionBlock(v_dim = embed_dim, 
                             l_dim = decoder_embed_dim, 
                             embed_dim = embed_dim, 
                             num_heads = num_heads)
            for _ in range(depth)
        ])

        # Predictor
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        
        # Encode images into latent space
        latent1 = self.dino(img1,return_dict=False)[0]
        #latent2 = self.dino(img2,retrun_dict=False)

        # encode the language goal
        lang_emb = self.get_lang_embed(lang)[0]

        # prediction
        pred = self.decoder(latent1, lang_emb)
        loss = self.forward_loss(img2, pred)
        return loss, pred, None
    
    def decoder(self, x, lang_emb):
        """
        fusion decoder + predictor
        """
        # fuse the two modalities
        for fuse_block in self.fuse_blocks:
            x, lang_emb = fuse_block(x, lang_emb, attention_mask_v=None, attention_mask_l=None)
        
        out = self.decoder_norm(x)
        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch

        return loss
    
    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x
    
    def unpatchify(self, x):
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    


class MAERobotLangFuseDinoDecoder(MAERobotLangFuseDino):
    def __init__(self, img_size=(320, 160), patch_size=14, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False,
                 text_model="openai/clip-vit-base-patch32"):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                         mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_im2_in_dec=norm_im2_in_dec, norm_pix_loss=norm_pix_loss,
                         text_model=text_model)
        self.img_size = img_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 256 + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # Positional encoding
        decoder_pos_embed = get_2d_varsize_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                            int(self.img_size[0] // patch_size), int(self.img_size[1] // patch_size),
                                                            cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def decoder(self, x, lang_emb):
        """
        fusion decoder + predictor
        """
        # fuse the two modalities
        for fuse_block in self.fuse_blocks:
            x, lang_emb = fuse_block(x, lang_emb, attention_mask_v=None, attention_mask_l=None)
        
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        out1 = x
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, None)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out