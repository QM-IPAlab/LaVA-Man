"""
Cross view completion, extended from MAERobot fuse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from transformers import CLIPTextModel
from blocks import DecoderCABlockLang
from models_mae_robot import MAERobot
from mae.blocks import DropPath
import transformers

from mae.models_mae_robot_fuse import BiAttentionBlock


class MAERobotLangFuseCV(MAERobot):
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False,
                 text_model="openai/clip-vit-base-patch32"):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained(text_model)
        self.clip_text.requires_grad_(False)
        print(f"Loaded CLIP text model: {text_model}")

        self.fuse_blocks = nn.ModuleList([
            BiAttentionBlock(embed_dim, decoder_embed_dim, embed_dim, num_heads)
            for _ in range(depth)
        ])

        # the crossviews block
        self.decoder_blocks_cv = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])
        
        self.decoder_embed_cv = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token_cv = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_cv = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_norm_cv = norm_layer(decoder_embed_dim)
        self.decoder_pred_cv = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        # initialization
        self.decoder_pos_embed_cv.data.copy_(self.decoder_pos_embed)
        torch.nn.init.normal_(self.mask_token_cv, std=.02)
        self.apply(self._init_weights)

    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        """
        image 2 contais  two part: img2 and imgcv
        """
        img2, imgcv = img2

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)
        latentcv, maskcv, ids_restorecv = self.forward_encoder(imgcv, mask_ratio)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # fuse the two modalities
        lang_emb = lang_emb[0]
        for fuse_block in self.fuse_blocks:
            latent1, lang_emb = fuse_block(latent1, lang_emb, attention_mask_v=None, attention_mask_l=None)
        
        pred = self.forward_pred(latent1, latent2, ids_restore2)
        complete = self.forward_complete(latent1, latentcv, ids_restorecv)

        loss_pred = self.forward_loss(img2, pred, mask2)
        loss_complete = self.forward_loss(imgcv, complete, maskcv)

        loss = 0.7*loss_pred + 0.3*loss_complete
        
        if torch.isnan(loss).any():
            print("At least one loss is NaN")
            import pdb; pdb.set_trace()

        return loss, pred, mask2   
    
    def forward_pred(self, latent1, masked_latent2, ids_restore2):
        
        # encoder to decoder layer
        fea1 = self.decoder_embed(latent1)
        fea2 = self.decoder_embed(masked_latent2)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        # interpolate position encoding if necessary
        decoder_pos_embed = self.decoder_pos_embed
        if self.decoder_pos_embed.shape[1] != fea2.shape[1]:
            decoder_pos_embed = self.interpolate_pos_encoding(fea2, decoder_pos_embed, 320, 160)
                   
        # add positional embedding
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + decoder_pos_embed
            fea2 = fea2 + decoder_pos_embed

        out1 = fea1
        out2 = fea2
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, None)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    
    def forward_complete(self, latent1, masked_latent2, ids_restore2):
        """ cross view completion """

        # encoder to decoder layer
        fea1 = self.decoder_embed_cv(latent1)
        fea2 = self.decoder_embed_cv(masked_latent2)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token_cv.repeat(fea2.shape[0], ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1, index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        # interpolate position encoding if necessary
        decoder_pos_embed = self.decoder_pos_embed_cv
        if self.decoder_pos_embed_cv.shape[1] != fea2.shape[1]:
            decoder_pos_embed = self.interpolate_pos_encoding(fea2, decoder_pos_embed, 320, 160)
                   
        # add positional embedding
        if self.decoder_pos_embed_cv is not None:
            fea1 = fea1 + decoder_pos_embed
            fea2 = fea2 + decoder_pos_embed

        out1 = fea1
        out2 = fea2
        # apply Transformer blocks
        for blk in self.decoder_blocks_cv: out1, out2 = blk(out1, out2, None)
        out = self.decoder_norm_cv(out1)

        out = self.decoder_pred_cv(out)
        out = out[:, 1:, :]

        return out 