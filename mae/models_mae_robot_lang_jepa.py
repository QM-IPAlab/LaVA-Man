from copy import deepcopy
import torch.nn as nn
import torch
from transformers import CLIPTextModel
from blocks import DecoderCABlockLang
from mae import blocks
from models_mae_robot import MAERobot

class JEPARobotLang(MAERobot):
    """ MAE robot lang model with I-JEPA strcuture and loss.
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False, **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.blocks2 = deepcopy(self.blocks)
        self.patch_embed2 = deepcopy(self.patch_embed)
        self.norm2 = deepcopy(self.norm)
        
        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.requires_grad_(False)
        
        # Encoder 2
        self.blocks2.requires_grad_(False)
        self.patch_embed2.requires_grad_(False)
        self.norm2.requires_grad_(False)

        # projector  (project the predictor output back to encoder dim)
        self.projector = nn.Linear(decoder_embed_dim, embed_dim)


    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_encoder2(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed2(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks2:
            x = blk(x)
        x = self.norm2(x)

        return x, mask, ids_restore

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # forward target and context images
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder2(img2, mask_ratio=0.0)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder (predictor)
        latent = self.forward_ca_decoder(latent1, lang_emb)
        latent_pred = self.projector(latent)
        latent_loss = self.forward_latent_loss(latent2, latent_pred)

        # pixel_pred = self.decoder_pred(latent)
        # pixel_pred = pixel_pred[:, 1:, :]
        # pixel_loss = self.forward_pixel_loss(img2, pixel_pred)

        return latent_loss, None, None
    
    def forward_pixel_loss(self, imgs, pred):
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
        loss = loss.mean()
        return loss
    
    def forward_ca_decoder(self, latent1, lang_emb):
        # encoder to decoder layer
        fea1 = self.decoder_embed(latent1)

        # add positional embedding
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed

        out1 = fea1
        out2 = None
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)
        return out
    
    def forward_latent_loss(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = (pred - target) ** 2
        loss =loss.mean()
        return loss

    def cliport_forward(self, rgb, processed_lang):
        
        lang_emb = self.get_lang_embed(processed_lang)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])
        
        latent, mask, ids_restore = self.forward_ca_encoder(rgb, lang_emb, mask_ratio=0)
        
        fea = self.decoder_embed(latent)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out


class JEPARobotLang2loss(MAERobot):
    """ MAE robot lang model with I-JEPA strcuture and loss.
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False, **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.blocks2 = deepcopy(self.blocks)
        self.patch_embed2 = deepcopy(self.patch_embed)
        self.norm2 = deepcopy(self.norm)
        
        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.requires_grad_(False)
        
        # Encoder 2
        self.blocks2.requires_grad_(False)
        self.patch_embed2.requires_grad_(False)
        self.norm2.requires_grad_(False)

        # projector  (project the predictor output back to encoder dim)
        self.projector = nn.Linear(decoder_embed_dim, embed_dim)


    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_encoder2(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed2(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks2:
            x = blk(x)
        x = self.norm2(x)

        return x, mask, ids_restore

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # forward target and context images
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder2(img2, mask_ratio=mask_ratio)
        latent2_full, _, _ = self.forward_encoder2(img2, mask_ratio=0.0)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder (predictor)
        latent = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        latent_pred = self.projector(latent)
        latent_loss = self.forward_latent_loss(latent2_full, latent_pred)

        pixel_pred = self.decoder_pred(latent)
        pixel_pred = pixel_pred[:, 1:, :]
        pixel_loss = self.forward_pixel_loss(img2, pixel_pred)

        loss = latent_loss + pixel_loss


        return loss, pixel_pred, mask2
    
    def forward_pixel_loss(self, imgs, pred):
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
        loss = loss.mean()
        return loss
    
    def forward_ca_decoder(self, latent1, masked_latent2, ids_restore2, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
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

        # add positional embedding
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed
            fea2 = fea2 + self.decoder_pos_embed

        out1 = fea1
        out2 = fea2
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)

        return out
    
    def forward_latent_loss(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = (pred - target) ** 2
        loss =loss.mean()
        return loss

    def cliport_forward(self, rgb, processed_lang):
        
        lang_emb = self.get_lang_embed(processed_lang)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])
        
        latent, mask, ids_restore = self.forward_ca_encoder(rgb, lang_emb, mask_ratio=0)
        
        fea = self.decoder_embed(latent)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out
    

