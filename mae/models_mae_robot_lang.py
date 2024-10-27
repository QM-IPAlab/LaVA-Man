import random
import torch
import torch.nn as nn

from transformers import CLIPTextModel
from blocks import DecoderCABlockLang, DecoderCABlockLangNoRef, EncoderCABlockLang, DecoderCABlockLangReverse
from models_mae_robot import MAERobot
import transformers

transformers.logging.set_verbosity_error()


class MAERobotLang(MAERobot):
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

        # modify name
        # self.decoder_pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
        #                                        requires_grad=False)
        # self.decoder_pos_embed_2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
        #                                        requires_grad=False)

    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2


class MAERobotLangNoRef(MAERobot):
    """No Siamese encoder. Only one encoder for the o_t image
    and train to predict the o_t+1 
    Modified @ 30-Aug-2024
    """
    
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLangNoRef(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.requires_grad_(False)


    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        """Mask the input, disable the reference"""
        
        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=mask_ratio)
        #latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio=0.0)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent1, ids_restore1, lang_emb)
        loss = self.forward_loss(img2, pred, mask1)

        return loss, pred, mask1

    def forward_ca_decoder(self, x, ids_restore, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        # encoder to decoder layer
        x = self.decoder_embed(x)

        # append masked tokens to the sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add positional embedding
        x = x + self.decoder_pos_embed

        out1 = x
        out2 = None
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    

class MAERobotLang2(MAERobot):
    """Also encode the language in the encoder"""

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.blocks = nn.ModuleList([
            EncoderCABlockLang(
                512, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.requires_grad_(False)

        # modify name
        # self.decoder_pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
        #                                        requires_grad=False)
        # self.decoder_pos_embed_2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
        #                                        requires_grad=False)

    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb[0]

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        
        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_ca_encoder(img1, lang_emb, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_ca_encoder(img2, lang_emb, mask_ratio)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2

    def forward_ca_encoder(self, x, lang, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, lang)
        x = self.norm(x)

        return x, mask, ids_restore
    
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


class MAERobotLangRecon(MAERobotLang):
    """
    For ablation study:
    Reconstruction version of MAERobotLang (single frame input)
    """

    def forward_ca_decoder(self, masked_latent1, ids_restore1, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        # encoder to decoder layer
        fea1 = self.decoder_embed(masked_latent1)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token.repeat(fea1.shape[0],
                                               ids_restore1.shape[1] + 1 - fea1.shape[1], 1)
        fea1_ = torch.cat([fea1[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea1_ = torch.gather(fea1_, dim=1,
                             index=ids_restore1.unsqueeze(-1).repeat(1, 1, fea1.shape[2]))  # unshuffle
        fea1 = torch.cat([fea1[:, :1, :], fea1_], dim=1)  # append cls token

        # add positional embedding
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed

        out1 = fea1
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, None, lang_emb)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    
    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=mask_ratio)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent1, ids_restore1, lang_emb)
        loss = self.forward_loss(img1, pred, mask1)

        return loss, pred, mask1


class MAERobotLangDualMasking(MAERobotLang):
    """
    For ablation study:
    Dual mask. (Mask both two inputs)
    """
    
    def forward_ca_decoder(self, latent1, masked_latent2, ids_restore1, ids_restore2, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        # encoder to decoder layer
        fea1 = self.decoder_embed(latent1)
        fea2 = self.decoder_embed(masked_latent2)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token.repeat(fea1.shape[0],
                                               ids_restore1.shape[1] + 1 - fea1.shape[1], 1)
        fea1_ = torch.cat([fea1[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea1_ = torch.gather(fea1_, dim=1,
                             index=ids_restore1.unsqueeze(-1).repeat(1, 1, fea1.shape[2]))  # unshuffle
        fea1 = torch.cat([fea1[:, :1, :], fea1_], dim=1)  # append cls token

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

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.75)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio=mask_ratio)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore1, ids_restore2,lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2
    
    
class MAERobotLangReverse(MAERobot):
    """
    Reverse the order of the two images when inputting to the decoder
    The maksed image -> self_attention (qv)
    the unmasked not -> cross_attention (key)
    """
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLangReverse(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.requires_grad_(False)

        # modify name
        # self.decoder_pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
        #                                        requires_grad=False)
        # self.decoder_pos_embed_2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
        #                                        requires_grad=False)

    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2

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
            out2, out1 = blk(out2, out1, lang_emb)
        out = self.decoder_norm(out2)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out


class MAERobotLangCF(MAERobot):
    """
    Condition free trianing of MAE with language
    """

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

        self.mask_token_lang = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_recon = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_mae = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

    def copy_mask_tokens(self):
        """Deep copy the mask tokens for condition free training
        """
        self.mask_token_lang.data = self.mask_token.data.clone()
        self.mask_token_recon.data = self.mask_token.data.clone()
        self.mask_token_mae.data = self.mask_token.data.clone()
        print("Mask tokens deepy copied")

    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        
        # Reconstruction training
        if random.random() < 0.5 and self.training: 
            input_mask_ratio = 0.75        
            latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=input_mask_ratio)
            pred = self.forward_recon_decoder(latent1, ids_restore1)
            loss = self.forward_loss(img1, pred, mask1)
            mask2 = mask1

        # Prediciton training
        else: 
            input_mask_ratio = 0.0
            latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=input_mask_ratio)
            latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)

            lang_emb = self.get_lang_embed(lang)

            pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
            loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2

    def forward_recon_decoder(self, masked_latent, ids_restore):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        # encoder to decoder layer
        fea = self.decoder_embed(masked_latent)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token_mae.repeat(fea.shape[0],
                                               ids_restore.shape[1] + 1 - fea.shape[1], 1)
        fea_ = torch.cat([fea[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea_ = torch.gather(fea_, dim=1,
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, fea.shape[2]))  # unshuffle
        fea = torch.cat([fea[:, :1, :], fea_], dim=1)  # append cls token

        # add positional embedding
        fea = fea + self.decoder_pos_embed
        
        # reconstruction tokens
        fea2 = self.mask_token_recon.repeat(fea.shape[0], fea.shape[1], 1)
        fea2 = fea2 + self.decoder_pos_embed

        # lang_emb tokens
        lang_emb = self.mask_token_lang.repeat(fea.shape[0], 77, 1)

        out1 = fea
        out2 = fea2
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    
    def cliport_forward(self):
        
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=input_mask_ratio)