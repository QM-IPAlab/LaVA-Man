"""
DERT style decoder for the MAE robot language.
"""
import random
import torch
import torch.nn as nn

from transformers import CLIPTextModel
from blocks import DecoderDERTBlockLang
from models_mae_robot import MAERobot
import transformers

class MAERobotLangDertDecode(MAERobot):
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False,
                 text_model="openai/clip-vit-base-patch32"):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderDERTBlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip_text = CLIPTextModel.from_pretrained(text_model)
        self.clip_text.requires_grad_(False)
        print(f"Loaded CLIP text model: {text_model}")

        #FIXME: Now only support 224,224, not 320,160
        self.reconstruction_embedding = nn.Embedding(
            (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size), 
            decoder_embed_dim)
        print(self.reconstruction_embedding.weight.shape)

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
        Dert style deocder
        """

        # boradcast the queries
        reconstruction_queries = self.reconstruction_embedding.weight[None,:,:].repeat(latent1.shape[0], 1, 1)

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
        query = reconstruction_queries
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            query = blk(query, out1, out2, lang_emb)
        out = self.decoder_norm(query)

        out = self.decoder_pred(out)
        #out = out[:, 1:, :]

        return out