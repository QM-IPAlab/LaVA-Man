from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTextModel
from blocks import DecoderCABlockLang
from models_mae_robot import MAERobot
from mae.blocks import DropPath
from mae.models_mae_robot_fuse import BiAttentionBlock


class MAERobotLangFuseSingle(MAERobot):
    """ Use fusion module but no siamese encoder, just one encoder
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
        
        self.fuse_blocks = nn.ModuleList([
            BiAttentionBlock(embed_dim, decoder_embed_dim, embed_dim, num_heads)
            for _ in range(depth)
        ])


    def get_lang_embed(self, processed_lang):
        lang_emb = self.clip_text(**processed_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        #self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # encoder of the first observed image (no mask)
        latent, mask, ids_restore = self.forward_encoder(img1, mask_ratio)
    
        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent, ids_restore, lang_emb)
        loss = self.forward_loss(img2, pred, mask)

        return loss, pred, mask
    
    def forward_ca_decoder(self, x, ids_restore, lang_emb):
        """
        Dert style deocder
        """
        # fuse the two modalities
        lang_emb = lang_emb[0]
        for fuse_block in self.fuse_blocks:
            x, lang_emb = fuse_block(x, lang_emb, attention_mask_v=None, attention_mask_l=None)

        # encoder to decoder layer
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # interpolate position encoding if necessary
        decoder_pos_embed = self.decoder_pos_embed
        if self.decoder_pos_embed.shape[1] != x.shape[1]:
            decoder_pos_embed = self.interpolate_pos_encoding(x, decoder_pos_embed, 320, 160)
        x = x + decoder_pos_embed

        out1 = x
        out2 = None
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, None)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out

    def forward_loss2(self, imgs, pred, mask):
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


class MAERobotLangFuseSingleSiamese(MAERobotLangFuseSingle):
    """ Single for prediciton,
        Add one Siamese encoder for crossview reference only
    """
    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        """
        For Single Siamese model, the input should be 
        image1 + crossviw1, image2 + crossview2
        """
        if self.training:
            img1, cv1 = img1
            img2, cv2 = img2
        else: 
            cv2 = img2

        # encode the current observed image with masking
        latent_img1, mask_img1, ids_restore_img1 = self.forward_encoder(img1, mask_ratio)

        # encode the target crossview image (without masking, just for reference)
        latent_cv2, mask_cv2, ids_restore_img2 = self.forward_encoder(cv2, mask_ratio=0.0)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)[0]

        # fusion of each part
        for fuse_block in self.fuse_blocks:
            latent_img1, lang_emb = fuse_block(latent_img1, lang_emb, attention_mask_v=None, attention_mask_l=None)
    
        # decoder
        pred = self.forward_ca_decoder(latent_img1, latent_cv2, ids_restore_img1)
        loss = self.forward_loss(img2, pred, mask_img1)

        return loss, pred, mask_img1
    
    def forward_ca_decoder(self, latent1, latent2, ids_restore):

        # encoder to decoder layer
        fea1 = self.decoder_embed(latent1)
        fea2 = self.decoder_embed(latent2)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token.repeat(fea1.shape[0],
                                            ids_restore.shape[1] + 1 - fea1.shape[1], 1)
        fea1_ = torch.cat([fea1[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea1_ = torch.gather(fea1_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, fea1.shape[2]))  # unshuffle
        fea1 = torch.cat([fea1[:, :1, :], fea1_], dim=1)  # append cls token

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


class MAERobotLangFuseSingleSiamese2(MAERobotLangFuseSingle):
    """ Single for prediciton,
        Add one Siamese encoder for crossview reference only
    """

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        """
        For Single Siamese model, the input should be 
        image1 + crossviw1, image2 + crossview2
        """
        if self.training:
            img1, cv1 = img1
            img2, cv2 = img2
        else: 
            cv2 = img2

        # encode the current observed image with masking
        latent_img1, mask_img1, ids_restore_img1 = self.forward_encoder(img1, mask_ratio)

        # encode the target crossview image (without masking, just for reference)
        latent_cv2, mask_cv2, ids_restore_img2 = self.forward_encoder(cv2, mask_ratio=0.0)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)[0]
        lang_emb_cv = lang_emb

        # fusion of each part
        for fuse_block in self.fuse_blocks:
            latent_img1, lang_emb = fuse_block(latent_img1, lang_emb, attention_mask_v=None, attention_mask_l=None)
            latent_cv2, lang_emb_cv = fuse_block(latent_cv2, lang_emb_cv, attention_mask_v=None, attention_mask_l=None)

        # decoder
        pred = self.forward_ca_decoder(latent_img1, latent_cv2, ids_restore_img1)
        loss = self.forward_loss(img2, pred, mask_img1)

        return loss, pred, mask_img1
    
    def forward_ca_decoder(self, latent1, latent2, ids_restore):

        # encoder to decoder layer
        fea1 = self.decoder_embed(latent1)
        fea2 = self.decoder_embed(latent2)

        # append masked tokens to the sequence
        masked_tokens = self.mask_token.repeat(fea1.shape[0],
                                            ids_restore.shape[1] + 1 - fea1.shape[1], 1)
        fea1_ = torch.cat([fea1[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea1_ = torch.gather(fea1_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, fea1.shape[2]))  # unshuffle
        fea1 = torch.cat([fea1[:, :1, :], fea1_], dim=1)  # append cls token

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
