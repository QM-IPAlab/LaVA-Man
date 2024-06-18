"""
Extract the relevance map using clip first and then use MAE
"""

from re import T
import torch
import torch.nn as nn

from transformers import CLIPModel
from blocks import DecoderCABlockLang
from models_mae_robot import MAERobot
import transformers
from torchvision import transforms
import torch.nn.functional as F
from mae.relevance_tools import interpret_ours


transformers.logging.set_verbosity_error()

class MAERobotLangRel(MAERobot):
    """
    MAE model with language and relevance map
    """
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=4,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.resize_transform = transforms.Resize((224, 224))
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)
        self.clip.requires_grad_(False)

    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""
        
        lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
        lang_emb = lang_emb[0] # (b, 77, 512)
        
        if processed_img.shape[-1] != 64:
            processed_img = F.pad(processed_img, (80, 80, 0, 0), value=0)
            target_size = 320
        else:
            target_size = 64

        processed_img = self.resize_transform(processed_img)

        lang_ids = processed_lang['input_ids']
        if lang_ids.shape[0] != processed_img.shape[0]:
            lang_ids = lang_ids.repeat(processed_img.shape[0], 1)
        relevance_map = interpret_ours(processed_img, lang_ids, self.clip, 'cuda')

        dim = int(relevance_map.shape[-1]** 0.5)
        relevance_map = relevance_map.reshape(-1,dim,dim)
        relevance_map = relevance_map.unsqueeze(1)
        relevance_map = F.interpolate(relevance_map, size=target_size, mode='bilinear')

        if relevance_map.shape[-1] != 64:
            relevance_map = relevance_map[:, :, :, 80:-80]

        return lang_emb.detach(), relevance_map.detach()

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        
        lang_emb, relevance_map = self.get_hidden_embeds(lang, img1)
        assert relevance_map.shape[-2:] == img1.shape[-2:]

        img1_rev = torch.cat([img1, relevance_map], dim=1)
        img2_rev = torch.cat([img2, relevance_map], dim=1)

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1_rev, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2_rev, mask_ratio)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2

    def show_relevance_map(self, img, lang):
        lang_emb, relevance_map = self.get_hidden_embeds(lang, img)
        return relevance_map
    
    def cliport_forward(self, rgb, processed_lang):
        lang_emb, relevance_map = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])

        img1_rev = torch.cat([rgb, relevance_map], dim=1)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1_rev, mask_ratio=0.0)
        
        fea = self.decoder_embed(latent1)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)

        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out
