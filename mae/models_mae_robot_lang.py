import torch
import torch.nn as nn

from transformers import AutoTokenizer, CLIPTextModel
from blocks import DecoderCABlockLang
from models_mae_robot import MAERobot
import transformers

transformers.logging.set_verbosity_error()


class MAERobotLang(MAERobot):
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
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

        # The CLIP model
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.requires_grad_(False)

        # modify name
        self.decoder_pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
                                               requires_grad=False)
        del self.decoder_pos_embed_2

    def get_lang_embed(self, lang):
        with torch.no_grad():
            if type(lang) is str:
                decoded_strings = [lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
            input_lang = self.text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
            input_lang.to(self.clip_text.device)
            lang_emb = self.clip_text(**input_lang, return_dict=False)
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        self.decoder_pos_embed_2 = self.decoder_pos_embed2

        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)

        # encoder of the language goal
        lang_emb = self.get_lang_embed(lang)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2
