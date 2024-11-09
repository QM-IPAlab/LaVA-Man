import torch
import torch.nn as nn

from transformers import CLIPModel
from mae.blocks import DecoderCABlockLang
from mae.models_mae_robot import MAERobot
import transformers
from torchvision import transforms
import torch.nn.functional as F
transformers.logging.set_verbosity_error()


class MAERobotLangCLIPLoss(MAERobot):
    """ CLIP vision model in Encoder
    """

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
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.requires_grad_(False)
        self.resize_transform = transforms.Resize((224, 224))

    def get_pooled_embeds(self, processed_lang, processed_img):
        """get the embeddings of the language and the image
           different from the robotlang model, we use CLIP feature here,
           as the CLIP vison and Text model have different dimension and
           are not aligned, so we have to use projection layer
        """
        
        with torch.no_grad():
            lang_emb = self.clip.get_text_features(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb.unsqueeze(1)
            processed_img = F.pad(processed_img, (80, 80, 0, 0), value=0)
            processed_img = self.resize_transform(processed_img)

            img_emb = self.clip.get_image_features(processed_img, output_hidden_states=True, return_dict=False) 
            img_emb = img_emb.unsqueeze(1)
        return lang_emb, img_emb
    
    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""
        
        with torch.no_grad():
            lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb[0] # (b, 77, 512)
            
            processed_img = F.pad(processed_img, (80, 80, 0, 0), value=0)
            processed_img = self.resize_transform(processed_img)

            img_emb = self.clip.vision_model(processed_img, output_hidden_states=True, return_dict=False) 
            img_emb = img_emb[0] # (b, 50, 768)
        return lang_emb, img_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        # encoder of the language goal
        lang_emb, img_emb = self.get_hidden_embeds(lang, img1)
        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)

        # decoder
        import pdb; pdb.set_trace()
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2, img_emb)

        return loss, pred, mask2

    def forward_loss(self, imgs, pred, mask, clip_emb):
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
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        target_fea = self.patchify_feature(clip_emb)


        return loss
    
    def patchify_feature(self, clip_emb):
        import pdb; pdb.set_trace()
        p = self.patch_embed.patch_size[0]
        assert clip_emb.shape[2] % p == 0 and clip_emb.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x
    

    def cliport_forward(self, rgb, processed_lang):
        
        lang_emb, img_emb = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])
        
        latent, mask, ids_restore = self.forward_ca_encoder(rgb, img_emb, mask_ratio=0)
        
        fea = self.decoder_embed(latent)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out