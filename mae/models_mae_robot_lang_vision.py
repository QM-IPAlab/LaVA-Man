import torch
import torch.nn as nn

from transformers import CLIPModel
from blocks import EncoderCABlockVision, DecoderCABlockVisionLang, DecoderCABlockLang, DecoderCABlockVisionLangMul
from models_mae_robot import MAERobot
import transformers
from torchvision import transforms
import torch.nn.functional as F
from cliport.models.core.clip import build_model, load_clip, tokenize
from cliport.models.core.unet import Up
from cliport.models.core.fusion import FusionMultOurs
transformers.logging.set_verbosity_error()

class MAERobotLangVisonE(MAERobot):
    """ CLIP vision model in Encoder
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.blocks = nn.ModuleList([
            EncoderCABlockVision(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])

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
        latent1, mask1, ids_restore1 = self.forward_ca_encoder(img1, img_emb, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_ca_encoder(img2, img_emb, mask_ratio)

        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2
    
    def forward_ca_encoder(self, x, vis, mask_ratio):
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
            x = blk(x,vis)
        x = self.norm(x)

        return x, mask, ids_restore
    
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


class MAERobotLangVisonProjector(MAERobot):
    """ CLIP vision model and text model cross-attention
        Then cross attention to the image decoder
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockVisionLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.requires_grad_(False)
        self.resize_transform = transforms.Resize((224, 224))
    
    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""
        
        with torch.no_grad():
            lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb[1] # (b, 512)
            lang_emb = lang_emb.unsqueeze(1)
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
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, img_emb, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2
    
    def forward_ca_decoder(self, latent1, masked_latent2, ids_restore2, img_emb, lang_emb):
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
            out1, out2 = blk(out1, out2, img_emb, lang_emb)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    
    def cliport_forward(self, rgb, processed_lang):
        
        lang_emb, img_emb = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])
        
        latent, mask, ids_restore = self.forward_encoder(rgb, mask_ratio=0.0)
        
        fea = self.decoder_embed(latent)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, img_emb, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out
    

class MAERobotLangVisonProMul(MAERobot):
    """ CLIP vision model and text model multiply
        Then cross attention to the image decoder
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockVisionLangMul(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        self.lang_fuser1 = FusionMultOurs(input_dim=1024)
        self.lang_fuser2 = FusionMultOurs(input_dim=512)
        self.lang_fuser3 = FusionMultOurs(input_dim=256)

        self.lang_proj1 = nn.Linear(1024, 1024)
        self.lang_proj2 = nn.Linear(1024, 512)
        self.lang_proj3 = nn.Linear(1024, 256)

        self.up1 = Up(2048, 1024 // 2)
        self.up2 = Up(1024, 512 // 2)
        self.up3 = Up(512, 256 // 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.resize_transform = transforms.Resize((224, 224))

        self.dtype = self.conv1[0].weight.dtype
        model, _ = load_clip("RN50", device='cuda')
        self.clip = build_model(model.state_dict(), self.dtype).to('cuda')
        self.clip.requires_grad_(False)

    def encode_image(self, img):
        img = F.pad(img, (80, 80, 0, 0), value=0)
        img = self.resize_transform(img)
        with torch.no_grad():
            img_encoding, img_im = self.clip.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = x['input_ids']
            text_feat, text_emb = self.clip.encode_text_with_embeddings(tokens)
        return text_feat, text_emb
    
    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""
        
        lang_enc, _ = self.encode_text(processed_lang) # 64, 1024; 64, 77, 512
        _, img_im = self.encode_image(processed_img) #64, 2048, 7, 7
    
        out = self.cliport_lang_fusion(lang_enc, img_im)
        return out
        
    def cliport_lang_fusion(self, l_input, im):
        x = self.conv1(im[-1]) # 64 1024 7 7
        x = self.lang_fuser1(x, l_input, x2_mask=None, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2]) # # 64 512 14 14
        x = self.lang_fuser2(x, l_input, x2_mask=None, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3]) # #64 512 28 28
        x = self.lang_fuser3(x, l_input, x2_mask=None, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4]) #  64 512 56 
        return x
    
    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        # encoder of the language goal
        ref = self.get_hidden_embeds(lang, img1)
        # encoder of the first observed image (no mask)
        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio=0.0)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)
    
        ref = F.adaptive_max_pool2d(ref, (16, 16))
        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, ref)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2

    def cliport_forward(self, rgb, processed_lang):
         
        ref = self.get_hidden_embeds(processed_lang, rgb)
        ref = F.adaptive_max_pool2d(ref, (16, 16))
        
        latent, mask, ids_restore = self.forward_encoder(rgb, mask_ratio=0.0)
        
        fea = self.decoder_embed(latent)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, ref)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out


class MAERobotLangVisonProjector(MAERobot):
    """ CLIP vision model and text model cross-attention
        Then cross attention to the image decoder
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockVisionLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.requires_grad_(False)
        self.resize_transform = transforms.Resize((224, 224))
    
    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""
        
        with torch.no_grad():
            lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb[1] # (b, 512)
            lang_emb = lang_emb.unsqueeze(1)
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
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, img_emb, lang_emb)
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2
    
    def forward_ca_decoder(self, latent1, masked_latent2, ids_restore2, img_emb, lang_emb):
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
            out1, out2 = blk(out1, out2, img_emb, lang_emb)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    
    def cliport_forward(self, rgb, processed_lang):
        
        lang_emb, img_emb = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])
        
        latent, mask, ids_restore = self.forward_encoder(rgb, mask_ratio=0.0)
        
        fea = self.decoder_embed(latent)
        fea = fea + self.decoder_pos_embed

        out1 = fea
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, img_emb, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out