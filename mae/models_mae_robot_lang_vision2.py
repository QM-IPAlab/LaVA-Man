import torch
import torch.nn as nn

from transformers import CLIPModel
from blocks import DecoderCLIPBlock
from models_mae_robot import MAERobot
import transformers
from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange
transformers.logging.set_verbosity_error()
from cliport.models.core.clip import build_model, load_clip, tokenize
from mae.models_mae_clip import CLIPMaskModel
from blocks import DecoderCABlockLang


class MAERobotLangVisonCLIP(MAERobot):
    """ CLIP vision model and text model cross-attention
        Then cross attention to the image decoder
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False, **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCLIPBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip.text_model.requires_grad_(False)
        self.clip.vision_model.requires_grad_(True)
        self.resize_transform = transforms.Resize((224, 224))
        self.decoder_pred = nn.Linear(decoder_embed_dim, 16 ** 2 * in_chans, bias=True)

        # self.interpolated_pos_embed = F.interpolate(
        #     self.decoder_pos_embed.unsqueeze(1),
        #     size=(50, 512),
        #     mode='bilinear',
        #     align_corners=False
        #     ).squeeze(1)

    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""    
        with torch.no_grad():
            lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb[0] # (b, 77, 512)
            # processed_img = F.pad(processed_img, (80, 80, 0, 0), value=0)
            # processed_img = self.resize_transform(processed_img)
        
        img_emb = self.clip.vision_model(processed_img, output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True) 
        img_emb = img_emb[0] # (b, 50, 768)
        
        return lang_emb, img_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        # encoder of the language goal
        lang_emb, img_emb = self.get_hidden_embeds(lang, img1)
        
        # decoder
        pred = self.forward_ca_decoder(img_emb, lang_emb) # [N, L, p*p*3]
        loss = self.forward_loss(img2, pred)

        return loss, pred, None
    
    def forward_ca_decoder(self, img_emb, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        # # add positional embedding
        # if self.interpolated_pos_embed.device != img_emb.device:
        #     self.interpolated_pos_embed = self.interpolated_pos_embed.to(img_emb.device)

        fea1 = self.decoder_embed(img_emb)
        
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed

        out1 = fea1
        for blk in self.decoder_blocks:
            out1 = blk(out1, lang_emb)
        out = self.decoder_norm(out1)
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
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()
        return loss
    
    def cliport_forward(self, rgb, processed_lang):

        lang_emb, img_emb = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0]//lang_emb.shape[0], 1, 1])

        fea1 = self.decoder_embed(img_emb)
        
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed
        
        out1 = fea1
        for blk in self.decoder_blocks:
            out1 = blk(out1, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out
       

class MAERobotLangVisonCLIPRes(MAERobot):
    """ Use CLIP vision model as image encoder
        use CLIP resnet 
    """

    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCLIPBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
        self.clip_rn50.requires_grad_(False)

        self.decoder_pred = nn.Linear(decoder_embed_dim, 32 ** 2 * in_chans, bias=True)

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""    
        import pdb; pdb.set_trace()
        with torch.no_grad():
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(processed_lang) 
            # text_feat: [64, 1024], text_emb: [64, 77, 512]
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(processed_img)
            # img_encoding: [64, 2048, 10, 5]
            img_im = self.clip_rn50.visual.prepool(processed_img)
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(processed_img)

        return text_emb, img_im

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        # encoder of the language goal
        lang_emb, img_emb = self.get_hidden_embeds(lang, img1)
        
        # decoder
        img_emb = self.project(img_emb)
        pred = self.forward_ca_decoder(img_emb, lang_emb) # [N, L, p*p*3]
        pred = rearrange(pred, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h=7, w=7, ph=32, pw=32, c=3)
        pred = F.interpolate(pred, size=(320, 320), mode='bilinear', align_corners=False)
        pred = pred[:, :, :, 80:-80]
        loss = self.forward_loss(img2, pred)

        return loss, pred, None
    
    def forward_ca_decoder(self, img_emb, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        for blk in self.decoder_blocks:
            img_emb = blk(img_emb, lang_emb)
        out = self.decoder_norm(img_emb)
        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out
    
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        loss = (pred - imgs) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch
        return loss
    
    def cliport_forward(self, rgb, processed_lang):
        
        lang_emb, img_emb = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0], 1, 1])
        
        img_emb = self.project(img_emb)

        for blk in self.decoder_blocks:
            img_emb = blk(img_emb, lang_emb)
        out = self.decoder_norm(img_emb)
        out = self.decoder_pred(out)
        out = out[:, 1:, :]
        pred = rearrange(out, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h=7, w=7, ph=32, pw=32, c=3)
        pred = F.interpolate(pred, size=(320, 320), mode='bilinear', align_corners=False)
        pred = pred[:, :, :, 80:-80]
        return pred


class MAERobotLangVisonCLIPMask(MAERobot):
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
            DecoderCLIPBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip.text_model.requires_grad_(False)
        self.clip.vision_model.requires_grad_(True)
        self.resize_transform = transforms.Resize((224, 224))
        self.decoder_pred = nn.Linear(decoder_embed_dim, 16 ** 2 * in_chans, bias=True)

        # self.interpolated_pos_embed = F.interpolate(
        #     self.decoder_pos_embed.unsqueeze(1),
        #     size=(50, 512),
        #     mode='bilinear',
        #     align_corners=False
        #     ).squeeze(1)

    def get_hidden_embeds(self, processed_lang, processed_img):
        """get the hidden embeddings of the language and the image"""    
        with torch.no_grad():
            lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb[0] # (b, 77, 512)
            # processed_img = F.pad(processed_img, (80, 80, 0, 0), value=0)
            # processed_img = self.resize_transform(processed_img)
            img_emb = self.clip.vision_model(processed_img, output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True) 
            img_emb = img_emb[0] # (b, 50, 768)
        return lang_emb, img_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        # encoder of the language goal
        lang_emb, img_emb = self.get_hidden_embeds(lang, img1)
        
        # decoder
        pred = self.forward_ca_decoder(img_emb, lang_emb) # [N, L, p*p*3]
        loss = self.forward_loss(img2, pred)

        return loss, pred, None
    
    def forward_ca_decoder(self, img_emb, lang_emb):
        """
        latent1: visible
        masked_latent2: masked goal image
        """
        # # add positional embedding
        # if self.interpolated_pos_embed.device != img_emb.device:
        #     self.interpolated_pos_embed = self.interpolated_pos_embed.to(img_emb.device)

        fea1 = self.decoder_embed(img_emb)
        
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed

        out1 = fea1
        for blk in self.decoder_blocks:
            out1 = blk(out1, lang_emb)
        out = self.decoder_norm(out1)
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
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()
        return loss
    
    def cliport_forward(self, rgb, processed_lang):

        lang_emb, img_emb = self.get_hidden_embeds(processed_lang, rgb)
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0]//lang_emb.shape[0], 1, 1])

        fea1 = self.decoder_embed(img_emb)
        
        if self.decoder_pos_embed is not None:
            fea1 = fea1 + self.decoder_pos_embed
        
        out1 = fea1
        for blk in self.decoder_blocks:
            out1 = blk(out1, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out
    

class MAECLIP(MAERobot):
    """ CLIP vision model and text model cross-attention
        Then cross attention to the image decoder
    """
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False, **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_im2_in_dec, norm_pix_loss)

        self.decoder_blocks = nn.ModuleList([
            DecoderCABlockLang(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                               norm_mem=norm_im2_in_dec)
            for _ in range(decoder_depth)])

        # The CLIP model
        self.clip = CLIPMaskModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip.text_model.requires_grad_(False)
        self.clip.vision_model.requires_grad_(True)

    def get_lang_embed(self, processed_lang):
        with torch.no_grad():
            lang_emb = self.clip.text_model(**processed_lang, output_hidden_states=True, return_dict=False)
            lang_emb = lang_emb[0]
        return lang_emb

    def forward(self, img1, img2, pick=None, place=None, lang=None, mask_ratio=0.75):
        
        # encoder of the two images
        latent1, mask1, ids_restore1 = self.clip.vision_model(mask_ratio=0.0, pixel_values=img1,  output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True) 
        latent2, mask2, ids_restore2 = self.clip.vision_model(mask_ratio=mask_ratio, pixel_values=img2, output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True) 
        
        lang_emb = self.get_lang_embed(lang)
  
        # decoder
        pred = self.forward_ca_decoder(latent1, latent2, ids_restore2, lang_emb) # [N, L, p*p*3]
        loss = self.forward_loss(img2, pred, mask2)

        return loss, pred, mask2
    
    def cliport_forward(self, rgb, processed_lang):

        latent1, mask1, ids_restore1 = self.clip.vision_model(mask_ratio=0.0, pixel_values=rgb,  output_hidden_states=True, return_dict=False, interpolate_pos_encoding=True) 
        lang_emb = self.get_lang_embed(processed_lang)
        
        if rgb.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([rgb.shape[0]//lang_emb.shape[0], 1, 1])

        fea1 = self.decoder_embed(latent1)        
        fea1 = fea1 + self.decoder_pos_embed
        
        out1 = fea1
        out2 = None

        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.decoder_norm(out1)
        out = out[:, 1:, :]
        return out

