from transformers import CLIPTextModel
from blocks import DecoderCABlockLang
from models_mae_robot import MAERobot
from mae.blocks import DropPath



class MAERobotLangFuse(MAERobot):
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

        self.task_token1 = nn.Parameter(torch.randn(1, decoder_embed_dim))
        self.task_token2 = nn.Parameter(torch.randn(1, decoder_embed_dim))
        torch.nn.init.normal_(self.task_token1, std=.02)
        torch.nn.init.normal_(self.task_token2, std=.02)

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

        # save image
        # def save_image(img, path='saved_img.png'):
        #     from torchvision.utils import save_image
        #     #(C, H, W)
        #     img = (img- img.min()) / (img.max() - img.min())
        #     save_image(img, path)

        # save_image(img1[1], 'img1.png')
        # save_image(img2[1], 'img2.png')
        # save_image(self.unpatchify(pred)[1], 'pred.png')

        # noise_latent2 = torch.randn_like(latent2).to(latent2.device)
        # pred_noise_latent2 = self.forward_ca_decoder(latent1, noise_latent2, ids_restore2,lang_emb)
        # save_image(self.unpatchify(pred_noise_latent2)[1], 'pred_noise_latent2.png')
        # import pdb; pdb.set_trace()

        return loss, pred, mask2
    
    def forward_ca_decoder(self, latent1, latent2, ids_restore2, lang_emb):
        """
        Dert style deocder
        """

        # fuse the two modalities
        lang_emb = lang_emb[0]
        lang_emb2 = lang_emb
        for fuse_block in self.fuse_blocks:
            latent1, lang_emb = fuse_block(latent1, lang_emb, attention_mask_v=None, attention_mask_l=None)
            latent2, lang_emb2 = fuse_block(latent2, lang_emb2, attention_mask_v=None, attention_mask_l=None)
            

        # encoder to decoder layer
        fea1 = self.decoder_embed(latent1)
        fea2 = self.decoder_embed(latent2)

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

        task_token1 = self.task_token1.expand(fea1.shape[0], -1)
        task_token2 = self.task_token2.expand(fea2.shape[0], -1)
        # add to the cls token
        fea1[:, 0] = fea1[:, 0] + task_token1
        fea2[:, 0] = fea2[:, 0] + task_token2

        out1 = fea1
        out2 = fea2
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            out1, out2 = blk(out1, out2, None)
        out = self.decoder_norm(out1)

        out = self.decoder_pred(out)
        out = out[:, 1:, :]

        return out