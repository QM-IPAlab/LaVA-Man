"""
Change my current model's name in order to load corco's checkpoints
"""
import torch
import torch.nn as nn

from transformers import CLIPTextModel
import transformers
from mae.blocks import Block, DecoderCABlockLang
from mae.util.misc import PatchEmbedVarSize
from mae.util.pos_embed import get_2d_varsize_sincos_pos_embed
transformers.logging.set_verbosity_error()

class MAERobotLangCroco(nn.Module):
    def __init__(self, img_size=(320, 160), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_im2_in_dec=True, norm_pix_loss=False,
                 text_model="openai/clip-vit-base-patch32"):
        super().__init__()

        self.img_size = img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbedVarSize(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.enc_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.enc_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding


        self.dec_norm = norm_layer(decoder_embed_dim)
        self.prediction_head = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # store other parameters:
        self.embed_dim = decoder_embed_dim
        self.n_heads = decoder_num_heads

        self.dec_blocks = nn.ModuleList([
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
    
    def forward_refer(self, img, processed_lang):
        
        latent1, mask1, ids_restore1 = self.forward_encoder(img, mask_ratio=0.0)
        lang_emb = self.get_lang_embed(processed_lang)

        fea = self.decoder_embed(latent1)

        fea = fea + self.dec_pos_embed

        out1 = fea
        out2 = None

        for blk in self.dec_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.dec_norm(out1)
        out = out[:, 1:, :]
        return out

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
        if self.dec_pos_embed is not None:
            fea1 = fea1 + self.dec_pos_embed
            fea2 = fea2 + self.dec_pos_embed

        out1 = fea1
        out2 = fea2
        # apply Transformer blocks
        for blk in self.dec_blocks:
            out1, out2 = blk(out1, out2, lang_emb)
        out = self.dec_norm(out1)

        out = self.prediction_head(out)
        out = out[:, 1:, :]

        return out
    
    def initialize_weights(self):
        
        patch_size = self.patch_embed.patch_size[0]
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_varsize_sincos_pos_embed(self.enc_pos_embed.shape[-1],
                                                    int(self.img_size[0] // patch_size), int(self.img_size[1] // patch_size),
                                                    cls_token=True)
        self.enc_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        decoder_pos_embed = get_2d_varsize_sincos_pos_embed(self.dec_pos_embed.shape[-1],
                                                            int(self.img_size[0] // patch_size), int(self.img_size[1] // patch_size),
                                                            cls_token=True)
        self.dec_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.enc_pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.enc_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)

        return x, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
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
        return loss