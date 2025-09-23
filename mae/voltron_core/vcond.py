"""
vcond.py

PyTorch Module defining the Voltron `V-Cond` variant (single-frame with language-conditioning). In general, follows the
MAE recipe, with the architectural modifications described in the paper:
    - RMSNorm, for stability/performance ("Do Transformer Modifications Transfer...")
    - SwishGLU activations in the Attention Block Feed-Forward MLP (gated linear units) as used in PaLM
    - LayerScale with a default value of 0.1 (from Mistral/CaIT)

References:
    - https://github.com/facebookresearch/mae
    - https://github.com/lucidrains/x-transformers
"""
import re
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from einops import rearrange, repeat
import numpy as np
from mae.util.misc import PatchEmbedVarSize
from voltron.models.util.transformer import Block, RMSNorm, get_1D_sine_cosine, PatchEmbed
from mae.util.pos_embed import get_2d_varsize_sincos_pos_embed

# Suppress Transformers Logging
transformers.logging.set_verbosity_error()
CACHE_PATH = "/home/a/acw694/CLIPort_new_loss/cache"
#CACHE_PATH = "/home/robot/Repositories_chaoran/CLIPort_new_loss"

def get_2D_position_embeddings_ours(embed_dim: int, h: int, w: int, cls_token: bool = False):
    # Create 2D Position embeddings by taking cross product of height and width and splicing 1D embeddings...
    grid_h, grid_w = np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, w, h)  # w goes first?

    # Use half of dimensions to encode grid_h, other half to encode grid_w
    emb_h, emb_w = get_1D_sine_cosine(embed_dim // 2, grid[0]), get_1D_sine_cosine(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    # CLS token handling (only for R-MVP)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed

class VCond(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (224,224),
        patch_size: int = 16,
        encoder_depth: int = 12,
        encoder_embed_dim: int = 384,
        encoder_n_heads: int = 6,
        decoder_depth: int = 6,
        decoder_embed_dim: int = 192,
        decoder_n_heads: int = 6,
        language_dim: int = 768,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        norm_pixel_loss: bool = True,
        use_cls_token: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize a VCond model with the requisite architecture parameters.

        :param resolution: Base image resolution -- usually 224 (ImageNet size).
        :param patch_size: Height/Width of each patch in pixels -- usually 16.
        :param encoder_depth: Number of Transformer blocks in the encoder -- should be greater than decoder.
        :param encoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param encoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param decoder_depth: Number of Transformer blocks in the decoder -- should be relatively shallow.
        :param decoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param decoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param language_model: Language model to freeze for encoding narrations/utterances.
        :param hf_cache: Cache directory to store pretrained models, for safe distributed training.
        :param language_dim: Dimensionality of the language embedding coming out of the pretrained LM.
        :param optimizer: String denoting which optimizer to use (for MAEs, usually `adamw`)
        :param schedule: Learning rate schedule to use; for Transformers a linear warmup + decay is recommended!
        :param base_lr: Base learning rate, to be scaled via a linear scaling rule (from scaling laws).
        :param min_lr: Minimum learning rate to decay to over the course of learning (usually 0.0)
        :param effective_bsz: Global batch size for update, dictates the scaling of the base_lr.
        :param betas: Adam optimizer betas (only applicable for `adam` and `adamw`. Prevents early loss spiking.
        :param weight_decay: Weight decay for global weight regularization (only applied to non-bias, non-LN layers).
        :param warmup_epochs: Number of epochs to warmup learning rate for linear warmup schedule.
        :param max_epochs: Total number of training epochs to be run.
        :param mask_ratio: Ratio for number of patches to mask out for MAE -- should be fairly high!
        :param mlp_ratio: Ratio for embedding size to Position-wise Feed-Forward MLP (gets shrunk back down).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param norm_pixel_loss: Normalize decoder pixel targets for reconstruction (better perf, not interpretable).
        :param use_cls_token: Add <CLS> token for continued pretraining (NOTE: not used in MAE pretraining/finetuning!)
        """
        super().__init__()
        self.resolution, self.patch_size, self.mask_ratio = img_size, patch_size, mask_ratio
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.use_cls_token = use_cls_token
        self.language_dim = language_dim

        # Encoder/Decoder Parameters
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_embed_dim, self.encoder_n_heads = encoder_embed_dim, encoder_n_heads
        self.decoder_embed_dim, self.decoder_n_heads = decoder_embed_dim, decoder_n_heads

        # General Parameters (for downstream adaptation)
        self.embed_dim, self.n_heads = self.encoder_embed_dim, self.encoder_n_heads

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # MAE Encoder Parameters
        self.patch2embed = PatchEmbedVarSize(
            self.resolution, self.patch_size, embed_dim=self.encoder_embed_dim, in_chans=self.in_channels
        )
        #self.patch2embed = PatchEmbed(
        #    224, self.patch_size, embed_dim=self.encoder_embed_dim, in_channels=self.in_channels
        #)
        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.encoder_embed_dim),
            requires_grad=False,
        )
        self.patch_embed = self.patch2embed
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    self.encoder_embed_dim,
                    self.encoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.encoder_depth)
            ]
        )
        self.encoder_norm = RMSNorm(self.encoder_embed_dim)

        # Projection from Language Embedding to Decoder
        self.lang2encoder = nn.Linear(self.language_dim, self.encoder_embed_dim)

        # Projection from Encoder to Decoder
        self.encoder2decoder = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)

        # MAE Decoder Parameters -- Remember the CLS Token (if specified)!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.decoder_embed_dim,
                    self.decoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = RMSNorm(self.decoder_embed_dim)
        self.decoder_prediction = nn.Linear(self.decoder_embed_dim, (patch_size**2) * in_channels, bias=True)

        # VCond -- Add "Image" and "Language" Modifier Tokens...
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.lang_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # Initialize all Weights
        self.initialize_weights()

        # @AFTER INITIALIZATION -- Create Language Model & Language Reward MLP --> LM has requires_grad = False
        #   > For BERT models, our "embedding" is just going to be the last hidden state
        #   > Assumes inputs to forward pass are pre-tokenized!
        #self.tokenizer = transformers.AutoTokenizer.from_pretrained(language_model, cache_dir=hf_cache)
        self.lm = transformers.AutoModel.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_PATH)
        self.lm.eval()

        # Shape Assertion -- make sure self.language_dim actually is the same as the LM dimension!
        assert self.lm.config.dim == self.language_dim, "Language model embedding dimension != self.language_dim!"

        # Freeze the LM
        for _, param in self.lm.named_parameters():
            param.requires_grad = False

    def initialize_weights(self) -> None:
        # Position Encoding -- Fixed 2D Sine-Cosine Embeddings
        enc_pe = get_2d_varsize_sincos_pos_embed(
            self.encoder_embed_dim, 
            int(self.resolution[0] // self.patch_size),
            int(self.resolution[1]// self.patch_size),
            cls_token=self.use_cls_token
        )
        self.encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        dec_pe = get_2d_varsize_sincos_pos_embed(
            self.decoder_embed_dim, 
            int(self.resolution[0] // self.patch_size),
            int(self.resolution[1] // self.patch_size), 
              cls_token=self.use_cls_token
        )
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize Mask Token, Img Token, Lang Token w/ Truncated Normal
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.img_token, std=0.02)
        nn.init.normal_(self.lang_token, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

        # Default Transformer initializer on everything else...
        self.apply(self.transformer_initializer)

    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        # Use `xavier_uniform` following Jax ViT
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def encode_language(self, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Encode language by feeding the *pre-tokenized text* through the frozen language model."""
        self.lm.eval()
        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state
        return transformer_embeddings

    def mask(
        self, patches: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by shuffling :: uses argsort random noise to identify masked patches."""
        bsz, n_patches, embed_dim = patches.shape
        if mask_ratio is not None:
            n_keep = int(n_patches * (1 - mask_ratio))
        else:
            n_keep = int(n_patches * (1 - self.mask_ratio))

        # Sample noise of n_patches size, argsort to get shuffled IDs, argsort again to get "unshuffle"
        #   > For clarity -- argsort is an invertible transformation (if argsort `restore`, recovers `shuffle`)
        shuffle_idxs = torch.argsort(torch.rand(bsz, n_patches, device=patches.device), dim=1)
        restore_idxs = torch.argsort(shuffle_idxs, dim=1)

        # Get "keep" (visible) patches
        visible_patches = torch.gather(patches, dim=1, index=shuffle_idxs[:, :n_keep, None].repeat(1, 1, embed_dim))

        # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following MAE convention)
        mask = torch.ones(bsz, n_patches, device=patches.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_idxs)

        return visible_patches, mask, restore_idxs

    def forward_refer(self, img, language):
        return self.get_representations(img, language)


    def get_representations(
        self, img: torch.Tensor, language: Optional[Union[List[str], Tuple[str]]] = None, mode: str = "multimodal"
    ) -> torch.Tensor:
        """
        Given either a singleton (img, language) pair or a batch of images and language, extract representations
        subject to the specified mode in < multimodal | visual >.

        :param img: Processed batch of images :: [bsz, 3, 224, 224]
        :param language: Input language as `List[str] | Tuple[str] | None`
        :param mode: Type of representations to extract -- `multimodal` (both vision+text), `visual` (visual only)

        :return: Extracted representations given (img, language) input as sequence.
        """
        assert img.ndim == 4 
        #and (
        #    language is None or isinstance(language, list) or isinstance(language, tuple)
        #), "Invalid input to `get_representations()`"
        assert mode in {"multimodal", "visual"}, f"Extraction mode `{mode}` not supported!"

        # Tokenize Language --> note max length is 20!
        if language is None:
            lang, lang_mask = [torch.zeros(img.shape[0], 20, dtype=int, device=self.lm.device) for _ in range(2)]
            lang[:, 0], lang_mask[:, 0] = self.tokenizer.cls_token_id, 1
        else:
            #tokens = self.tokenizer(language, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
            lang, lang_mask = language["input_ids"].to(self.lm.device), language["attention_mask"].to(self.lm.device)

            # Tile Language & Language Mask if mismatch with # images!
            if not len(lang) == len(img):
                lang = repeat(lang, "b seq -> (bsz b) seq", bsz=img.size(0))
                lang_mask = repeat(lang_mask, "b seq -> (bsz b) seq", bsz=img.size(0))

        # Extract desired representations...
        representations = self.encode(img, lang, lang_mask)
        return representations if mode == "multimodal" else representations[:, : -lang_mask.shape[-1]]

    def encode(self, img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Default representation extraction function, given a batch of images and tokenized language."""
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_language = self.lang2encoder(lang_embeddings)

        # Patchify
        patches = self.patch2embed(img)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)

        # Position Encoding
        encoder_pe = self.encoder_pe
        if encoder_pe.shape[1] != patches.shape[1]:
            # Dynamic Position Embedding
           encoder_pe = self.interpolate_pos_encoding(patches, encoder_pe, 320, 160)
        patches_pe = patches + self.encoder_pe

        # Add "modality" embeddings to patches & language
        img_embeddings, lang_embeddings = patches_pe + self.img_token, projected_language + self.lang_token

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        patches_mask = torch.ones_like(img_embeddings[..., -1], dtype=lang_mask.dtype)
        multimodal_embeddings = torch.cat([img_embeddings, lang_embeddings], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([patches_mask, lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embeddings = block(multimodal_embeddings, multimodal_mask)
        multimodal_embeddings = self.encoder_norm(multimodal_embeddings)

        # Return the full sequence of multimodal embeddings...
        return multimodal_embeddings

    def forward_encoder(
        self, img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_lang = self.lang2encoder(lang_embeddings)

        # Patchify + Position Embedding (without <CLS> Token!)
        patches = self.patch2embed(img)
        # Dynamic Position Embedding
        encoder_pe = self.encoder_pe
        if encoder_pe.shape[1] != patches.shape[1]:
            # Dynamic Position Embedding
            encoder_pe = self.interpolate_pos_encoding(patches, encoder_pe, 320, 160)
        patches_pe = patches + (encoder_pe if not self.use_cls_token else encoder_pe[:, 1:, :])

        # Create mask (and go ahead and mask out patches at the same time)
        visible_patches, mask, restore_idxs = self.mask(patches_pe, mask_ratio)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :]
            cls_tokens = cls_token_pe.expand(img.shape[0], -1, -1)
            visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)

        # Add "modality" embeddings to patches & language
        visible_patches, projected_lang = visible_patches + self.img_token, projected_lang + self.lang_token

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=lang_mask.dtype)
        multimodal_embedding = torch.cat([visible_patches, projected_lang], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([visible_mask, lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embedding = block(multimodal_embedding, multimodal_mask)
        multimodal_embedding = self.encoder_norm(multimodal_embedding)

        # Split multimodal embedding, remove language and return only the visible patches (+ optional <CLS> token)!
        visible_patches = multimodal_embedding[:, : -lang_mask.shape[-1], ...]
        return visible_patches, mask, restore_idxs

    def forward_decoder(self, visible_patches: torch.Tensor, restore_idxs: torch.Tensor) -> torch.Tensor:
        # Project patches into decoder embedding dimension
        projected_patches = self.encoder2decoder(visible_patches)

        # Add Mask Tokens to Sequence & Unshuffle
        mask_tokens = self.mask_token.repeat(
            projected_patches.shape[0],
            restore_idxs.shape[1] - visible_patches.shape[1] + (1 if self.use_cls_token else 0),
            1,
        )

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            # Remove & add back CLS Token as part of the "unshuffling"
            concatenated_patches = torch.cat([projected_patches[:, 1:, :], mask_tokens], dim=1)  # Skip CLS Token
            no_cls_unshuffled_patches = torch.gather(
                concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
            )
            unshuffled_patches = torch.cat([projected_patches[:, :1, :], no_cls_unshuffled_patches], dim=1)
        else:
            concatenated_patches = torch.cat([projected_patches, mask_tokens], dim=1)
            unshuffled_patches = torch.gather(
                concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
            )

        # Add Position Embeddings
        decoder_pe = self.decoder_pe
        if decoder_pe.shape[1] != unshuffled_patches.shape[1]:
            # Dynamic Position Embedding
            decoder_pe = self.interpolate_pos_encoding(unshuffled_patches, decoder_pe, 320, 160)
        decoder_patches = unshuffled_patches + decoder_pe

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        decoder_patches = self.decoder_norm(decoder_patches)

        # Run final projection & return --> note <CLS> token handling!
        decoder_prediction = self.decoder_prediction(decoder_patches)
        return decoder_prediction if not self.use_cls_token else decoder_prediction[:, 1:, :]

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images to their patched equivalents, by naive reshaping"""
        return rearrange(
            imgs,
            "bsz c (height patch_h) (width patch_w) -> bsz (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = self.resolution[0] // p
        w = self.resolution[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def compute_loss(self, imgs: torch.Tensor, reconstructions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.norm_pixel_loss, "`norm_pixel_loss` should always be true... false only for visualizations!"
        targets = self.patchify(imgs)

        # Normalize targets...
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        # Compute mean loss per patch first...
        mse = (reconstructions - targets) ** 2
        avg_loss_per_patch = mse.mean(dim=-1)

        # Compute mean loss only on *removed* patches and return
        return (avg_loss_per_patch * mask).sum() / mask.sum()

    def forward(
        self, 
        img: torch.Tensor,
        img2: torch.Tensor,
        pick: torch.Tensor,
        place: torch.Tensor,
        lang: dict, 
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        input_ids, attention_mask = lang["input_ids"], lang["attention_mask"]
        visible_patches, mask, restore_idxs = self.forward_encoder(img, input_ids, attention_mask, mask_ratio)
        reconstructions = self.forward_decoder(visible_patches, restore_idxs)
        loss = self.compute_loss(img, reconstructions, mask)
        return loss, reconstructions, mask

    def cliport_forward(self, img: torch.Tensor, lang: dict) -> torch.Tensor:
        input_ids, attention_mask = lang["input_ids"], lang["attention_mask"]
        _, _, H, W = img.shape
        self.img_size = (H, W)
        visible_patches = self.clip_encode(img, input_ids, attention_mask, H, W)
        #reconstructions = self.clip_decode(visible_patches, H, W)
        return visible_patches

    def clip_encode(self, img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:

        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_lang = self.lang2encoder(lang_embeddings)

        # Patchify + Position Embedding (without <CLS> Token!)
        patches = self.patch2embed(img)
        
        encoder_pe = self.encoder_pe
        if encoder_pe.shape[1] != patches.shape[1]:
            # Dynamic Position Embedding
            encoder_pe = self.interpolate_pos_encoding(patches, encoder_pe, H, W)
        patches_pe = patches + (encoder_pe if not self.use_cls_token else encoder_pe[:, 1:, :])
    
        # Create mask (and go ahead and mask out patches at the same time)
        visible_patches = patches_pe

        # Add "modality" embeddings to patches & language
        visible_patches, projected_lang = visible_patches + self.img_token, projected_lang + self.lang_token

        if visible_patches.shape[0] != projected_lang.shape[0]:
            projected_lang = projected_lang.repeat([visible_patches.shape[0]//projected_lang.shape[0], 1, 1])

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        multimodal_embedding = torch.cat([visible_patches, projected_lang], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embedding = block(multimodal_embedding)
        multimodal_embedding = self.encoder_norm(multimodal_embedding)

        # Split multimodal embedding, remove language and return only the visible patches (+ optional <CLS> token)!
        visible_patches = multimodal_embedding[:, :visible_patches.size(1), ...]
        return visible_patches
    
    def clip_decode(self, visible_patches: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Project patches into decoder embedding dimension
        projected_patches = self.encoder2decoder(visible_patches)

        # Dynamic Position Embedding
        decoder_position_embeddings = get_2d_varsize_sincos_pos_embed(self.decoder_embed_dim, H // self.patch_size, W // self.patch_size)
        decoder_position_embeddings = torch.from_numpy(decoder_position_embeddings).to(projected_patches.device)
        decoder_position_embeddings = decoder_position_embeddings.unsqueeze(0).expand(projected_patches.size(0), -1, -1)

        # Add Position Embeddings
        decoder_patches = projected_patches + decoder_position_embeddings

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        decoder_patches = self.decoder_norm(decoder_patches)

        # Run final projection & return --> note <CLS> token handling!
        decoder_prediction = self.decoder_prediction(decoder_patches)
        return decoder_prediction
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, position_embeddings, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        num_patches = embeddings.shape[1] 
        num_positions = position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return position_embeddings

        patch_pos_embed = position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed
    

class MVP(VCond):
    
    def forward(
        self, 
        img: torch.Tensor,
        img2: torch.Tensor,
        pick: torch.Tensor,
        place: torch.Tensor,
        lang: dict, 
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        visible_patches, mask, restore_idxs = self.forward_encoder(img, mask_ratio)
        reconstructions = self.forward_decoder(visible_patches, restore_idxs)
        loss = self.compute_loss(img, reconstructions, mask)
        return loss, reconstructions, mask

    def forward_encoder(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Patchify + Position Embedding (without the CLS Token)
        patches = self.patch2embed(imgs)
        patches_pe = patches + self.encoder_pe[:, 1:, :]

        # Create mask (and go ahead and mask out patches at the same time)
        visible_patches, mask, restore_idxs = self.mask(patches_pe, mask_ratio)

        # Add the CLS Token
        cls_token = self.cls_token + self.encoder_pe[:, :1, :]
        cls_tokens = cls_token.expand(imgs.shape[0], -1, -1)
        cls_visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            cls_visible_patches = block(cls_visible_patches)
        cls_visible_patches = self.encoder_norm(cls_visible_patches)

        return cls_visible_patches, mask, restore_idxs

    def forward_decoder(self, visible_patches: torch.Tensor, restore_idxs: torch.Tensor) -> torch.Tensor:
        # Project patches into decoder embedding dimension
        projected_patches = self.encoder2decoder(visible_patches)

        # Add Mask Tokens to Sequence
        mask_tokens = self.mask_token.repeat(
            projected_patches.shape[0], restore_idxs.shape[1] - visible_patches.shape[1] + 1, 1
        )

        # Remove & add back CLS Token as part of the "unshuffling"
        concatenated_patches = torch.cat([projected_patches[:, 1:, :], mask_tokens], dim=1)  # Skip CLS Token
        unshuffled_patches = torch.gather(
            concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
        )
        cls_unshuffled_patches = torch.cat([projected_patches[:, :1, :], unshuffled_patches], dim=1)  # Add CLS Token

        # Add Position Embeddings
        cls_decoder_patches = cls_unshuffled_patches + self.decoder_pe

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            cls_decoder_patches = block(cls_decoder_patches)
        cls_decoder_patches = self.decoder_norm(cls_decoder_patches)

        # Run final projection, remove the CLS token, and return
        cls_decoder_prediction = self.decoder_prediction(cls_decoder_patches)
        decoder_prediction = cls_decoder_prediction[:, 1:, :]
        return decoder_prediction