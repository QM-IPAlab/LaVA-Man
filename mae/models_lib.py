"""
Choose which model to load
"""
from mae.models_mae_robot_fuse import MAERobotLangFuse
from functools import partial
import torch.nn as nn

def mae_vit_base_patch16_fuse(**kwargs):
    model = MAERobotLangFuse(
        patch_size=16, embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# new models
mae_fuse = mae_vit_base_patch16_fuse