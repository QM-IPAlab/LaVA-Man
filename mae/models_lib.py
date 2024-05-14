"""
Choose which model to load
"""
from models_mae_robot import MAERobotBase, MAERobot
from models_mae_robot_lang import MAERobotLang
from functools import partial
import torch.nn as nn


def mae_vit_base_patch16_rl(**kwargs):
    model = MAERobotLang(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_robot(**kwargs):
    model = MAERobot(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_robot_base(**kwargs):
    model = MAERobotBase(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# models
mae_robot_base = mae_vit_base_patch16_robot_base  # original mae model with cliport image
mae_robot = mae_vit_base_patch16_robot  # two state mae without language
mae_robot_lang = mae_vit_base_patch16_rl
mae_robot_lang_encoder = ''  # two state with language in both encoder and decoder
