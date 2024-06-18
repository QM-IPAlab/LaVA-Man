"""
Choose which model to load
"""
from models_mae_robot import MAERobotBase, MAERobot
from models_mae_robot_lang import MAERobotLang, MAERobotLangNoRef, MAERobotLang2
from models_mae_robot_lang_vision import MAERobotLangVisonE, MAERobotLangVisonProjector, MAERobotLangVisonProMul
from models_mae_robot_lang_relevance import MAERobotLangRel
from functools import partial
import torch.nn as nn


def mae_vit_base_patch16_rl_noref(**kwargs):
    model = MAERobotLangNoRef(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rl(**kwargs):
    model = MAERobotLang(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch8_rl(**kwargs):
    model = MAERobotLang(
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
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

def mae_vit_base_patch16_rlv(**kwargs):
    """MAE with clip vision in the encoder part"""
    model = MAERobotLangVisonE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rlre(**kwargs):
    """MAE with relevance map"""
    model = MAERobotLangRel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_path16_rl2(**kwargs):
    """MAE with language in both encoder and decoder"""
    model = MAERobotLang2(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_path16_rlp(**kwargs):
    """MAE with cross attention between clip text and clip vision in decoder"""
    model = MAERobotLangVisonProjector(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_path16_rlpm(**kwargs):
    """MAE with multiplication between clip text and clip vision in deocder"""
    model = MAERobotLangVisonProMul(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# models
mae_robot_base = mae_vit_base_patch16_robot_base  # original mae model with cliport image
mae_robot = mae_vit_base_patch16_robot  # two state mae without language
mae_robot_lang = mae_vit_base_patch16_rl
mae_robot_lang_p8 = mae_vit_base_patch8_rl
mae_robot_lang_noref = mae_vit_base_patch16_rl_noref  # two state mae with language in decoder only
mae_robot_lang_visonencoder = mae_vit_base_patch16_rlv  # two state mae with language in decoder only
mae_robot_lang_relevance = mae_vit_base_patch16_rlre
mae_robot_lang2 = mae_vit_base_path16_rl2  # two state with language in both encoder and decoder
mae_robot_projector = mae_vit_base_path16_rlp
mae_robot_promul = mae_vit_base_path16_rlpm