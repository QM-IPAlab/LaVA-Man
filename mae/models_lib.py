"""
Choose which model to load
"""
from models_mae_robot import MAERobotBase, MAERobot
from models_mae_robot_lang import MAERobotLang, MAERobotLangNoRef, MAERobotLang2, MAERobotLangRecon
from models_mae_robot_lang import MAERobotLangDualMasking, MAERobotLangReverse, MAERobotLangCF, MAERobotLangReverse2, MAERobotLangNodec
from models_mae_robot_lang_vision import MAERobotLangVisonE, MAERobotLangVisonProjector, MAERobotLangVisonProMul, MAERobotLangVisonProMulCat
from models_mae_robot_lang_vision2 import MAERobotLangVisonCLIP, MAERobotLangVisonCLIPRes, MAECLIP, MAECLIPPE
from models_mae_robot_lang_relevance import MAERobotLangRel
from models_mae_robot_cliploss import MAERobotLangCLIPLoss
from models_mae_robot_lang_jepa import JEPARobotLang, JEPARobotLang2loss
from voltron_instantiate import voltron_vcond
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

def mae_vit_base_path16_rl2_enc_only(**kwargs):
    """MAE with language in both encoder and decoder"""
    model = MAERobotLangNodec(
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

def vit_base_patch16_clip_only(**kwargs):
    model = MAERobotLangVisonCLIP(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_mae_clip(**kwargs):
    model = MAECLIP(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_mae_clip_pe(**kwargs):
    model = MAECLIPPE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_clip_res_only(**kwargs):
    model = MAERobotLangVisonCLIPRes(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_path16_rlpmc(**kwargs):
    """MAE with multiplication between clip text and clip vision in deocder"""
    model = MAERobotLangVisonProMulCat(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rl_cliploss(**kwargs):
    "MAE with CLIP vision model reconstruction loss"
    model = MAERobotLangCLIPLoss(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rl_recon(**kwargs):
    """MAE robot lang reconsturction version (single input)"""
    model = MAERobotLangRecon(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rl_dm(**kwargs):
    """MAE robot lang dual masking version (mask two inputs) """
    model = MAERobotLangDualMasking(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rl_rev(**kwargs):
    model = MAERobotLangReverse(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_rl_rev2(**kwargs):
    model = MAERobotLangReverse2(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def jepa_vit_base_patch16_rl(**kwargs):
    model = JEPARobotLang(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def jepa_vit_base_patch16_rl_2loss(**kwargs):
    model = JEPARobotLang2loss(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vcond(**kwargs):
    model = voltron_vcond()
    return model

def mae_vit_base_patch16_rlcf(**kwargs):
    model = MAERobotLangCF(
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
mae_robot_lang2 = mae_vit_base_path16_rl2  # encoder in language
mae_robot_lang2_enc_only = mae_vit_base_path16_rl2_enc_only  # encoder in language
mae_robot_projector = mae_vit_base_path16_rlp
mae_robot_promul = mae_vit_base_path16_rlpm
robot_clip = vit_base_patch16_clip_only
robot_clip_res = vit_base_patch16_clip_res_only
mae_robot_promulcat = mae_vit_base_path16_rlpmc
mae_robot_cliploss = mae_vit_base_patch16_rl_cliploss
mae_robot_recon = mae_vit_base_patch16_rl_recon
mae_robot_dm = mae_vit_base_patch16_rl_dm
mae_robot_lang_rev = mae_vit_base_patch16_rl_rev
jepa_robot_lang = jepa_vit_base_patch16_rl
jepa_2loss = jepa_vit_base_patch16_rl_2loss
voltron = vcond
mae_robot_lang_cf = mae_vit_base_patch16_rlcf
mae_clip = vit_base_patch16_mae_clip
mae_clip_pe = vit_base_patch16_mae_clip_pe
mae_robot_lang_rev2 = mae_vit_base_patch16_rl_rev2