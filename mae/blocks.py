"""
Ref: https://github.com/naver/croco/blob/master/models/blocks.py#L32
"""

from hpack import Encoder
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
from visualizer import get_local
from einops import rearrange

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    @get_local('attn')
    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class CrossAttentionVarydim(nn.Module):

    def __init__(self, input_dim, output_dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = output_dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(output_dim, output_dim, bias=qkv_bias)
        self.projk = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.projv = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(output_dim, output_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    @get_local('attn')
    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderCABlockLang(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_img = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.cross_attn_lang = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y=None, lang=None):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # cross attention between current (K,V) ang goal image (Q)
        if y is not None:
            y = self.norm_y(y)
            x = x + self.drop_path(self.cross_attn_img(y, self.norm2(x), self.norm2(x)))

        # cross attention between current (Q) and language (K,V)
        if lang is not None:
            lang = lang[0] if type(lang) is tuple else lang
            x = x + self.drop_path(self.cross_attn_lang(self.norm3(x), lang, lang))

        # final output
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        return x, y


class DecoderCABlockLangNoRef(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_lang = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y=None, lang=None):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # cross attention between current (Q) and language (K,V)
        if lang is not None:
            lang = lang[0] if type(lang) is tuple else lang
            x = x + self.drop_path(self.cross_attn_lang(self.norm3(x), lang, lang))
        # final output
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        return x, y


class DecoderCABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                         proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        y = self.norm_y(y)

        # cross attention between current (K,V) ang goal image (Q)
        x = x + self.drop_path(self.cross_attn(y, self.norm2(x), self.norm2(x)))

        # final output
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class DecoderCABlockVisionLang(nn.Module):
    """Deocder with cross attention between clip vision and clip language"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_img = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.cross_attn_ref = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)
        self.cross_attn_clip = CrossAttentionVarydim(768, dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x, y=None, img=None, lang=None):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # cross attention between current (K,V) ang goal image (Q)
        if y is not None:
            y = self.norm_y(y)
            x = x + self.drop_path(self.cross_attn_img(y, self.norm2(x), self.norm2(x)))

        # cross attention between text and image
        reference = self.cross_attn_clip(lang, img, img) # lang (b,d), img (b,t,d)
        x = x + self.drop_path(self.cross_attn_ref(self.norm3(x), reference, reference))

        # final output
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        return x, y


class DecoderCABlockVisionLangMul(nn.Module):
    """Deocder with mutliply between clip vision and clip language"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_img = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.cross_attn_ref = CrossAttentionVarydim(128, dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x, y=None, ref=None):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # cross attention between current (K,V) ang goal image (Q)
        if y is not None:
            y = self.norm_y(y)
            x = x + self.drop_path(self.cross_attn_img(y, self.norm2(x), self.norm2(x)))

        # cross attention between text and image
        # b,c,h,w -> b, 
        reference = rearrange(ref, 'b c h w -> b (h w) c')
        x = x + self.drop_path(self.cross_attn_ref(self.norm3(x), reference, reference))

        # final output
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        return x, y


class DecoderCLIPBlock(nn.Module):
    "Decocer of CLIP vision and CLIP text"
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_clip = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)       

    def forward(self, x, lang):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # cross attention between text and image
        if lang is not None:
            lang = lang[0] if type(lang) is tuple else lang
            x = x + self.drop_path(self.cross_attn_clip(self.norm2(x), lang, lang))

        # final output
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x 


class DecoderCABlockLangReverse(DecoderCABlockLang):
    """
    Reverse the order of cross attention
    Masked image -> query, lang and input -> key, value
    """
    def forward(self, x, y=None, lang=None):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # cross attention between current (K,V) ang goal image (Q)
        if y is not None:
            y = self.norm_y(y)
            x = self.norm2(x)
            x = x + self.drop_path(self.cross_attn_img(x, y, y))

        # cross attention between current (Q) and language (K,V)
        if lang is not None:
            lang = lang[0] if type(lang) is tuple else lang
            x = x + self.drop_path(self.cross_attn_lang(self.norm3(x), lang, lang))

        # final output
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        return x, y


class DecoderCABlockLangReverse2(DecoderCABlockLang):
    """
    Reverse the order of cross attention
    first cross attention between text and masked image,
    then with masked image
    """
    def forward(self, x, y=None, lang=None):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # cross attention between current (Q) and language (K,V)
        if lang is not None:
            lang = lang[0] if type(lang) is tuple else lang
            x = x + self.drop_path(self.cross_attn_lang(self.norm3(x), lang, lang))
        
        # cross attention between current (K,V) ang goal image (Q)
        if y is not None:
            y = self.norm_y(y)
            x = x + self.drop_path(self.cross_attn_img(y, self.norm2(x), self.norm2(x)))

        # final output
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        return x, y


class EncoderCABlockVision(nn.Module):
    """Encoder + CLIP Vision"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=None, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, vision=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        if vision is not None:
            x = x + self.drop_path(self.cross_attn(self.norm2(x), vision, vision))
        
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class EncoderCABlockLang(nn.Module):
    """Encoder + CLIP Language"""

    def __init__(self, lang_dim, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttentionVarydim(input_dim=lang_dim, output_dim=dim,rope=None, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, lang=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.cross_attn(self.norm2(x), lang, lang))
        
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class DecoderDERTBlockLang(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_img1 = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.cross_attn_img2 = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
        self.cross_attn_lang = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm_x = norm_layer(dim)
        self.norm_y = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, x, y, lang):
        """
        x: norm -> self_attn -> drop -> norm -> cross_attn -> drop
        """
        query = query + self.drop_path(self.attn(self.norm1(query)))

        # cross attention between query （Q） and x (K,V)
        query = query + self.drop_path(self.cross_attn_img1(self.norm2(query), self.norm_x(x), self.norm_x(x)))

        # cross attention between current (K,V) ang goal image (Q)
        query = query + self.drop_path(self.cross_attn_img2(self.norm3(query), self.norm_y(y), self.norm_y(y)))

        # cross attention between current (Q) and language (K,V)
        lang = lang[0] if isinstance(lang, tuple) else lang
        query = query + self.drop_path(self.cross_attn_lang(self.norm4(query), lang, lang))

        # final output
        query = query + self.drop_path(self.mlp(self.norm5(query)))
        return query