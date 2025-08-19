from functools import partial

import torch
import torch.nn as nn

from networks.swinT_components import Mlp


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
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
        x_1, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class SpatialGateUnit(nn.Module):
    def __init__(self, lengthx, dim, attn_drop=0.):
        super(SpatialGateUnit, self).__init__()
        mlp_ratio = 4.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=int(dim // 2))
        self.proj_back = nn.Conv1d(lengthx, lengthx, 1)
        self.act = nn.Sigmoid()
        self.norm = nn.LayerNorm(1)
        self.scale = dim ** -0.5
        self.gate_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.proj(x)
        x_avg = self.avg_pool(x)

        rsp_map = (x @ x.transpose(-1, -2)) * self.scale
        rsp_map = self.gate_drop(rsp_map.softmax(dim=-1))

        rsp = self.norm(self.proj_back(rsp_map @ x_avg)).squeeze(-1)
        rsp = self.act(rsp.float()).unsqueeze(-1)

        return rsp


class GCAM(nn.Module):
    def __init__(
            self,
            x_length: int,
            embedding_dim: int,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()

        self.norm_final_attn_q = nn.LayerNorm(embedding_dim)

        self.spatial_gate = SpatialGateUnit(x_length, embedding_dim, attn_drop=0.)

        self.norm_spatial_gate1 = nn.LayerNorm(embedding_dim)

        self.proj_back_q = nn.Linear(2 * embedding_dim, embedding_dim, bias=True)

    def get_spatial_weights(self, x):
        return self.spatial_gate(x)

    def apply_gate(self, x, spatial_weights):
        return self.norm_spatial_gate1(x * spatial_weights)

    def forward(
            self,
            queries,
    ):
        return self.norm_final_attn_q(
            self.proj_back_q(torch.cat((
                queries,
                self.apply_gate(
                    queries,
                    self.get_spatial_weights(queries)
                )), dim=-1))
        )


class SkipGatedFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(SkipGatedFusion, self).__init__()
        self.fusion1 = nn.Linear(2 * embedding_dim, embedding_dim)

        self.proj = nn.Linear(embedding_dim, 1)
        self.act = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x1 = self.norm1(x * self.act(self.proj(x)))

        return self.norm2(self.fusion1(torch.cat([x, x1], -1)))
