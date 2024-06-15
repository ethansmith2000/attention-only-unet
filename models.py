# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

# modified from meta DiT https://github.com/facebookresearch/DiT/tree/main

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AttentionResample(nn.Module):

    def __init__(self, scale_factor, in_dim, out_dim, heads=8, norm=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.h = heads
        # not a very useful operation if using nearest when upscaling, maybe bilinear is a tad more interesting
        self.mode = "nearest" if scale_factor < 1 else "bilinear"
        # self.to_q = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.to_q = nn.Linear(in_dim, out_dim, bias=True)
        self.to_kv = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.to_o = nn.Linear(out_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(in_dim) if norm else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape
        new_l = int(l * self.scale_factor * self.scale_factor)
        h = w = int(l ** 0.5)
        x = self.norm(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        new_d = k.shape[-1]
        q = self.to_q(x).reshape(b, h, w, new_d).permute(0, 3, 1, 2)
        q = torch.nn.functional.interpolate(q, scale_factor=self.scale_factor, mode=self.mode).permute(0, 2, 3, 1).reshape(b, -1, new_d)
        q = q.reshape(b, new_l, self.h, new_d // self.h).permute(0, 2, 1, 3)
        k,v = map(lambda t: t.reshape(b, l, self.h, new_d // self.h).permute(0, 2, 1, 3), (k,v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, new_l, new_d)

        return self.to_o(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=8):
        super().__init__()
        self.h = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_o = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = map(lambda t: t.reshape(b, l, self.h, d // self.h).permute(0, 2, 1, 3), self.to_qkv(x).chunk(3, dim=-1))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, l, d)
        return self.to_o(out)


class SkipAttention(nn.Module):

    def __init__(self, dim, heads=8):
        super().__init__()
        self.h = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_o = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

    def forward(self, x, skip):
        b, l, d = x.shape
        q = self.to_q(self.q_norm(x))
        k, v = self.to_kv(self.kv_norm(skip)).chunk(2, dim=-1)
        q,k,v = map(lambda t: t.reshape(b, l, self.h, d // self.h).permute(0, 2, 1, 3), (q,k,v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, l, d)
        return x + self.to_o(out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, c_dim, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Level(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    def forward(self, x, c):
        if self.gradient_checkpointing:
            for layer in self.layers:
                x = layer(x, c)
        else:
            for layer in self.layers:
                x = layer(x, c)
        return x


class UDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=1,
        channels=3,
        hidden_size=[256, 512, 512, 1024],
        layers_per_level=[2, 2, 2, 2],
        num_heads=[4, 4, 4, 4],
        mid_block_layers=2,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=None,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        inp_dim = hidden_size[0]
        self.x_embedder = PatchEmbed(input_size, patch_size, channels, inp_dim, bias=True)
        self.t_embedder = TimestepEmbedder(inp_dim)
        self.y_embedder = LabelEmbedder(num_classes, inp_dim, class_dropout_prob) if num_classes is not None else None

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, inp_dim), requires_grad=False)
        self.downs = nn.ModuleList([])
        self.downsamplers = nn.ModuleList([])
        self.ups = []
        self.upsamplers = []
        self.skip_connects = []
        for i, (dim, heads) in enumerate(zip(hidden_size, num_heads)):
            next_dim = hidden_size[min(i + 1, len(hidden_size) - 1)]
            next_head = num_heads[min(i + 1, len(num_heads) - 1)]
            layers = [
                DiTBlock(dim, heads, c_dim=inp_dim, mlp_ratio=mlp_ratio) for _ in range(layers_per_level[i])
            ]
            self.downs.append(Level(layers))
            self.downsamplers.append(AttentionResample(0.5, dim, next_dim, next_head))
            self.ups.append(Level([DiTBlock(dim, heads, c_dim=inp_dim, mlp_ratio=mlp_ratio) for _ in range(mid_block_layers)]))
            self.upsamplers.append(AttentionResample(2, next_dim, dim, heads))
            self.skip_connects.append(SkipAttention(dim, heads))

        self.ups = nn.ModuleList(reversed(self.ups))
        self.upsamplers = nn.ModuleList(reversed(self.upsamplers))
        self.skip_connects = nn.ModuleList(reversed(self.skip_connects))

        self.mid_block = Level([DiTBlock(hidden_size[-1], num_heads[-1], c_dim=inp_dim, mlp_ratio=mlp_ratio) for _ in range(mid_block_layers)])
        self.final_layer = FinalLayer(hidden_size[0], patch_size, self.channels)
        self.initialize_weights()

    def enable_gradient_checkpointing(self):
        for down, up in zip(self.downs, self.ups):
            down.gradient_checkpointing = True
            up.gradient_checkpointing = True
        self.mid_block.gradient_checkpointing = True

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for n,m in self.named_modules():
            if isinstance(m, DiTBlock):
                nn.init.constant_(m.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(m.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)        
        c = t
        if self.y_embedder is not None:
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)

        skips = []
        for i, (level, downsample) in enumerate(zip(self.downs, self.downsamplers)):
            x = level(x, c)
            # print(i, x.shape, "block")
            skips.append(x)
            x = downsample(x)
            # print(i, x.shape, "downsample")
        
        # print(x.shape, "mid block before")
        x = self.mid_block(x, c)
        # print(x.shape, "mid block after")

        for i, (level, upsample, skip_connect) in enumerate(zip(self.ups, self.upsamplers, self.skip_connects)):
            x = upsample(x)
            # print(i, x.shape, "upsample")
            x = skip_connect(x, skips.pop())
            # print(i, x.shape, "skip connect")
            x = level(x, c)
            # print(i, x.shape, "block")

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        eps = self.forward(torch.cat([x, x], dim=0), t, y)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def UDiT_XL(**kwargs):
    return UDiT(
                layers_per_level=[2, 3, 3, 4],
                mid_block_layers=4,
                hidden_size=[512, 768, 1024, 1536], 
                num_heads=[8, 12, 16, 24],
                **kwargs)

def UDiT_L(**kwargs):
    return UDiT(
                layers_per_level=[2, 2, 3, 3],
                mid_block_layers=3,
                hidden_size=[384, 512, 768, 1024], 
                num_heads=[6, 8, 12, 16],
                **kwargs)

def UDiT_B(**kwargs):
    return UDiT(
                layers_per_level=[2, 2, 2, 3],
                mid_block_layers=3,
                hidden_size=[256, 512, 512, 1024], 
                num_heads=[4, 8, 8, 16],
                **kwargs)

def UDiT_S(**kwargs):
    return UDiT(
                layers_per_level=[1, 1, 1, 1],
                mid_block_layers=2,
                hidden_size=[256, 384, 384, 512],
                num_heads=[4, 6, 6, 8],
                **kwargs)


