# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn
import torch.nn.functional as F

from xformers.ops import memory_efficient_attention, unbind, fmha

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        if q.device.type == 'cuda' and attn_bias is None:
            try:
                x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            except Exception as e:
                print(f"xformers memory_efficient_attention failed: {e}, falling back to PyTorch native.")
                q = q.permute(0, 2, 1, 3) # B, H, S, D
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
                x = x.permute(0, 2, 1, 3) # B, S, H, D
        else:
            if attn_bias is not None:
                print(f"Warning: Using PyTorch native attention. Complex attn_bias type {type(attn_bias)} detected.")
                try:
                    attn_mask = None 
                    print("Warning: Could not materialize attn_bias for PyTorch native attention. Ignoring bias.")
                except Exception as bias_e:
                    print(f"Warning: Failed to materialize attn_bias ({bias_e}). Ignoring bias.")
                    attn_mask = None
            else:
                attn_mask = None 

            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            x = x.permute(0, 2, 1, 3)

        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x