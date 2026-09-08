from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
import xformers.ops as xops

from .rotary_embedding import apply_rotary_pos_emb, invert_rotatry_pos_emb


def _memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[xops.fmha.AttentionBias],
    dropout_p: float,
) -> torch.Tensor:
    """Use xFormers on CUDA and PyTorch SDPA as the exact CPU fallback."""
    if q.device.type == "cuda":
        return xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            attn_bias=attn_bias,
            p=dropout_p,
            op=xops.MemoryEfficientAttentionFlashAttentionOp,
        )
    # xFormers layout is [B,N,H,D]; scaled_dot_product_attention expects
    # [B,H,N,D]. Both implement softmax(QK^T/sqrt(D))V.
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    if (
        attn_bias is not None
        and not isinstance(attn_bias, torch.Tensor)
        and hasattr(attn_bias, "q_seqinfo")
        and hasattr(attn_bias, "k_seqinfo")
    ):
        # Avoid materializing a potentially huge [N,N] BlockDiagonalMask.
        # The NuCLR spatial attention has B=1 and independent neuron blocks.
        if q_sdpa.shape[0] != 1:
            raise ValueError("CPU BlockDiagonalMask fallback currently requires batch size 1")
        output = torch.empty_like(q_sdpa)
        q_intervals = list(attn_bias.q_seqinfo.intervals())
        k_intervals = list(attn_bias.k_seqinfo.intervals())
        for (q_start, q_end), (k_start, k_end) in zip(q_intervals, k_intervals):
            output[:, :, q_start:q_end] = F.scaled_dot_product_attention(
                q_sdpa[:, :, q_start:q_end],
                k_sdpa[:, :, k_start:k_end],
                v_sdpa[:, :, k_start:k_end],
                dropout_p=dropout_p,
                is_causal=False,
            )
        return output.transpose(1, 2)
    if attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
        raise ValueError(f"Unsupported CPU attention bias: {type(attn_bias)!r}")
    bias = attn_bias
    if isinstance(bias, torch.Tensor) and bias.ndim == 4:
        bias = bias.to(device=q.device, dtype=q.dtype)
    output = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=bias,
        dropout_p=dropout_p,
        is_causal=False,
    )
    return output.transpose(1, 2)


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int = None,
        heads: int = 8,
        dim_head: Optional[int] = None,
        atn_dropout: float = 0.0,
        rotate_value: bool = False,
        to_kv_bias: bool = True,
        to_q_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm_q: bool = True,
        pre_norm_kv: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)

        # build networks
        ctx_dim = ctx_dim or dim
        self.norm = nn.LayerNorm(dim) if pre_norm_q else nn.Identity()
        self.norm_context = nn.LayerNorm(ctx_dim) if pre_norm_kv else nn.Identity()

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=to_q_bias)
        self.to_kv = nn.Linear(ctx_dim, inner_dim * 2, bias=to_kv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x_q: torch.Tensor,
        x_ctx: torch.Tensor,
        rotary_q: Optional[torch.Tensor] = None,
        rotary_ctx: Optional[torch.Tensor] = None,
        attn_bias: Optional[xops.fmha.AttentionBias] = None,
    ):

        q = self.to_q(self.norm(x_q))
        k, v = self.to_kv(self.norm_context(x_ctx)).chunk(2, dim=-1)

        if q.ndim == 2:
            q = rearrange(q, "n (h d) -> 1 n h d", h=self.heads)
            k = rearrange(k, "n (h d) -> 1 n h d", h=self.heads)
            v = rearrange(v, "n (h d) -> 1 n h d", h=self.heads)
            batched_input = False
        elif q.ndim == 3:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
            batched_input = True
        else:
            raise ValueError("Unknown input format")

        # apply rotary embeddings
        if rotary_q is not None:
            assert rotary_ctx is not None
            if not batched_input:
                assert rotary_q.ndim == 2
                assert rotary_ctx.ndim == 2
                rotary_q = rotary_q[None, :, :]
                rotary_ctx = rotary_ctx[None, :, :]

            q = apply_rotary_pos_emb(rotary_q, q)
            k = apply_rotary_pos_emb(rotary_ctx, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_ctx, v)

        # perform attention, by default will use the optimal attention implementation
        out = _memory_efficient_attention(
            q, k, v, attn_bias,
            self.atn_dropout if self.training else 0,
        )

        if rotary_ctx is not None and self.rotate_value:
            out = apply_rotary_pos_emb(invert_rotatry_pos_emb(rotary_q), out)

        # project back to output
        if batched_input:
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = rearrange(out, "b n h d -> (b n) (h d)", b=1)

        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        atn_dropout: float = 0.0,
        rotate_value: bool = False,
        to_qkv_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)

        # build networks
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=to_qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary: Optional[torch.Tensor] = None,
        attn_bias: Optional[xops.fmha.AttentionBias] = None,
    ):

        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        if q.ndim == 2:
            q = rearrange(q, "n (h d) -> 1 n h d", h=self.heads)
            k = rearrange(k, "n (h d) -> 1 n h d", h=self.heads)
            v = rearrange(v, "n (h d) -> 1 n h d", h=self.heads)
            batched_input = False
        elif q.ndim == 3:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
            batched_input = True
        else:
            raise ValueError("Unknown input format")

        # apply rotary embeddings
        if rotary is not None:
            if not batched_input:
                assert rotary.ndim == 2
                rotary = rotary[None, :, :]

            q = apply_rotary_pos_emb(rotary, q)
            k = apply_rotary_pos_emb(rotary, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary, v)

        # perform attention, by default will use the optimal attention implementation
        out = _memory_efficient_attention(
            q, k, v, attn_bias,
            self.atn_dropout if self.training else 0,
        )

        if rotary is not None and self.rotate_value:
            out = apply_rotary_pos_emb(invert_rotatry_pos_emb(rotary), out)

        # project back to output
        if batched_input:
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = rearrange(out, "b n h d -> (b n) (h d)", b=1)

        out = self.to_out(out)
        return out
