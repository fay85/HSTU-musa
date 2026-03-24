"""
HSTU Attention implementations for MUSA / non-MUSA backends.

The PyTorch implementation is ported directly from recsys-examples
ops/pt_ops/pt_hstu_attention.py to ensure mathematical equivalence.
"""

import abc
from typing import Optional, Union

import torch
import torch.nn.functional as F

from ops.jagged_ops import jagged_to_padded_dense, dense_to_jagged


class HSTUAttention(torch.nn.Module):
    @abc.abstractmethod
    def forward(self, tq, tk, tv, offsets, max_seqlen, scaling_seqlen,
                target_group_size=1, num_candidates=None, num_contextuals=None):
        pass


def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    num_contextuals: Union[int, torch.Tensor] = 0,
    min_full_attn_seq_len: int = 0,
    target_group_size: int = 1,
):
    """Exact port of _get_valid_attn_mask from recsys-examples."""
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)

    if isinstance(num_contextuals, int):
        if num_contextuals > 0:
            ids = ids - num_contextuals + 1
            ids = torch.clamp(ids, min=0)
            max_ids = max_ids - num_contextuals + 1
    else:
        ids = ids - num_contextuals.view(-1, 1) + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - num_contextuals.view(-1, 1, 1) + 1

    row_ids = ids.unsqueeze(-1).expand(-1, N, N)
    col_ids = row_ids.transpose(1, 2)
    row_col_dist = row_ids - col_ids

    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)

    if num_targets is not None:
        target_group_row_ids = torch.clamp(
            row_ids - max_ids + num_targets.view(-1, 1, 1), min=-1
        ) // target_group_size
        target_group_col_ids = target_group_row_ids.transpose(1, 2)
        target_dist = target_group_row_ids - target_group_col_ids

        target_group_mask = torch.logical_or(
            target_dist == 0,
            (target_group_row_ids < 0) + (target_group_col_ids < 0),
        )
        valid_attn_mask = torch.logical_and(valid_attn_mask, target_group_mask)
        max_ids = max_ids - num_targets.view(-1, 1, 1)

    if max_attn_len is not None and max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )

    if (isinstance(num_contextuals, int) and num_contextuals > 0) or isinstance(
        num_contextuals, torch.Tensor
    ):
        valid_attn_mask = torch.logical_or(
            valid_attn_mask,
            torch.logical_and(row_ids == 0, col_ids < max_ids),
        )

    return valid_attn_mask


def _pad_qkv(q, k, v, seq_offsets, N):
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )
    padded_k = (
        jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )
    padded_v = (
        jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )
    return padded_q, padded_k, padded_v


def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    max_attn_len: Optional[int] = None,
    target_group_size: int = 1,
    scaling_seqlen: int = -1,
) -> torch.Tensor:
    """Exact port of pytorch_hstu_mha from recsys-examples."""
    if num_contextuals is None:
        num_contextuals = 0
    if scaling_seqlen == -1:
        scaling_seqlen = max_seq_len

    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = _pad_qkv(q, k, v, seq_offsets, max_seq_len)

    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / scaling_seqlen

    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        num_contextuals=num_contextuals,
        target_group_size=target_group_size,
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)

    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)

    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)
    return dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),
        [seq_offsets],
        total_length=L,
    )[0].view(L, H, V)


class PyTorchHSTUAttention(HSTUAttention):
    """
    Pure PyTorch HSTU attention -- exact port from recsys-examples.
    Handles contextual tokens, candidate masking, target groups, and max_attn_len.
    """

    def __init__(self, num_heads, attention_dim, linear_dim, is_causal):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal

    def forward(self, tq, tk, tv, offsets, max_seqlen, scaling_seqlen=-1,
                target_group_size=1, num_candidates=None, num_contextuals=None):
        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )

        return pytorch_hstu_mha(
            max_seq_len=max_seqlen,
            alpha=1.0 / (self.attention_dim ** 0.5),
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_contextuals=num_contextuals,
            num_targets=num_candidates,
            causal=self.is_causal,
            dropout_pr=0.0,
            training=self.training,
            target_group_size=target_group_size,
            scaling_seqlen=scaling_seqlen,
        ).view(-1, self.num_heads * self.linear_dim)


class TritonHSTUAttention(HSTUAttention):
    """Triton-based HSTU attention. Requires Triton to be installed."""

    def __init__(self, num_heads, attention_dim, linear_dim, is_causal):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        assert is_causal, "TritonHSTUAttention only supports causal mode"

    def forward(self, tq, tk, tv, offsets, max_seqlen, scaling_seqlen=-1,
                target_group_size=1, num_candidates=None, num_contextuals=None):
        try:
            from ops.triton_hstu_attention import triton_hstu_mha
        except ImportError:
            raise RuntimeError(
                "Triton HSTU attention not available. Use pytorch backend."
            )

        if num_contextuals is None:
            num_contextuals = 0

        return triton_hstu_mha(
            N=max_seqlen,
            alpha=1.0 / (self.attention_dim ** 0.5),
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_targets=num_candidates,
            contextual_seq_len=num_contextuals,
            scaling_seqlen=scaling_seqlen,
            enable_tma=False,
        ).view(-1, self.num_heads * self.linear_dim)


def create_hstu_attention(kernel_backend, num_heads, attention_dim, linear_dim, is_causal):
    if kernel_backend == "triton" and is_causal:
        try:
            import triton
            return TritonHSTUAttention(num_heads, attention_dim, linear_dim, is_causal)
        except ImportError:
            pass
    return PyTorchHSTUAttention(num_heads, attention_dim, linear_dim, is_causal)
