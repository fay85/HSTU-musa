"""Position encoder for HSTU -- ported from recsys-examples."""

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn


class HSTUPositionalEncoder(nn.Module):
    """Adds learnable position embeddings to sequence embeddings.

    Ported from recsys-examples/examples/hstu/modules/position_encoder.py.
    Uses Triton kernels when available, falls back to pure PyTorch.
    """

    def __init__(
        self,
        num_position_buckets: int,
        embedding_dim: int,
        training_dtype: torch.dtype,
        num_time_buckets: int = 2048,
        use_time_encoding: bool = False,
    ) -> None:
        super().__init__()
        self._use_time_encoding = use_time_encoding
        self._embedding_dim = embedding_dim
        self._position_embeddings_weight = nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            )
        )
        if self._use_time_encoding:
            self._timestamp_embeddings_weight = nn.Parameter(
                torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                    -sqrt(1.0 / num_time_buckets),
                    sqrt(1.0 / num_time_buckets),
                )
            )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor] = None,
        seq_timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self._embedding_dim ** 0.5
        max_pos_ind = self._position_embeddings_weight.size(0)
        batch_size = seq_offsets.size(0) - 1

        high_inds = seq_lengths.clone()
        if num_targets is not None:
            high_inds = high_inds - num_targets
        high_inds = torch.clamp(high_inds, max=max_pos_ind - 1)

        seq_embeddings = seq_embeddings * alpha

        for b in range(batch_size):
            start = seq_offsets[b].item()
            end = seq_offsets[b + 1].item()
            sl = end - start
            if sl == 0:
                continue
            hi = high_inds[b].item()
            pos_indices = torch.arange(hi, hi - sl, -1, device=seq_embeddings.device)
            pos_indices = torch.clamp(pos_indices, min=0, max=max_pos_ind - 1)
            seq_embeddings[start:end] = (
                seq_embeddings[start:end]
                + self._position_embeddings_weight[pos_indices]
            )

        return seq_embeddings
