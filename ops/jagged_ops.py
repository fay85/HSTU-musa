"""
Pure PyTorch jagged tensor operations.
Replaces FBGEMM jagged ops for MUSA and other non-MUSA backends.
"""

import torch
from typing import List


def asynchronous_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [lengths.new_zeros(1, dtype=lengths.dtype), torch.cumsum(lengths, dim=0)],
        dim=0,
    )


def asynchronous_inclusive_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(lengths, dim=0)


def asynchronous_exclusive_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    cum = torch.cumsum(lengths, dim=0)
    return torch.cat([lengths.new_zeros(1, dtype=lengths.dtype), cum[:-1]], dim=0)


def length_to_complete_offsets(lengths: torch.Tensor) -> torch.Tensor:
    return asynchronous_complete_cumsum(lengths)


def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Convert jagged tensor to padded dense tensor.
    Equivalent to torch.ops.fbgemm.jagged_to_padded_dense."""
    assert len(offsets) == 1, "Only single jagged dimension supported"
    off = offsets[0]
    batch_size = off.size(0) - 1
    max_len = max_lengths[0]

    if values.dim() == 1:
        values = values.unsqueeze(-1)
    feat_dim = values.size(-1)

    output = torch.full(
        (batch_size, max_len, feat_dim),
        padding_value,
        device=values.device,
        dtype=values.dtype,
    )

    for b in range(batch_size):
        s = off[b].item()
        e = off[b + 1].item()
        n = min(e - s, max_len)
        if n > 0:
            output[b, :n, :] = values[s : s + n, :]

    if feat_dim == 1 and values.dim() == 2 and values.size(-1) == 1:
        pass  # keep 3D for consistency with caller
    return output


def dense_to_jagged(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_length: int = -1,
) -> List[torch.Tensor]:
    """Convert padded dense tensor back to jagged format.
    Equivalent to torch.ops.fbgemm.dense_to_jagged."""
    assert len(offsets) == 1
    off = offsets[0]
    batch_size = off.size(0) - 1
    total = off[-1].item() if total_length < 0 else total_length

    if dense.dim() == 2:
        dense = dense.unsqueeze(-1)
    feat_dim = dense.size(2)

    output = torch.zeros(total, feat_dim, device=dense.device, dtype=dense.dtype)

    for b in range(batch_size):
        s = off[b].item()
        e = off[b + 1].item()
        n = e - s
        if n > 0:
            output[s:e, :] = dense[b, :n, :]

    if feat_dim == 1:
        output = output.squeeze(-1)
    return [output]
