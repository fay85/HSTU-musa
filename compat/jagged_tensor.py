"""Pure-PyTorch reimplementation of torchrec JaggedTensor / KeyedJaggedTensor.

These are lightweight data containers -- no FBGEMM or torchrec dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor


class JaggedTensor:
    """Variable-length 1-D tensor described by (values, lengths/offsets)."""

    def __init__(
        self,
        values: Tensor,
        lengths: Optional[Tensor] = None,
        offsets: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> None:
        if lengths is None and offsets is None:
            raise ValueError("At least one of lengths or offsets must be provided")
        self._values = values
        self._lengths = lengths
        self._offsets = offsets
        self._weights = weights

    def values(self) -> Tensor:
        return self._values

    def lengths(self) -> Tensor:
        if self._lengths is None:
            # offsets is guaranteed non-None here by __init__ check
            self._lengths = self._offsets[1:] - self._offsets[:-1]
        return self._lengths

    def offsets(self) -> Tensor:
        if self._offsets is None:
            self._offsets = torch.zeros(
                self._lengths.size(0) + 1,
                dtype=self._lengths.dtype,
                device=self._lengths.device,
            )
            torch.cumsum(self._lengths, dim=0, out=self._offsets[1:])
        return self._offsets

    def weights(self) -> Optional[Tensor]:
        return self._weights

    def to(self, device: torch.device, non_blocking: bool = False) -> "JaggedTensor":
        return JaggedTensor(
            values=self._values.to(device, non_blocking=non_blocking),
            lengths=(
                self._lengths.to(device, non_blocking=non_blocking)
                if self._lengths is not None
                else None
            ),
            offsets=(
                self._offsets.to(device, non_blocking=non_blocking)
                if self._offsets is not None
                else None
            ),
            weights=(
                self._weights.to(device, non_blocking=non_blocking)
                if self._weights is not None
                else None
            ),
        )

    def __repr__(self) -> str:
        return (
            f"JaggedTensor(values={self._values}, "
            f"lengths={self.lengths()}, "
            f"weights={self._weights})"
        )


class KeyedJaggedTensor:
    """Multiple named JaggedTensors packed into a single flat values tensor.

    Layout convention (matches torchrec):
      * ``lengths`` has shape ``[num_keys * batch_size]``.
        The first ``batch_size`` entries belong to ``keys[0]``, the next
        ``batch_size`` to ``keys[1]``, etc.
      * ``values`` is the flat concatenation of all features in that order.
    """

    def __init__(
        self,
        keys: List[str],
        values: Tensor,
        lengths: Optional[Tensor] = None,
        offsets: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> None:
        if lengths is None and offsets is None:
            raise ValueError("At least one of lengths or offsets must be provided")
        self._keys = keys
        self._values = values
        self._lengths = lengths
        self._offsets = offsets
        self._weights = weights

    def keys(self) -> List[str]:
        return self._keys

    def values(self) -> Tensor:
        return self._values

    def lengths(self) -> Tensor:
        if self._lengths is None:
            self._lengths = self._offsets[1:] - self._offsets[:-1]
        return self._lengths

    def offsets(self) -> Tensor:
        if self._offsets is None:
            self._offsets = torch.zeros(
                self._lengths.size(0) + 1,
                dtype=self._lengths.dtype,
                device=self._lengths.device,
            )
            torch.cumsum(self._lengths, dim=0, out=self._offsets[1:])
        return self._offsets

    def weights(self) -> Optional[Tensor]:
        return self._weights

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, JaggedTensor]:
        num_keys = len(self._keys)
        lengths = self.lengths()
        total_len = lengths.size(0)
        if num_keys == 0:
            return {}
        batch_size = total_len // num_keys

        offsets = self.offsets()
        result: Dict[str, JaggedTensor] = {}
        for i, key in enumerate(self._keys):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            key_lengths = lengths[start_idx:end_idx]
            # offsets index: first value offset for this key's rows
            val_start = offsets[start_idx].item()
            val_end = offsets[end_idx].item()
            key_values = self._values[val_start:val_end]
            key_weights = (
                self._weights[val_start:val_end]
                if self._weights is not None
                else None
            )
            result[key] = JaggedTensor(
                values=key_values,
                lengths=key_lengths,
                weights=key_weights,
            )
        return result

    def split(self, split_sizes: List[int]) -> List["KeyedJaggedTensor"]:
        """Split along the key dimension.

        ``split_sizes[i]`` is the number of keys in the i-th output KJT.
        """
        if sum(split_sizes) != len(self._keys):
            raise ValueError(
                f"split_sizes sum {sum(split_sizes)} != num keys {len(self._keys)}"
            )

        lengths = self.lengths()
        num_keys = len(self._keys)
        if num_keys == 0:
            return [
                KeyedJaggedTensor(
                    keys=[], values=self._values[:0], lengths=self._values.new_empty(0, dtype=torch.long)
                )
                for _ in split_sizes
            ]
        batch_size = lengths.size(0) // num_keys
        offsets = self.offsets()

        results: List[KeyedJaggedTensor] = []
        key_cursor = 0
        for sz in split_sizes:
            sub_keys = self._keys[key_cursor : key_cursor + sz]
            len_start = key_cursor * batch_size
            len_end = (key_cursor + sz) * batch_size
            sub_lengths = lengths[len_start:len_end]

            val_start = offsets[len_start].item()
            val_end = offsets[len_end].item()
            sub_values = self._values[val_start:val_end]
            sub_weights = (
                self._weights[val_start:val_end]
                if self._weights is not None
                else None
            )
            results.append(
                KeyedJaggedTensor(
                    keys=sub_keys,
                    values=sub_values,
                    lengths=sub_lengths,
                    weights=sub_weights,
                )
            )
            key_cursor += sz
        return results

    def permute(
        self,
        order: List[int],
        order_tensor: Optional[Tensor] = None,
    ) -> "KeyedJaggedTensor":
        """Reorder keys according to ``order`` (index list)."""
        d = self.to_dict()
        new_keys = [self._keys[i] for i in order]
        jts = [d[k] for k in new_keys]

        new_values = torch.cat([jt.values() for jt in jts], dim=0)
        new_lengths = torch.cat([jt.lengths() for jt in jts], dim=0)
        new_weights: Optional[Tensor] = None
        if self._weights is not None:
            new_weights = torch.cat(
                [jt.weights() for jt in jts if jt.weights() is not None], dim=0
            )
        return KeyedJaggedTensor(
            keys=new_keys,
            values=new_values,
            lengths=new_lengths,
            weights=new_weights,
        )

    @classmethod
    def concat(cls, kjts: List["KeyedJaggedTensor"]) -> "KeyedJaggedTensor":
        """Concatenate multiple KJTs along the key dimension."""
        if not kjts:
            raise ValueError("Cannot concat empty list")
        all_keys: List[str] = []
        val_parts: List[Tensor] = []
        len_parts: List[Tensor] = []
        wt_parts: List[Tensor] = []
        has_weights = kjts[0]._weights is not None

        for kjt in kjts:
            all_keys.extend(kjt._keys)
            val_parts.append(kjt._values)
            len_parts.append(kjt.lengths())
            if has_weights and kjt._weights is not None:
                wt_parts.append(kjt._weights)

        return cls(
            keys=all_keys,
            values=torch.cat(val_parts, dim=0),
            lengths=torch.cat(len_parts, dim=0),
            weights=torch.cat(wt_parts, dim=0) if has_weights else None,
        )

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            lengths=(
                self._lengths.to(device, non_blocking=non_blocking)
                if self._lengths is not None
                else None
            ),
            offsets=(
                self._offsets.to(device, non_blocking=non_blocking)
                if self._offsets is not None
                else None
            ),
            weights=(
                self._weights.to(device, non_blocking=non_blocking)
                if self._weights is not None
                else None
            ),
        )

    def __repr__(self) -> str:
        return (
            f"KeyedJaggedTensor(keys={self._keys}, "
            f"values={self._values}, "
            f"lengths={self.lengths()})"
        )
