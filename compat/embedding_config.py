"""Pure-Python reimplementation of torchrec EmbeddingConfig and DataType.

No torchrec or fbgemm dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import torch
from torch import Tensor


class DataType(Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


_TORCH_DTYPE_TO_DATA_TYPE = {
    torch.float32: DataType.FP32,
    torch.float16: DataType.FP16,
    torch.bfloat16: DataType.BF16,
    torch.int8: DataType.INT8,
}


def dtype_to_data_type(dtype: torch.dtype) -> DataType:
    """Map a torch.dtype to the corresponding DataType enum value.

    Falls back to FP32 for unmapped dtypes.
    """
    return _TORCH_DTYPE_TO_DATA_TYPE.get(dtype, DataType.FP32)


@dataclass
class EmbeddingConfig:
    name: str
    embedding_dim: int
    num_embeddings: int
    feature_names: List[str] = field(default_factory=list)
    data_type: DataType = DataType.FP32
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    init_fn: Optional[Callable[[Tensor], None]] = None

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = [self.name]
