"""TorchRec compatibility shim -- pure PyTorch, no fbgemm/torchrec."""

from compat.embedding_config import DataType, EmbeddingConfig, dtype_to_data_type
from compat.jagged_tensor import JaggedTensor, KeyedJaggedTensor

__all__ = [
    "DataType",
    "EmbeddingConfig",
    "JaggedTensor",
    "KeyedJaggedTensor",
    "dtype_to_data_type",
]
