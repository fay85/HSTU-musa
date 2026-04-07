from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ShardedEmbeddingConfig:
    feature_names: List[str]
    table_name: str
    vocab_size: int
    dim: int
    sharding_type: str = "data_parallel"


@dataclass
class PositionEncodingConfig:
    num_position_buckets: int = 8192
    num_time_buckets: int = 2048
    use_time_encoding: bool = False


@dataclass
class HSTUConfig:
    hidden_size: int = 128
    kv_channels: int = 32
    num_attention_heads: int = 4
    num_layers: int = 2
    hidden_dropout: float = 0.2
    layernorm_epsilon: float = 1e-5
    is_causal: bool = True
    residual: bool = True
    bf16: bool = False
    fp16: bool = False
    kernel_backend: str = "pytorch"
    learnable_input_layernorm: bool = True
    learnable_output_layernorm: bool = True
    add_uvqk_bias: bool = True
    target_group_size: int = 1
    scaling_seqlen: int = -1
    sequence_parallel: bool = False
    hstu_preprocessing_config: Optional[object] = None
    position_encoding_config: Optional[PositionEncodingConfig] = None
    recompute_input_layernorm: bool = False
    recompute_input_silu: bool = False
    fuse_norm_mul_dropout: bool = False


@dataclass
class RankingConfig:
    embedding_configs: List[ShardedEmbeddingConfig] = field(default_factory=list)
    prediction_head_arch: List[int] = field(default_factory=lambda: [64, 1])
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: List[str] = field(default_factory=lambda: ["auc"])
