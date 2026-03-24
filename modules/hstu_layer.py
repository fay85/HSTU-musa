import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.jagged_data import JaggedData
from modules.hstu_attention import create_hstu_attention


class HSTULayer(nn.Module):
    """Single-GPU HSTU layer without Megatron tensor parallelism."""
    def __init__(self, config):
        super().__init__()
        self._embedding_dim = config.hidden_size
        self._linear_dim_per_head = config.kv_channels
        self._attention_dim_per_head = config.kv_channels
        self._eps = config.layernorm_epsilon
        self._dropout_ratio = config.hidden_dropout
        self._num_heads = config.num_attention_heads
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head,
            self._linear_dim_per_head,
            self._attention_dim_per_head,
            self._attention_dim_per_head,
        ]
        total_head_dim = sum(self._split_arg_list)

        if config.learnable_input_layernorm:
            self._input_layernorm = nn.LayerNorm(self._embedding_dim, eps=self._eps)
        else:
            self._input_layernorm = None

        self._linear_uvqk = nn.Linear(
            self._embedding_dim, total_head_dim * self._num_heads,
            bias=getattr(config, 'add_uvqk_bias', False))

        self._output_layernorm = nn.LayerNorm(
            self._num_heads * self._linear_dim_per_head, eps=self._eps)
        self._output_dropout = nn.Dropout(self._dropout_ratio)

        self._linear_proj = nn.Linear(
            self._linear_dim_per_head * self._num_heads, self._embedding_dim, bias=False)

        self._target_group_size = getattr(config, 'target_group_size', 1)

        self._attn_func = create_hstu_attention(
            kernel_backend=config.kernel_backend,
            num_heads=self._num_heads,
            attention_dim=self._attention_dim_per_head,
            linear_dim=self._linear_dim_per_head,
            is_causal=config.is_causal,
        )

    def forward(self, jd: JaggedData) -> JaggedData:
        x = jd.values

        if self._input_layernorm is not None:
            normed_x = self._input_layernorm(x)
        else:
            normed_x = F.layer_norm(x, [self._embedding_dim], eps=self._eps)

        mixed_uvqk = self._linear_uvqk(normed_x)
        silu_uvqk = F.silu(mixed_uvqk)
        silu_uvqk = silu_uvqk.view(-1, self._num_heads, sum(self._split_arg_list))

        user, value, query, key = torch.split(silu_uvqk, self._split_arg_list, dim=-1)

        jagged_attn_output = self._attn_func(
            query, key, value,
            jd.seqlen_offsets,
            num_contextuals=jd.contextual_seqlen,
            num_candidates=jd.num_candidates,
            max_seqlen=jd.max_seqlen,
            scaling_seqlen=jd.scaling_seqlen,
            target_group_size=self._target_group_size,
        )

        normed_attn = self._output_layernorm(jagged_attn_output)
        user_flat = user.reshape(-1, self._num_heads * self._linear_dim_per_head)
        parallel_input = self._output_dropout(normed_attn * user_flat)

        output = self._linear_proj(parallel_input)
        if self._residual:
            output = output + x

        return JaggedData(
            values=output, seqlen=jd.seqlen, seqlen_offsets=jd.seqlen_offsets,
            padding_length=jd.padding_length, max_seqlen=jd.max_seqlen,
            max_num_candidates=jd.max_num_candidates, num_candidates=jd.num_candidates,
            num_candidates_offsets=jd.num_candidates_offsets,
            contextual_max_seqlen=jd.contextual_max_seqlen,
            contextual_seqlen=jd.contextual_seqlen,
            contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
            has_interleaved_action=jd.has_interleaved_action,
            scaling_seqlen=jd.scaling_seqlen,
        )
