import itertools
from typing import Dict, Optional
import torch
from compat.jagged_tensor import JaggedTensor
from data.batch import HSTUBatch
from ops.jagged_ops import asynchronous_complete_cumsum
from ops.jagged_concat import jagged_2D_tensor_concat
from modules.jagged_data import JaggedData
from modules.mlp import MLP

try:
    from ops.triton_jagged import triton_split_2D_jagged
    HAS_TRITON_JAGGED = True
except ImportError:
    HAS_TRITON_JAGGED = False


def split_2D_jagged_fallback(values, max_seqlen, offsets_a, offsets_b):
    """Pure PyTorch fallback for triton_split_2D_jagged."""
    batch_size = offsets_a.size(0) - 1
    dim = values.size(1)
    device = values.device
    dtype = values.dtype

    total_a = offsets_a[-1].item()
    total_b = offsets_b[-1].item()
    part_a = torch.zeros(int(total_a), dim, device=device, dtype=dtype)
    part_b = torch.zeros(int(total_b), dim, device=device, dtype=dtype)

    src_offset = 0
    for b in range(batch_size):
        len_a = (offsets_a[b+1] - offsets_a[b]).item()
        len_b = (offsets_b[b+1] - offsets_b[b]).item()
        total_len = len_a + len_b
        src_start = src_offset
        if len_a > 0:
            dst_start_a = offsets_a[b].item()
            part_a[dst_start_a:dst_start_a+len_a] = values[src_start:src_start+len_a]
        if len_b > 0:
            dst_start_b = offsets_b[b].item()
            part_b[dst_start_b:dst_start_b+len_b] = values[src_start+len_a:src_start+len_a+len_b]
        src_offset += total_len

    return part_a, part_b


def hstu_preprocess_embeddings(embeddings, batch, is_inference, item_mlp=None, contextual_mlp=None, dtype=None, scaling_seqlen=-1):
    item_jt = embeddings[batch.item_feature_name]
    dtype = item_jt.values().dtype if dtype is None else dtype
    sequence_embeddings = item_jt.values().to(dtype)
    sequence_embeddings_lengths = item_jt.lengths()
    sequence_embeddings_lengths_offsets = item_jt.offsets()
    sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

    if batch.action_feature_name is not None:
        action_jt = embeddings[batch.action_feature_name]
        jagged_size = sequence_embeddings.size(0)
        embedding_dim = sequence_embeddings.size(1)
        sequence_embeddings = torch.cat(
            [sequence_embeddings, action_jt.values().to(dtype)], dim=1
        ).view(2 * jagged_size, embedding_dim)
        sequence_embeddings_lengths = sequence_embeddings_lengths * 2
        sequence_embeddings_lengths_offsets = sequence_embeddings_lengths_offsets * 2
        sequence_max_seqlen = sequence_max_seqlen * 2
        if item_mlp is not None:
            sequence_embeddings = item_mlp(sequence_embeddings)

    if batch.num_candidates is not None and batch.action_feature_name is not None:
        num_candidates = batch.num_candidates * 2
        max_num_candidates = batch.max_num_candidates * 2
    else:
        num_candidates = batch.num_candidates
        max_num_candidates = batch.max_num_candidates

    contextual_max_seqlen = 0
    contextual_seqlen = None
    contextual_seqlen_offsets = None
    if len(batch.contextual_feature_names) > 0:
        contextual_max_seqlens = [batch.feature_to_max_seqlen[name] for name in batch.contextual_feature_names]
        contextual_jts = [embeddings[name] for name in batch.contextual_feature_names]
        contextual_jts_values = [jt.values().to(dtype) for jt in contextual_jts]
        contextual_jts_offsets = [jt.offsets() for jt in contextual_jts]
        contextual_sequence_embeddings, contextual_seqlen = jagged_2D_tensor_concat(
            contextual_jts_values, contextual_jts_offsets, contextual_max_seqlens)
        if torch.sum(contextual_seqlen, dim=0).cpu().item() == 0:
            contextual_seqlen = None
        else:
            if contextual_mlp is not None:
                contextual_sequence_embeddings = contextual_mlp(contextual_sequence_embeddings)
            contextual_seqlen_offsets = asynchronous_complete_cumsum(contextual_seqlen)
            contextual_max_seqlen = max(len(batch.contextual_feature_names), sum(contextual_max_seqlens))
            sequence_embeddings, sequence_embeddings_lengths = jagged_2D_tensor_concat(
                [contextual_sequence_embeddings, sequence_embeddings],
                [contextual_seqlen_offsets, sequence_embeddings_lengths_offsets],
                [contextual_max_seqlen, sequence_max_seqlen])
            sequence_embeddings_lengths_offsets = asynchronous_complete_cumsum(sequence_embeddings_lengths)
            sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

    from ops.jagged_ops import length_to_complete_offsets
    return JaggedData(
        values=sequence_embeddings,
        seqlen=sequence_embeddings_lengths.to(torch.int32),
        seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
        max_seqlen=sequence_max_seqlen,
        max_num_candidates=max_num_candidates,
        num_candidates=num_candidates.to(torch.int32) if num_candidates is not None else None,
        num_candidates_offsets=length_to_complete_offsets(num_candidates).to(torch.int32) if num_candidates is not None else None,
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=contextual_seqlen.to(torch.int32) if contextual_seqlen is not None else None,
        contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32) if contextual_seqlen_offsets is not None else None,
        has_interleaved_action=batch.action_feature_name is not None,
        scaling_seqlen=scaling_seqlen,
    )


class HSTUBlockPreprocessor(torch.nn.Module):
    def __init__(self, config, is_inference=False):
        super().__init__()
        self.config = config
        self._training_dtype = torch.float32
        if config.bf16:
            self._training_dtype = torch.bfloat16
        if config.fp16:
            self._training_dtype = torch.float16
        self._item_mlp = None
        self._contextual_mlp = None
        self._is_inference = is_inference
        self._dropout_ratio = 0.0 if is_inference else config.hidden_dropout
        self._scaling_seqlen = config.scaling_seqlen

        self._positional_encoder = None
        if getattr(config, 'position_encoding_config', None) is not None:
            from modules.position_encoder import HSTUPositionalEncoder
            pe_cfg = config.position_encoding_config
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=pe_cfg.num_position_buckets,
                embedding_dim=config.hidden_size,
                training_dtype=self._training_dtype,
                num_time_buckets=getattr(pe_cfg, 'num_time_buckets', 2048),
                use_time_encoding=getattr(pe_cfg, 'use_time_encoding', False),
            )

    def forward(self, embeddings, batch):
        jd = hstu_preprocess_embeddings(
            embeddings, batch, is_inference=self._is_inference,
            item_mlp=self._item_mlp, contextual_mlp=self._contextual_mlp,
            dtype=self._training_dtype, scaling_seqlen=self._scaling_seqlen)

        if self._positional_encoder is not None:
            jd.values = self._positional_encoder(
                max_seq_len=jd.max_seqlen,
                seq_lengths=jd.seqlen,
                seq_offsets=jd.seqlen_offsets,
                seq_embeddings=jd.values,
                num_targets=jd.num_candidates,
            )

        jd.values = torch.nn.functional.dropout(
            jd.values, p=self._dropout_ratio, training=self.training
        ).to(self._training_dtype)
        return jd


class HSTUBlockPostprocessor(torch.nn.Module):
    def __init__(self, is_inference=False):
        super().__init__()
        self._is_inference = is_inference

    def forward(self, jd):
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            split_fn = triton_split_2D_jagged if HAS_TRITON_JAGGED else split_2D_jagged_fallback
            _, sequence_embeddings = split_fn(
                jd.values, jd.max_seqlen,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets)
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            split_fn = triton_split_2D_jagged if HAS_TRITON_JAGGED else split_2D_jagged_fallback
            _, sequence_embeddings = split_fn(
                jd.values, jd.max_seqlen,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets)
        else:
            sequence_embeddings = jd.values
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        if jd.has_interleaved_action and not self._is_inference:
            sequence_embeddings = sequence_embeddings[0::2, ...]
            seqlen_offsets = seqlen_offsets // 2
            max_seqlen = max_seqlen // 2

        sequence_embeddings = sequence_embeddings / torch.linalg.norm(
            sequence_embeddings, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)

        return JaggedData(
            values=sequence_embeddings,
            seqlen=torch.diff(seqlen_offsets).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
            scaling_seqlen=jd.scaling_seqlen,
        )
