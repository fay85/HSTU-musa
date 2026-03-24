from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from compat.jagged_tensor import JaggedTensor
from data.batch import HSTUBatch
from modules.embedding import ShardedEmbedding
from modules.hstu_block import HSTUBlock
from modules.mlp import MLP
from modules.multi_task_loss import MultiTaskLossModule


class RankingGR(nn.Module):
    def __init__(self, hstu_config, task_config):
        super().__init__()
        self._hstu_config = hstu_config
        self._task_config = task_config
        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)
        self._hstu_block = HSTUBlock(hstu_config)
        self._mlp = MLP(hstu_config.hidden_size, task_config.prediction_head_arch,
                        task_config.prediction_head_act_type, task_config.prediction_head_bias)
        self._loss_module = MultiTaskLossModule(
            num_classes=task_config.prediction_head_arch[-1],
            num_tasks=task_config.num_tasks, reduction="none")

    def get_logit_and_labels(self, batch):
        embeddings = self._embedding_collection(batch.features)
        embeddings = self._embedding_collection._maybe_detach(embeddings)
        hidden_states_jagged, seqlen_after = self._hstu_block(embeddings=embeddings, batch=batch)
        logits = self._mlp(hidden_states_jagged.values)
        return logits, seqlen_after, batch.labels.values()

    def forward(self, batch):
        logits, seqlen_after, labels = self.get_logit_and_labels(batch)
        losses = self._loss_module(logits.float(), labels)
        return losses, (losses.detach(), logits.detach(), labels.detach(), seqlen_after)
