import functools
from math import sqrt

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from compat.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from compat.embedding_config import EmbeddingConfig


def _default_embedding_init_fn(num_embeddings: int) -> callable:
    """Match TorchRec EmbeddingConfig default: uniform(-sqrt(1/N), sqrt(1/N))."""
    bound = sqrt(1.0 / num_embeddings)
    return functools.partial(torch.nn.init.uniform_, a=-bound, b=bound)


class MUSAEmbeddingCollection(nn.Module):
    def __init__(self, configs: List[EmbeddingConfig], device: Optional[torch.device] = None):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self._configs = configs
        for config in configs:
            self.embeddings[config.name] = nn.Embedding(
                config.num_embeddings, config.embedding_dim, device=device
            )
            init_fn = config.init_fn or _default_embedding_init_fn(config.num_embeddings)
            init_fn(self.embeddings[config.name].weight)

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        feature_dict = features.to_dict()
        result = {}
        for config in self._configs:
            for fname in config.feature_names:
                if fname in feature_dict:
                    jt = feature_dict[fname]
                    emb = self.embeddings[config.name](jt.values().long())
                    result[fname] = JaggedTensor(values=emb, lengths=jt.lengths(), offsets=jt.offsets())
        return result

class ShardedEmbedding(nn.Module):
    """Drop-in replacement for the original ShardedEmbedding, using plain PyTorch embeddings."""
    def __init__(self, embedding_configs):
        super().__init__()
        configs = []
        for cfg in embedding_configs:
            configs.append(EmbeddingConfig(
                name=cfg.table_name,
                embedding_dim=cfg.dim,
                num_embeddings=cfg.vocab_size,
                feature_names=cfg.feature_names,
            ))
        self._collection = MUSAEmbeddingCollection(configs)
        self.freeze_embedding = "0"

    def _maybe_detach(self, embeddings):
        if self.freeze_embedding == "1":
            for key, emb in embeddings.items():
                emb._values = emb._values.detach()
        return embeddings

    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        return self._collection(kjt)
