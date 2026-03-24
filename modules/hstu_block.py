from typing import Dict, Tuple
import torch
import torch.nn as nn
from compat.jagged_tensor import JaggedTensor
from data.batch import HSTUBatch
from modules.jagged_data import JaggedData
from modules.hstu_processor import HSTUBlockPreprocessor, HSTUBlockPostprocessor
from modules.hstu_layer import HSTULayer


class HSTUBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._preprocessor = HSTUBlockPreprocessor(config, is_inference=False)
        self._postprocessor = HSTUBlockPostprocessor(is_inference=False)
        self._attention_layers = nn.ModuleList([HSTULayer(config) for _ in range(config.num_layers)])

    def forward(self, embeddings: Dict[str, JaggedTensor], batch: HSTUBatch) -> Tuple[JaggedData, Tuple]:
        jd = self._preprocessor(embeddings, batch)
        seqlen_after = jd.seqlen
        num_ctx = jd.contextual_seqlen if jd.contextual_seqlen is not None else torch.zeros_like(seqlen_after)
        num_cand = jd.num_candidates if jd.num_candidates is not None else torch.zeros_like(seqlen_after)
        for layer in self._attention_layers:
            jd = layer(jd)
        return self._postprocessor(jd), (seqlen_after.detach(), num_ctx.detach(), num_cand.detach())
