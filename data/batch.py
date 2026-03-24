import dataclasses
from typing import Dict, List, Optional

import torch

from compat.jagged_tensor import JaggedTensor, KeyedJaggedTensor


@dataclasses.dataclass
class HSTUBatch:
    features: KeyedJaggedTensor
    labels: JaggedTensor
    batch_size: int
    item_feature_name: str
    action_feature_name: Optional[str] = None
    contextual_feature_names: List[str] = dataclasses.field(default_factory=list)
    feature_to_max_seqlen: Dict[str, int] = dataclasses.field(default_factory=dict)
    num_candidates: Optional[torch.Tensor] = None
    max_num_candidates: int = 0

    def to(self, device, non_blocking=False):
        return HSTUBatch(
            features=self.features.to(device),
            labels=JaggedTensor(
                values=self.labels.values().to(device, non_blocking=non_blocking),
                lengths=self.labels.lengths().to(device, non_blocking=non_blocking)
                if self.labels.lengths() is not None
                else None,
            ),
            batch_size=self.batch_size,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            contextual_feature_names=self.contextual_feature_names,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            num_candidates=self.num_candidates.to(
                device, non_blocking=non_blocking
            )
            if self.num_candidates is not None
            else None,
            max_num_candidates=self.max_num_candidates,
        )
