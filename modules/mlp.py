import torch
import torch.nn as nn
from typing import List, Optional

def init_mlp_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class MLP(nn.Module):
    def __init__(self, in_size: int, layer_sizes: List[int], activation: str = "relu",
                 bias: bool = True, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(layer_sizes[i-1] if i > 0 else in_size, size, bias=bias, device=device, dtype=dtype))
            layers.append(act_fn() if i < len(layer_sizes) - 1 else nn.Identity())

        self._mlp = nn.Sequential(*layers)
        self._mlp.apply(init_mlp_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2
        return self._mlp(input)
