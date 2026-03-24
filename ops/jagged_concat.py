import torch
from typing import List, Tuple

def jagged_2D_tensor_concat(
    values_list: List[torch.Tensor],
    offsets_list: List[torch.Tensor],
    max_lengths: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate multiple 2D jagged tensors along the sequence dimension.

    Args:
        values_list: List of [total_i, D] tensors
        offsets_list: List of [B+1] offset tensors
        max_lengths: List of max sequence lengths

    Returns:
        (concatenated_values, concatenated_lengths) where:
        - concatenated_values: [new_total, D]
        - concatenated_lengths: [B]
    """
    batch_size = offsets_list[0].size(0) - 1
    device = values_list[0].device
    dtype = values_list[0].dtype
    dim = values_list[0].size(1)

    all_lengths = []
    for offsets in offsets_list:
        lengths = offsets[1:] - offsets[:-1]
        all_lengths.append(lengths)

    total_lengths = sum(l for l in all_lengths)
    total_elements = total_lengths.sum().item()

    result = torch.zeros(int(total_elements), dim, device=device, dtype=dtype)
    result_offset = 0
    for b in range(batch_size):
        for i, (vals, offs) in enumerate(zip(values_list, offsets_list)):
            start = offs[b].item()
            end = offs[b + 1].item()
            n = end - start
            if n > 0:
                result[result_offset:result_offset + n] = vals[start:end]
                result_offset += n

    return result, total_lengths
