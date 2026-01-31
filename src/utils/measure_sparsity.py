import torch
from typing import Dict


def measure_sparsity(
        model: torch.nn.Module
) -> Dict[str, float]:
    """
    Measure sparsity of model

    Args:
        model: Pytorch model

    Returns:
        Dict contains overall and per-layer sparsity
    """
    total_params = 0
    zero_params = 0
    layer_stats = {}

    for name, param in model.named_parameters():
        if 'weight' not in name or param.dim() <= 1:
            continue

        layer_total = param.numel()
        layer_zero = (param == 0).sum().item()

        total_params += layer_total
        zero_params += layer_zero

        layer_stats[name] = layer_zero / layer_total

    return {
        'overall': zero_params / total_params if total_params > 0 else 0,
        'layers': layer_stats
    }