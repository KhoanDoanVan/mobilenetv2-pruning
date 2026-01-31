import torch
from typing import List

def get_prunable_layers(
        model: torch.nn.Module
) -> List[str]:
    """
    Get list of layers can be prune

    Args:
        model: Pytorch model
    
    Returns:
        List name of layers
    """
    prunable = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            prunable.append(name)

    return prunable