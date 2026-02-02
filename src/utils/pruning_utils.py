import torch
import torch.nn as nn
import os
import json
from typing import Dict, List, Tuple


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count amount parameters in model

    Args:
        model: Pytorch model

    Returns:
        Dict contains info parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }



def count_conv_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in Conv Layers Only

    Args:
        model: Pytorch model

    Returns:
        Dict contains info Conv Parameters
    """

    conv_params = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            conv_params += sum(p.numel() for p in module.parameters())
        total_params += sum(p.numel() for p in module.parameters())

    return {
        'conv_params': conv_params,
        'total_params': total_params,
        'conv_ratio': conv_params / total_params if total_params > 0 else 0
    }



def calculate_flops(
        model: nn.Module,
        input_size: Tuple[int, int, int] = (3, 224, 224)
) -> int:
    """
    Estimate FLOPs of model

    Args:
        model: Pytorch model
        input_size: (C, H, W)

    Returns:
        Sum of FLOPs (Floating-Point Operations Per Second)
    """
    flops = 0

    def conv_flops_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        output_height, output_width = output.size(2), output.size(3)

        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        output_ops = output_height * output_width * module.out_channels

        flops += int(batch_size * kernel_ops * output_ops)


    def linear_flops_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        flops += int(batch_size * module.in_features * module.out_features)

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_flops_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops_hook))

    # Forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    model.eval()

    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return flops



def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate size of the model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict contains size info
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_size_bytes': param_size,
        'buffer_size_bytes': buffer_size,
        'total_size_bytes': param_size + buffer_size,
        'total_size_mb': size_mb
    }



def save_pruning_info(
        model: nn.Module, 
        save_path: str,
        pruning_config: Dict = None,
        metrics: Dict = None
):
    """
    Save information about the model pruned
    
    Args:
        model: Pruned model
        save_path: path to save
        pruning_config: Config 
        metrics: Metrics after prune
    """
    info = {
        'parameters': count_parameters(model),
        'conv_parameters': count_conv_parameters(model),
        'model_size': get_model_size(model),
    }
    
    if pruning_config:
        info['pruning_config'] = pruning_config
    
    if metrics:
        info['metrics'] = metrics
    
    # Save to JSON
    json_path = save_path.replace('.pth', '_info.json')
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Pruning info saved to {json_path}")


def compare_models(
        original_model: nn.Module,
        pruned_model: nn.Module,
        input_size: Tuple[int, int, int] = (3, 224, 224)
) -> Dict:
    """
    Compare orignial model and model pruned

    Args:
        original_model: Model original
        pruned_model: Model pruned
        input_size: Input size to cal FLOPs
    """
    orig_params = count_conv_parameters(original_model)
    pruned_params = count_conv_parameters(pruned_model)

    orig_size = get_model_size(original_model)
    pruned_size = get_model_size(pruned_model)

    orig_flops = calculate_flops(original_model, input_size)
    pruned_flops = calculate_flops(pruned_model, input_size)

    comparison = {
        'parameters': {
            'original': orig_params['total'],
            'pruned': pruned_params['total'],
            'reduction': 1 - pruned_params['total'] / orig_params['total'],
            'compression_ratio': orig_params['total'] / pruned_params['total']
        },
        'model_size': {
            'original_mb': orig_size['total_size_mb'],
            'pruned_mb': pruned_size['total_size_mb'],
            'reduction': 1 - pruned_size['total_size_mb'] / orig_size['total_size_mb'],
            'compression_ratio': orig_size['total_size_mb'] / pruned_size['total_size_mb']
        },
        'flops': {
            'original': orig_flops,
            'pruned': pruned_flops,
            'reduction': 1 - pruned_flops / orig_flops if orig_flops > 0 else 0,
            'speedup': orig_flops / pruned_flops if pruned_flops > 0 else 0
        }
    }

    return comparison



def print_comparison(comparison: Dict):
    """
    Print out comparison
    
    Args:
        comparison: Dict from compare_models()
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print("\nParameters:")
    print(f"  Original: {comparison['parameters']['original']:,}")
    print(f"  Pruned:   {comparison['parameters']['pruned']:,}")
    print(f"  Reduction: {comparison['parameters']['reduction']*100:.2f}%")
    print(f"  Compression: {comparison['parameters']['compression_ratio']:.2f}x")
    
    print("\nModel Size:")
    print(f"  Original: {comparison['model_size']['original_mb']:.2f} MB")
    print(f"  Pruned:   {comparison['model_size']['pruned_mb']:.2f} MB")
    print(f"  Reduction: {comparison['model_size']['reduction']*100:.2f}%")
    print(f"  Compression: {comparison['model_size']['compression_ratio']:.2f}x")
    
    print("\nFLOPs:")
    print(f"  Original: {comparison['flops']['original']:,}")
    print(f"  Pruned:   {comparison['flops']['pruned']:,}")
    print(f"  Reduction: {comparison['flops']['reduction']*100:.2f}%")
    print(f"  Speedup: {comparison['flops']['speedup']:.2f}x")
    
    print("\n" + "="*60 + "\n")


def load_checkpoint(checkpoint_path: str, model: nn.Module) -> Tuple[nn.Module, Dict]:
    """
    Load checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model để load weights vào
        
    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, checkpoint


def save_checkpoint(
        model: nn.Module,
        save_path: str,
        epoch: int = None,
        optimizer = None,
        metrics: Dict = None
):
    """
    Save checkpoint
    
    Args:
        model: Model to save
        save_path: path save
        epoch: Epoch current
        optimizer: Optimizer state
        metrics: Training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
