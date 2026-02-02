import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import sys
import yaml
from copy import deepcopy
from mobilev2_structured_pruner import MobileNetV2StructuredPruner
from magnitude_pruner import MagnitudePruner
from utils.pruning_utils import (
    count_parameters, get_model_size, calculate_flops,
    compare_models, print_comparison, save_checkpoint
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def prune_mobilenetv2_structured(model, sparsity_config):
    """
    Prune MobileNetV2 with structured pruning

    Args:
        model: MobileNetV2 model
        sparsity_config: Dict contains config
    
    Returns:
        Pruned model
    """

    pruner = MobileNetV2StructuredPruner(
        pointwise_sparsity=sparsity_config.get('pointwise_sparsity', 0.7),
        depthwise_sparsity=sparsity_config.get('depthwise_sparsity', 0.3),
        skip_first_last=sparsity_config.get('skip_first_last', True)
    )

    print("Analyzing layers...")

    layer_count = {
        'pointwise': 0,
        'depthwise': 0,
        'regular': 0,
        'skipped': 0
    }

    # Clone model structure
    pruned_model = deepcopy(model)

    # Prune features (convolutional layers)
    for i, module in enumerate(pruned_model.features):

        if isinstance(module, nn.Conv2d):
            layer_name = f'features.{i}'
            sparsity = pruner.determine_layer_sparsity(module, layer_name)

            if sparsity > 0:
                print(f"  Pruning {layer_name}: sparsity={sparsity:.2f}")

                # Decided type of layer
                if module.groups == module.in_channels and module.groups > 1:
                    layer_count['depthwise'] += 1
                elif module.kernel_size == (1, 1):
                    layer_count['pointwise'] += 1
                else:
                    layer_count['regular'] += 1

                # Prune layer
                pruned_conv = pruner.prune_conv_layer(
                    module,
                    sparsity=sparsity,
                    granularity='filter'
                )

                # Replace in model
                pruned_model.features[i] = pruned_conv

                # If had BatchNorm in the next layer of conv -> let's prune that batchnorm layer
                if i + 1 < len(pruned_model.features):
                    next_module = pruned_model.features[i + 1]
                    if isinstance(next_module, nn.BatchNorm2d):
                        # Get indices had remained
                        keep_indices = list(range(pruned_conv.out_channels))
                        pruned_bn = pruner.prune_batchnorm_layer(
                            next_module,
                            keep_indices
                        )
                        pruned_model.features[i+1] = pruned_bn
                else:
                    layer_count['skipped'] += 1

    print(f"\n===> Pruned layers:")
    print(f"  - Pointwise (1x1): {layer_count['pointwise']}")
    print(f"  - Depthwise: {layer_count['depthwise']}")
    print(f"  - Regular: {layer_count['regular']}")
    print(f"  - Skipped: {layer_count['skipped']}")

    return pruned_model


def prune_mobilenetv2_magnitude(model, sparsity):
    """
    Prune MobileNetV2 with magnitude-based pruning

    Args:
        model: MobileNetV2 model
        sparsity: Global sparsity

    Returns:
        (Pruned model, masks)
    """

    pruner = MagnitudePruner(sparsity=sparsity, global_pruning=True)

    pruned_model = pruner.prune_model(model)

    # Get stats
    stats = pruner.get_sparsity_stats()
    print(f"\n==> Pruned {len(stats)} layers")

    return pruned_model, pruner.masks