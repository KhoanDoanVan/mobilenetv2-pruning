import torch.nn as nn
from structured_pruner import StructuredPruner


if __name__ == "__main__":

    # Create example conv layer
    conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    print(f"Original Conv: in={conv.in_channels}, out={conv.out_channels}")
    print(f"Original params: {sum(p.numel() for p in conv.parameters())}")

    # Prune filters (output channels)
    pruner = StructuredPruner(sparsity=0.5, granularity='filter')
    # pruner = StructuredPruner(sparsity=0.5, granularity='channel')
    pruned_conv = pruner.prune_conv_layer(conv)

    print(f"Pruned Conv: in={pruned_conv.in_channels}, out={pruned_conv.out_channels}")
    print(f"Pruned params: {sum(p.numel() for p in pruned_conv.parameters())}")
    print(f"Compression ratio: {sum(p.numel() for p in conv.parameters()) / sum(p.numel() for p in pruned_conv.parameters()):.2f}x")