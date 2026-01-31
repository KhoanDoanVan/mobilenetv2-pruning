import torch
from magnitude_pruner import MagnitudePruner

if __name__ == "__main__":

    # Test with Random Tensor

    # Create random weights
    weights = torch.randn(64, 64, 3, 3)
    print(f"Original Shape: {weights.shape}")
    print(f"Original none-zero: {(weights != 0).sum()}/{weights.numel()}")

    # Prune
    pruner = MagnitudePruner(sparsity=0.7)
    pruned_weights, mask = pruner.prune_layer("test_layer", weights)

    print(f"Pruned non-zero: {(pruned_weights != 0).sum().item()}/{pruned_weights.numel()}")
    print(f"Actual sparsity: {(pruned_weights == 0).sum().item() / pruned_weights.numel():.4f}")
          
    # Stats
    stats = pruner.get_sparsity_stats()
    print(f"Stats: {stats}")