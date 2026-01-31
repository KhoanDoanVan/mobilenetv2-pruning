import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class StructuredPruner:
    """
    Structured Pruning: Remove entire filters or channels

    Formulas:
    1. score[i] = L1-norm(W[i, :, : , :]) = \sum(|W[i, j, h, w]|)
    2. sorted_indices = argsort(score, descending=True)
    3. keep_indices = sorted_indices[:n_keep]
    4. W_pruned = W[keep_indices, :, : , :]
    """

    def __init__(
            self,
            sparsity: float = 0.5,
            granularity: str = 'filter',
            importance_metric: str = 'l1'
    ):
        """
        Args:
            sparsity: ratio of filters/channels will be remove (0 - 1)
            granularity: 'filter' (output channel) or 'channel' (input channel)
            importance_metric: 'l1', 'l2', or 'mean'
        """
        self.sparsity = sparsity
        self.granularity = granularity
        self.importance_metric = importance_metric
        self.pruned_filters = {} # Save filters were pruned



    def prune_conv_layer(
            self,
            conv_layer: nn.Conv2d,
            sparsity: float = None,
            granularity: str = None
    ) -> nn.Conv2d:
        """
        Prune a Conv2D Layer

        Args:
            conv_layer: Conv2d layer
            sparsity: override sparsity
            granularity: override granularity

        Returns:
            Pruned Conv2d new layer
        """
        if sparsity is None:
            sparsity = self.sparsity

        if granularity is None:
            granularity = self.granularity

        weights = conv_layer.weight.data

        # Calculate importance scores
        scores = self.compute_importance_scores(weights, granularity)

        # Select filters/channels to remain
        keep_indices, prune_indices = self.select_filters_to_prune(scores, sparsity)

        if granularity == 'filter':
            # Prune output channels
            new_out_channels = len(keep_indices)
            new_conv = nn.Conv2d(
                in_channels=conv_layer.in_channels,
                out_channels=new_out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                dilation=conv_layer.dilation,
                groups=conv_layer.groups,
                bias=conv_layer.bias is not None
            )

            # Copy weights
            new_conv.weight.data = weights[keep_indices, :, :, :].clone()
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data[keep_indices].clone()

        elif granularity == 'channel':
            # Prune input channels
            new_in_channels = len(keep_indices)
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=conv_layer.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                dilation=conv_layer.dilation,
                groups=conv_layer.groups if conv_layer.groups == 1 else new_in_channels,
                bias=conv_layer.bias is not None
            )

            # Copy weights
            if conv_layer.groups == 1:
                new_conv.weight.data = weights[:, keep_indices, :, :].clone()
            else:
                # Depthwise conv: groups = in_channels = out_channels
                new_conv.weight.data = weights[keep_indices, :, :, :].clone()
            
            if conv_layer.bias is not None:
                if conv_layer.groups == 1:
                    new_conv.bias.data = conv_layer.bias.data.clone()
                else:
                    new_conv.bias.data = conv_layer.bias.data[keep_indices].clone()

        return new_conv


    def prune_batchnorm_layer(
            self,
            bn_layer: nn.BatchNorm2d,
            keep_indices: np.ndarray
    ) -> nn.BatchNorm2d:
        """
        Prune BatchNorm layer (must match with Conv Layer)

        Args:
            bn_layer: BatchNorm2d Layer
            keep_indices: Indices of channels will be remains

        Returns:
            Pruned BatchNorm2d layer
        """
        new_num_features = len(keep_indices)
        new_bn = nn.BatchNorm2d(
            num_features=new_num_features,
            eps=bn_layer.eps,
            momentum=bn_layer.momentum,
            affine=bn_layer.affine,
            track_running_stats=bn_layer.track_running_stats
        )

        # Copy parameters
        if bn_layer.affine:
            new_bn.weight.data = bn_layer.weight.data[keep_indices].clone()
            new_bn.bias.data = bn_layer.bias.data[keep_indices].clone()
        
        if bn_layer.track_running_stats:
            new_bn.running_mean.data = bn_layer.running_mean.data[keep_indices].clone()
            new_bn.running_var.data = bn_layer.running_var.data[keep_indices].clone()
            new_bn.num_batches_tracked = bn_layer.num_batches_tracked

        return new_bn


    
    def compute_importance_scores(
            self,
            weights: torch.Tensor,
            granularity: str = None
    ) -> np.ndarray:
        """
        Calculate importance score for each filter/channel

        Args:
            weights: Conv weight tensor [out_channels, in_channels, H, W]
            granularity: 'filter' or 'channel'

        Returns:
            Array of importance scores
        """
        if granularity is None:
            granularity = self.granularity

        weights_np = weights.cpu().detach().numpy()

        if granularity == 'filter':
            # Score for each output channel (filter)
            # Shape: [out_channels, in_channels, H, W]
            # Reduce over: (in_channels, H, W)
            if self.importance_metric == 'l1':
                # L1-norm: \sum(|w|)
                scores = np.sum(np.abs(weights_np), axis=(1, 2, 3))
            elif self.importance_metric == 'l2':
                # L2-norm: \sqrt(\sum(w*w))
                scores = np.sqrt(np.sum(weights_np ** 2), axis=(1, 2, 3))
            elif self.importance_metric == 'mean':
                # Mean absolute value
                scores = np.mean(np.abs(weights_np), axis=(1, 2, 3))
            else:
                raise ValueError(f"Unknown metric: {self.importance_metric}")
            
        elif granularity == 'channel':
            # Score for each input channel
            # Reduce over: (out_channels, H, W)
            if self.importance_metric == 'l1':
                # L1-norm: \sum(|w|)
                scores = np.sum(np.abs(weights_np), (0, 2, 3))
            elif self.importance_metric == 'l2':
                # L2-norm: \sqrt(\sum(w*w))
                scores = np.sqrt(np.sum(weights_np ** 2), axis=(0, 2, 3))
            elif self.importance_metric == 'mean':
                # Mean absolute value
                scores = np.mean(np.abs(weights_np), axis=(0, 2, 3))
            else:
                raise ValueError(f"Unknown metric: {self.importance_metric}")
            
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        
    
        return scores
    

    def select_filters_to_prune(
            self,
            scores: np.ndarray,
            sparsity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select filters/channels to prune depend on scores

        Args:
            scores: Importance scores
            sparsity: Ratio to prune

        Returns:
            (indices_to_keep, indices_to_prune)
        """

        num_total = len(scores)
        num_to_prune = int(num_total * sparsity)
        num_to_keep = num_total - num_to_prune

        # Sort by importance (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Remains top importance filters
        indices_to_keep = sorted_indices[:num_to_keep]
        indices_to_prune = sorted_indices[num_to_keep:]

        return np.sort(indices_to_keep), np.sort(indices_to_prune)
    

    def get_layer_sparsity_stats(self, model: nn.Module) -> Dict[str, Dict]:
        """
        Statistic sparsity able to achive for each layer

        Args:
            model: Pytorch model

        Returns:
            Dict contains stats
        """
        stats = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weights = module.weight.data
                scores = self.compute_importance_scores(weights)

                stats[name] = {
                    'out_channels': module.out_channels,
                    'in_channels': module.in_channels,
                    'kernel_size': module.kernel_size,
                    'total_params': weights.numel(),
                    'min_score': float(scores.min()),
                    'max_score': float(scores.max()),
                    'mean_score': float(scores.mean()),
                    'std_score': float(scores.std())
                }

        return stats