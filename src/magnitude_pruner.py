import torch
import numpy as np
from typing import Dict, List, Tuple


class MagnitudePruner:
    """
    Magnitude-based pruning: remove minimum weights

    Formulas:
    1. M[i, j] = |M[i, j]|
    2. threshold = percentile(M, sparsity * 100)
    3. mask[i, j] = 1 if M[i, j] >= threshold else 0
    4. W_pruned = W * mask
    """

    def __init__(
            self,
            sparsity: float = 0.5,
            global_pruning: bool = True
    ):
        """
        Args:
            sparsity: ratio weights will be remove (0 - 1)
            global_pruning: True = threshold for entire model,
                            False = threshold for each layer
        """
        self.sparsity = sparsity
        self.global_pruning = global_pruning
        self.masks = {} # Save mask of layers


    def prune_model(
            self,
            model: torch.nn.Module,
            layer_sparsity: Dict[str, float] = None
    ) -> torch.nn.Module:
        """
        Prune entire the model

        Args:
            model: Pytorch model
            layer_sparsity: Dict mapping layer name -> sparsity 

        Returns:
            Pruned model
        """

        # Collect entire weights if global pruning
        if self.global_pruning and layer_sparsity is None:
            all_weights = []
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1: # bypass bias and 1D Tensors
                    all_weights.append(
                        param.data.cpu().numpy().flatten()
                    )

            all_weights = np.concatenate(all_weights)
            global_threshold = np.percentile(np.abs(all_weights), self.sparsity * 100)
        
        else:
            global_threshold = None


        # Prune each of layer 
        for name, param in model.named_parameters():
            if 'weight' not in name or param.dim() <= 1:
                continue

            # Find sparsity for current layer
            if layer_sparsity and name in layer_sparsity:
                layer_sp = layer_sparsity[name]
            else:
                layer_sp = self.sparsity


            # Prune
            if global_threshold is not None:
                mask = self.create_mask(param.data, global_threshold)
                param.data *= mask
                self.masks[name] = mask
            else:
                pruned_weights, mask = self.prune_layer(
                    name,
                    param.data,
                    layer_sp
                )
                param.data = pruned_weights

        return model

    
    def calculate_threshold(
            self,
            weights: torch.Tensor,
            sparsity: float
    ) -> float:
        """
        Calculate threshold depend on percentile

        Args:
            weights: Tensor include weights
            sparsity: Ratio want to prune

        Returns:
            threshold value
        """

        # Calculate magnitude (absolute values)
        magnitudes = torch.abs(weights).cpu().numpy().flatten()

        # Calculate percentile
        # Example: sparsity = 70% -> hold 30% largest of weights
        # -> threshold = percentile at 70%
        percentile_value = sparsity * 100
        threshold = np.percentile(magnitudes, percentile_value)

        return threshold
    

    def create_mask(
            self,
            weights: torch.Tensor,
            threshold: float
    ) -> torch.Tensor:
        """
        Create binary mask

        Args:
            weights: weight tensor
            threshold: threshold for prune

        Return:
            Binary mask (1 = hold, 0 = prune)
        """
        
        # Calculate magnitude
        magnitudes = torch.abs(weights)

        # Create mask: 1 if magnitude >= threshold, 0 if converse
        mask = (magnitudes >= threshold).float()

        return mask
    

    def prune_layer(
            self,
            layer_name: str,
            weights: torch.Tensor,
            sparsity: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune a layer

        Args:
            layer_name: name of layer
            weights: weight tensor
            sparsity: override sparsity for this layer

        Returns:
            (pruned_weights, mask)
        """
        if sparsity is None:
            sparsity = self.sparsity

        # Calculate threshold
        threshold = self.calculate_threshold(weights, sparsity)

        # Create mask
        mask = self.create_mask(weights, threshold)

        # Apply mask (element-wise multiplication)
        pruned_weights = weights * mask

        # Save mask
        self.masks[layer_name] = mask

        return pruned_weights, mask
    

    def get_sparsity_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Statistic sparsity for each layer

        Returns:
            Dict contains stats of each layer
        """

        stats = {}
        for layer_name, mask in self.masks.items():
            total_params = mask.numel()
            zero_params = (mask == 0).sum().item()
            nonzero_params = total_params - zero_params

            stats[layer_name] = {
                'total': total_params,
                'zero': zero_params,
                'nonzero': nonzero_params,
                'sparsity': zero_params / total_params
            }
        return stats
    

    def apply_mask(
            self,
            model: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Apply masks was saved on model (use in training phase)

        Args:
            model: Pytorch model

        Returns:
            Model with masks applied
        """
        for name, param in model.named_parameters():
            if name in self.masks:
                param += self.masks[name]
        return model