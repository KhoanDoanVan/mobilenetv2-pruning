from structured_pruner import StructuredPruner
import torch.nn as nn



class MobileNetV2StructuredPruner(StructuredPruner):
    """
    Spectialize pruner for MobileNetV2 with many special rules
    """

    def __init__(
            self,
            pointwise_sparsity: float = 0.7,
            depthwise_sparsity: float = 0.3,
            skip_first_last: bool = True
    ):
        """
        Args:
            pointwise_sparsity: sparsity for pointwise (1x1) convs
            depthwise_sparsity: sparsity for depthwise convs
            skip_first_last: ignore the first layer and the last
        """
        super().__init__()
        self.pointwise_sparsity = pointwise_sparsity
        self.depthwise_sparsity = depthwise_sparsity
        self.skip_first_last = skip_first_last
    
    def determine_layer_sparsity(
            self,
            module: nn.Conv2d,
            layer_name: str
    ) -> float:
        """
        Find sparsity for each layer

        Args:
            module: Conv2d module
            layer_name: name of the layer

        Returns:
            Sparsity for current layer
        """
        # Skip first/last layers
        if self.skip_first_last:
            if 'features.0' in layer_name or 'classifier' in layer_name:
                return 0.0
            
        # Depthwise conv (groups = in_channels)
        if module.groups == module.in_channels and module.groups > 1:
            return self.depthwise_sparsity
        
        # Pointwise conv (1x1)
        elif module.kernel_size == (1, 1):
            return self.pointwise_sparsity
        
        # Regular conv
        else:
            return (self.pointwise_sparsity  + self.depthwise_sparsity) / 2