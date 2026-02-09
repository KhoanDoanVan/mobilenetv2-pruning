import torch
import torchvision.models as models
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.pruning_utils import get_model_size


def export_to_coreml(model, output_path, quantize=True):
    """
    Export PyTorch model sang CoreML
    
    Args:
        model: PyTorch model
        output_path: Output path (.mlmodel)
        quantize: Apply quantization (int8)
        
    Returns:
        CoreML model path
    """
    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools not installed!")
        print("Install with: pip install coremltools")
        return None
    
    print(f"\nConverting to CoreML...")
    
    # Prepare model
    model.eval()
    
    # Trace model
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    print("  - Tracing model...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=(1, 3, 224, 224))],
        convert_to="mlprogram" if not quantize else "neuralnetwork"
    )
    
    # Quantization
    if quantize:
        print("  - Applying int8 quantization...")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=8
        )
    
    # Save
    mlmodel.save(output_path)
    print(f"CoreML model saved to {output_path}")
    
    # Get size
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  - File size: {size_mb:.2f} MB")
    
    return output_path