import torch
import torchvision.models as models
import os
import sys
from utils.pruning_utils import (
    count_parameters,
    get_model_size,
    calculate_flops
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def download_model(
        save_dir: str = '../models'
):
    """
    Download pretrained MobileNetV2 from torchvision

    Args:
        save_dir: folder to contains model
    """

    os.makedirs(save_dir, exist_ok=True)

    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    print("Download Completed !!!")

    params = count_parameters(model)
    size = get_model_size(model)
    flops = calculate_flops(model, input_size=(3, 224, 224))

    print(f"\n===> Model Statistics:")
    print(f"  - Total parameters: {params['total']:,}")
    print(f"  - Trainable parameters: {params['trainable']:,}")
    print(f"  - Model size: {size['total_size_mb']:.2f} MB")
    print(f"  - FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")

    # save model
    save_path = os.path.join(save_dir, 'mobilenetv2_pretrained.pth')
    print(f"\nSaving to {save_path}...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': {
            'architecture': 'mobilenet_v2',
            'parameters': params,
            'size': size,
            'flops': flops,
            'pretrained': True
        }
    }, save_path)

    print("\nModel saved !!!\n")

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"==> Forward pass successful! Output shape: {output.shape}")
    print(f"\nModel saved to: {save_path}")

    return model, save_path