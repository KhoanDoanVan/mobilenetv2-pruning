from torchvision.models import mobilenet_v2
from utils.pruning_utils import (
    count_parameters,
    get_model_size,
    calculate_flops
)



if __name__ == "__main__":

    model = mobilenet_v2(pretrained=False)

    params = count_parameters(model)
    print(f"Total: {params['total']:,}")
    print(f"Trainable: {params['trainable']:,}")


    size = get_model_size(model)
    print(f"Size: {size['total_size_mb']:.2f} MB")


    flops = calculate_flops(model)
    print(f"FLOPs: {flops:,}")