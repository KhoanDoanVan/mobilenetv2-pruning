import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import argparse
import os
import sys
from training.pruning_aware_trainer import PruningAwareTrainer
from training.quick_evaluator import QuickEvaluator

sys.path.append(os.path.dirname(__file__), '..', 'src')



def create_dataloader(
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        use_subset: bool = True,
        subset_size: int = 10000
):
    """
    Create data loaders for training and validation

    Args:
        data_dir: path to ImageNet dataset
        batch_size: batch size
        num_workers: num of workers
        use_subset: using small subset for demo
        subset_size: size of subset

    Return:
        (train_loader, val_loader)
    """
    # Transform for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
    if os.path.exists(data_dir):
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transforms
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transforms
        )

        # Use subset for quick demo
        if use_subset:
            train_indices = torch.randperm(
                len(train_dataset)
            )[:subset_size]
            val_indices = torch.randperm(
                len(val_dataset)
            )[:subset_size // 10]

            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)

    else:
        print("   Creating dummy datasets for demonstration...")

        from torch.utils.data import TensorDataset
        
        # Random data
        train_data = torch.randn(1000, 3, 224, 224)
        train_labels = torch.randint(0, 1000, (1000,))
        train_dataset = TensorDataset(train_data, train_labels)
        
        val_data = torch.randn(100, 3, 224, 224)
        val_labels = torch.randint(0, 1000, (100,))
        val_dataset = TensorDataset(val_data, val_labels)


    # Create loaders
    train_loaders = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loaders = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loaders, val_loaders