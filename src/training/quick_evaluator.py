import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Dict
import time
from tqdm import tqdm


class QuickEvaluator:
    """
    Quick evaluation of model
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.model.eval()
    

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model

        Args:
            data_loader: Data loade

        Returns:
            Dict containss metrics
        """

        correct = 0
        total = 0
        top5_correct = 0

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        start_time = time.time()

        for inputs, outputs in tqdm(data_loader, desc='Evaluating'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))

        inference_time = time.time() - start_time

        return {
            'loss': total_loss / len(data_loader),
            'accuracy': 100. * correct / total,
            'top5_accuracy': 100. * top5_correct / total,
            'inference_time': inference_time,
            'samples_per_second': total / inference_time
        }