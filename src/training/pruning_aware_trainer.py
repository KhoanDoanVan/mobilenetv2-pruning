import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Tuple
import time
from tqdm import tqdm


class PruningAwareTrainer:
    """
    Trainer for fine-tuning model pruned

    Special:
        - Maintain pruning masks in training
        - Learning rate smaller
        - Early stopping
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cuda',
            learning_rate: float = 0.0001,
            optimizer_name: str = 'adam',
            masks: Dict = None
    ):
        """
        Args:
            model: model pruned
            train_loader: training data loader
            val_loader: validation data loader
            device: 'cuda' or 'cpu'
            learning_rate: learning rate (almost small to fine-tuning)
            optimizer_name: 'adam', 'sgd' or 'adamw'
            masks: dict of pruning masks (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.masks = masks

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50
        )

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }


    def train(
            self,
            num_epochs: int,
            save_best: bool = True,
            save_path: str = None,
            early_stopping_patience: int = 5
    ) -> Dict:
        """
        Args:
            early_stopping_patience: num of epochs to early stop

        Returns:
            history dict
        """
        best_val_acc = 0.0
        patience_counter = 0

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Scheduler step
            self.scheduler.step()

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'history': self.history
                    }, save_path)
                    print(f"====> Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

        return self.history



    def apply_masks(self):
        """
        Apply pruning masks to model weights
        Make sure pruned weights still 0 after each gradient update
        """
        if self.masks is None:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name].to(self.device)


    def train_epoch(self) -> Tuple[float, float]:
        """
        Train an epoch

        Return:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Apply masks for maintain sparsity
            self.apply_masks()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy
    

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validation on validation set

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(self.val_loader, desc='Validation'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy