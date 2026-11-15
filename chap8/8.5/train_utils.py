"""Model training: ModelTrainer (one epoch of train/validate) and the outer training loop."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from model import HierarchicalGNN


class ModelTrainer:
    """Model trainer."""

    def __init__(self, model: HierarchicalGNN, device: str = 'cuda',
                 lr: float = 5e-4, weight_decay: float = 1e-3,
                 scheduler_factor: float = 0.7, scheduler_patience: int = 8,
                 batch_size: int = 16, grad_clip_norm: float = 0.5):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience
        )

    def train_epoch(self, train_dataset: List[Data]) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_size = self.batch_size

        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i + batch_size]

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            predictions = outputs['predictions']
            risks = outputs['risks']

            targets = torch.stack([data.y for data in batch]).to(self.device)

            mse_loss = F.mse_loss(predictions, targets)
            risk_loss = F.mse_loss(risks, torch.abs(targets))

            # Add a prediction-diversity regularizer
            pred_diversity = -torch.var(predictions.flatten())

            # Add L1 regularization to curb overfitting
            l1_reg = sum(p.abs().sum() for p in self.model.parameters())

            total_loss_batch = (mse_loss +
                                0.1 * risk_loss +
                                0.01 * pred_diversity +
                                1e-5 * l1_reg)

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            total_loss += total_loss_batch.item()

        return total_loss / len(range(0, len(train_dataset), batch_size))

    def validate(self, val_dataset: List[Data]) -> Dict:
        """Validation."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            batch_size = self.batch_size
            for i in range(0, len(val_dataset), batch_size):
                batch = val_dataset[i:i + batch_size]

                outputs = self.model(batch)
                predictions = outputs['predictions']
                targets = torch.stack([data.y for data in batch]).to(self.device)

                mse_loss = F.mse_loss(predictions, targets)
                total_loss += mse_loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute the IC
        ic_scores = []
        for i in range(min(len(all_predictions), 30)):
            pred = all_predictions[i]
            target = all_targets[i]

            mask = ~(np.isnan(pred) | np.isnan(target) | (np.abs(target) < 1e-6))
            if np.sum(mask) > 5:
                pred_filtered = pred[mask]
                target_filtered = target[mask]

                if np.std(pred_filtered) > 1e-6 and np.std(target_filtered) > 1e-6:
                    ic = np.corrcoef(pred_filtered, target_filtered)[0, 1]
                    if not np.isnan(ic) and abs(ic) < 0.99:
                        ic_scores.append(ic)

        mean_ic = np.mean(ic_scores) if len(ic_scores) > 0 else 0

        return {
            'loss': total_loss / len(range(0, len(val_dataset), batch_size)),
            'ic': mean_ic,
            'rank_ic': mean_ic,
            'ic_count': len(ic_scores)
        }


def train_model(model: HierarchicalGNN, train_dataset: List[Data],
                val_dataset: List[Data], epochs: int = 50, patience: int = 20,
                lr: float = 5e-4, weight_decay: float = 1e-3,
                scheduler_factor: float = 0.7, scheduler_patience: int = 8,
                batch_size: int = 16, grad_clip_norm: float = 0.5,
                checkpoint_path: str = 'best_hierarchical_model.pth') -> HierarchicalGNN:
    """Train the model."""
    print(f"Starting training of the hierarchical multi-edge-type model, {epochs} epochs...")

    trainer = ModelTrainer(model, lr=lr, weight_decay=weight_decay,
                            scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience,
                            batch_size=batch_size, grad_clip_norm=grad_clip_norm)
    best_ic = -float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_dataset)
        val_metrics = trainer.validate(val_dataset)
        val_ic = val_metrics['ic']

        trainer.scheduler.step(val_ic)

        if val_ic > best_ic:
            best_ic = val_ic
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val IC = {val_ic:.4f}, '
                  f'IC Count = {val_metrics["ic_count"]}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load(checkpoint_path))
    print("Hierarchical model training complete!")
    return model
