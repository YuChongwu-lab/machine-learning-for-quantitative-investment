
"""
Training framework -- includes validation, early stopping, and best-model selection.
═══════════════════════════════════════════════════════════════════════════
Core functionality:
1. The Trainer class -- automatically handles validation, early stopping, and best-model saving
2. AugmentationEvaluator -- evaluates data-augmentation configurations
3. Full training monitoring and visualization
"""

import copy
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score


# =================== 1. The Trainer class ===================

class Trainer:
    """
    A general-purpose trainer -- automatically handles validation, early
    stopping, and best-model selection.
    Core functionality:
    - automatically splits into train/validation sets
    - evaluates on the validation set every epoch
    - early stopping (stops if no improvement for N epochs)
    - saves the model with the lowest validation loss
    - records the full training history
    - supports a custom evaluation metric

    """

    def __init__(self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        patience: int = 10,
        min_delta: float = 1e-4,
        metric_fn: Optional[Callable] = None,
        metric_mode: str = 'max',  # 'max' for DA/R², 'min' for loss
        verbose: bool = True
    ):
        """
        Args:
            model: the model to train
            loss_fn: loss function
            optimizer: optimizer
            device: device ('cuda' or 'cpu')
            patience: early-stopping patience
            min_delta: the minimum change counted as an "improvement"
            metric_fn: an optional evaluation-metric function (y_pred, y_true) -> float
            metric_mode: 'max' means higher is better, 'min' means lower is better
            verbose: whether to print training progress
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.metric_fn = metric_fn
        self.metric_mode = metric_mode
        self.verbose = verbose

        # training state
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf') if metric_mode == 'max' else float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.epochs_no_improve = 0

        # training history
        self.history = defaultdict(list)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        eval_every: int = 1,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the model.
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: maximum number of training epochs
            eval_every: evaluate every this many epochs
            save_path: path to save the best model (optional)
        Returns:
            history: a dict containing the training history
                - 'train_loss': train loss per epoch
                - 'val_loss': validation loss per epoch
                - 'val_metric': validation metric per epoch (if provided)
                - 'best_epoch': the best epoch
                - 'best_val_loss': the best validation loss
                - 'stopped_early': whether training stopped early
        """
        for epoch in range(epochs):
            # train
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # validate
            if (epoch + 1) % eval_every == 0:
                val_loss, val_metric = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                if val_metric is not None:
                    self.history['val_metric'].append(val_metric)

                # check for improvement
                improved = self._check_improvement(val_loss, val_metric)

                if improved:
                    self.best_val_loss = val_loss
                    if val_metric is not None:
                        self.best_val_metric = val_metric
                    self.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.epochs_no_improve = 0

                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        if self.verbose:
                            print(f" Saved best model to {save_path}")
                else:
                    self.epochs_no_improve += 1

                # print progress
                if self.verbose:
                    self._print_progress(epoch, epochs, train_loss, val_loss, val_metric)

                # early stopping
                if self.epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"\n  Early stopping at epoch {epoch+1}")
                        print(f"   Best epoch: {self.best_epoch+1}")
                        print(f"   Best val_loss: {self.best_val_loss:.6f}")
                        if val_metric is not None:
                            print(f"   Best val_metric: {self.best_val_metric:.6f}")
                    self.history['stopped_early'] = True
                    break

        # restore the best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"\n Restored best model from epoch {self.best_epoch+1}")

        # finalize the history
        self.history['best_epoch'] = self.best_epoch
        self.history['best_val_loss'] = self.best_val_loss
        if self.metric_fn is not None:
            self.history['best_val_metric'] = self.best_val_metric

        return dict(self.history)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # forward pass
            pred = self.model(xb)

            # make sure the shapes match
            if pred.dim() > 2:
                pred = pred.view(pred.size(0), -1)
            if yb.dim() > 2:
                yb = yb.view(yb.size(0), -1)

            loss = self.loss_fn(pred, yb)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)

        return total_loss / max(1, count)

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Optional[float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = self.model(xb)

                # make sure the shapes match
                if pred.dim() > 2:
                    pred = pred.view(pred.size(0), -1)
                if yb.dim() > 2:
                    yb = yb.view(yb.size(0), -1)

                loss = self.loss_fn(pred, yb)

                total_loss += loss.item() * xb.size(0)
                count += xb.size(0)

                # collect predictions for computing the metric
                if self.metric_fn is not None:
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(yb.cpu().numpy())

        val_loss = total_loss / max(1, count)

        # compute the metric
        val_metric = None
        if self.metric_fn is not None and len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            val_metric = self.metric_fn(all_preds, all_targets)

        return val_loss, val_metric

    def _check_improvement(self, val_loss: float, val_metric: Optional[float]) -> bool:
        """Check whether there was an improvement."""
        # primarily based on validation loss
        loss_improved = val_loss < (self.best_val_loss - self.min_delta)

        # also consider the metric, if available
        metric_improved = False
        if val_metric is not None:
            if self.metric_mode == 'max':
                metric_improved = val_metric > (self.best_val_metric + self.min_delta)
            else:
                metric_improved = val_metric < (self.best_val_metric - self.min_delta)

        # loss is the primary criterion, the metric is a secondary reference
        return loss_improved or (val_metric is not None and metric_improved)

    def _print_progress(self, epoch: int, total_epochs: int,
                       train_loss: float, val_loss: float,
                       val_metric: Optional[float]):
        """Print training progress."""
        msg = f"Epoch {epoch+1}/{total_epochs} | "
        msg += f"Train Loss: {train_loss:.6f} | "
        msg += f"Val Loss: {val_loss:.6f}"

        if val_metric is not None:
            msg += f" | Val Metric: {val_metric:.6f}"

        if val_loss < self.best_val_loss:
            msg += " ⭐"

        print(msg)


# =================== 2. The data-augmentation evaluator ===================

class AugmentationEvaluator:
    """
    Evaluator for data-augmentation configurations.

    Functionality:
    - compares the effect of different data-augmentation configurations
    - automatically selects the best configuration
    - generates an evaluation report

    """

    def __init__(
        self,
        model_fn: Callable,  # a function that returns a new model instance
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        base_config: Dict,
        loss_fn: nn.Module,
        device: str = 'cuda'
    ):
        """
        Args:
            model_fn: a function that returns a new model instance
            train_data: (X_train, Y_train)
            val_data: (X_val, Y_val)
            base_config: the base training config
            loss_fn: loss function
            device: device
        """
        self.model_fn = model_fn
        self.X_train, self.Y_train = train_data
        self.X_val, self.Y_val = val_data
        self.base_config = base_config
        self.loss_fn = loss_fn
        self.device = device

        self.results = []

    def compare_configs(
        self,
        augmentation_configs: List[Tuple[Dict, str]],
        n_runs: int = 3
    ) -> pd.DataFrame:
        """
        Compare multiple data-augmentation configurations.

        Args:
            augmentation_configs: [(config_dict, name), ...]
            n_runs: how many times to run each configuration (then average)

        Returns:
            results_df: a DataFrame with results for every configuration
        """
        print("\n" + "="*70)
        print(" Data-augmentation configuration evaluation")
        print("="*70)

        for aug_config, config_name in augmentation_configs:
            print(f"\n Testing configuration: {config_name}")
            print(f"   Parameters: {aug_config}")

            metrics_runs = []

            for run in range(n_runs):
                print(f"   Run {run+1}/{n_runs}...", end=" ")

                # build a new model
                model = self.model_fn().to(self.device)

                # build the data loaders (applying augmentation per the config)
                train_loader = self._create_dataloader(
                    self.X_train, self.Y_train,
                    aug_config,
                    shuffle=True
                )
                val_loader = self._create_dataloader(
                    self.X_val, self.Y_val,
                    {'enable': False},  # no augmentation for the validation set
                    shuffle=False
                )

                # train
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.base_config.get('lr', 1e-3)
                )

                trainer = Trainer(
                    model=model,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    device=self.device,
                    patience=self.base_config.get('patience', 10),
                    metric_fn=self._compute_da,
                    metric_mode='max',
                    verbose=False
                )

                history = trainer.fit(
                    train_loader,
                    val_loader,
                    epochs=self.base_config.get('epochs', 20)
                )

                # record the metrics
                metrics = {
                    'val_loss': history['best_val_loss'],
                    'val_da': history.get('best_val_metric', 0),
                    'best_epoch': history['best_epoch'],
                    'stopped_early': history.get('stopped_early', False)
                }
                metrics_runs.append(metrics)

                print(f"Val Loss: {metrics['val_loss']:.6f}, DA: {metrics['val_da']:.4f}")

            # average the metrics
            avg_metrics = {
                'config_name': config_name,
                'config': str(aug_config),
                'val_loss_mean': np.mean([m['val_loss'] for m in metrics_runs]),
                'val_loss_std': np.std([m['val_loss'] for m in metrics_runs]),
                'val_da_mean': np.mean([m['val_da'] for m in metrics_runs]),
                'val_da_std': np.std([m['val_da'] for m in metrics_runs]),
                'avg_best_epoch': np.mean([m['best_epoch'] for m in metrics_runs]),
                'early_stop_rate': np.mean([m['stopped_early'] for m in metrics_runs])
            }

            self.results.append(avg_metrics)

            print(f"    Average: Val Loss={avg_metrics['val_loss_mean']:.6f}±{avg_metrics['val_loss_std']:.6f}, "
                  f"DA={avg_metrics['val_da_mean']:.4f}±{avg_metrics['val_da_std']:.4f}")

        # build the results table
        results_df = pd.DataFrame(self.results)

        print("\n" + "="*70)
        print(" Evaluation summary")
        print("="*70)
        print(results_df[['config_name', 'val_loss_mean', 'val_da_mean', 'avg_best_epoch']].to_string(index=False))

        return results_df

    def get_best_config(self, metric: str = 'val_da_mean') -> Dict:
        """
        Return the best configuration.

        Args:
            metric: the metric to select on ('val_loss_mean' or 'val_da_mean')

        Returns:
            best_config: a dict with the best configuration
        """
        if not self.results:
            raise ValueError("call compare_configs() first")

        results_df = pd.DataFrame(self.results)

        if 'loss' in metric:
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()

        best_result = results_df.iloc[best_idx]

        print(f"\n Best configuration (by {metric}):")
        print(f"   Name: {best_result['config_name']}")
        print(f"   Config: {best_result['config']}")
        print(f"   Val Loss: {best_result['val_loss_mean']:.6f}±{best_result['val_loss_std']:.6f}")
        print(f"   Val DA: {best_result['val_da_mean']:.4f}±{best_result['val_da_std']:.4f}")

        return best_result.to_dict()

    def _create_dataloader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        aug_config: Dict,
        shuffle: bool
    ) -> DataLoader:
        """Build a data loader."""
        # simplified here; adapt to your actual augmentation implementation as needed
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32)
        )

        return DataLoader(
            dataset,
            batch_size=self.base_config.get('batch_size', 64),
            shuffle=shuffle,
            drop_last=False
        )

    def _compute_da(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute direction accuracy."""
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        return np.mean(np.sign(y_pred_flat) == np.sign(y_true_flat))


# =================== 3. Helper functions ===================

def split_train_val(X, Y, val_ratio=0.2, shuffle=False):
    """
    Split into training and validation sets.
    Args:
        X: feature array
        Y: label array
        val_ratio: validation-set ratio
        shuffle: whether to shuffle (a time series usually isn't)

    Returns:
        X_train, X_val, Y_train, Y_val
    """
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    if shuffle:
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        # time series: first 80% for training, last 20% for validation
        train_indices = np.arange(n_train)
        val_indices = np.arange(n_train, n_samples)

    X_train = X[train_indices]
    X_val = X[val_indices]
    Y_train = Y[train_indices]
    Y_val = Y[val_indices]

    return X_train, X_val, Y_train, Y_val


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Compute several evaluation metrics.

    Args:
        y_pred: predicted values
        y_true: true values

    Returns:
        metrics: {'da': ..., 'r2': ..., 'mae': ..., 'rmse': ...}
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    da = np.mean(np.sign(y_pred_flat) == np.sign(y_true_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))

    return {
        'da': da,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


