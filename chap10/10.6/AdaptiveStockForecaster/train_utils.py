
import copy
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from validation_framework import Trainer, split_train_val


# =================== Core function 1: pretrain the backbone (with validation) ===================

def pretrain_backbone_with_validation(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cfg: Dict,
    loss_fn: nn.Module,
    device: str = 'cuda',
    val_ratio: float = 0.2,
    save_path: Optional[str] = None
) -> Dict:
    """
    Pretrain the PatchTST backbone network (with a validation set and early stopping).
    Args:
        model: the PatchTST model
        X_train: training data (N, seq_len, feature_dim)
        Y_train: training labels (N, pred_len)
        cfg: config dict {'lr': ..., 'batch_size': ..., 'epochs': ...}
        loss_fn: loss function (DirectionalLoss or MSE)
        device: device
        val_ratio: validation-set ratio
        save_path: path to save the best model

    Returns:
        history: training history
            - 'train_loss': train loss per epoch
            - 'val_loss': validation loss per epoch
            - 'val_da': validation DA per epoch
            - 'best_epoch': best epoch
            - 'best_val_loss': best validation loss
    """
    print("\n" + "="*70)
    print("Starting pretraining (with validation)")
    print("="*70)

    # 1. Split into train/validation sets
    X_tr, X_val, Y_tr, Y_val = split_train_val(
        X_train, Y_train,
        val_ratio=val_ratio,
        shuffle=False  # don't shuffle a time series
    )

    print(f"Training set size: {len(X_tr)}")
    print(f"Validation set size: {len(X_val)}")

    # 2. Build the data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(Y_tr, dtype=torch.float32)
        ),
        batch_size=cfg.get('batch_size', 64),
        shuffle=True,
        drop_last=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32)
        ),
        batch_size=cfg.get('batch_size', 64),
        shuffle=False,
        drop_last=False
    )

    # 3. Build the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))

    # 4. Define the evaluation metric (DA)
    def compute_da(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute direction accuracy."""
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        return np.mean(np.sign(y_pred_flat) == np.sign(y_true_flat))

    # 5. Build the Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patience=cfg.get('patience', 10),  # stop if no improvement for 10 epochs
        min_delta=cfg.get('min_delta', 1e-4),
        metric_fn=compute_da,  # monitor DA
        metric_mode='max',  # higher DA is better
        verbose=True
    )

    # 6. Train
    if save_path is None:
        save_path = "checkpoints/best_pretrain_model.pth"

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.get('epochs', 100),  # max number of epochs
        save_path=save_path
    )

    # 7. Set the model to eval mode and freeze its parameters
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("\nPretraining complete!")
    print(f"Best epoch: {history['best_epoch']+1}")
    print(f"Best val loss: {history['best_val_loss']:.6f}")
    if 'best_val_metric' in history:
        print(f"Best val DA: {history['best_val_metric']:.4f}")

    return history


# =================== Core function 2: train the Adapter (with validation) ===================

def train_adapter_with_validation(
    enc_src: nn.Module,
    enc_tgt: nn.Module,
    adapter: nn.Module,
    base_model: nn.Module,
    X_train: np.ndarray,  # originally Xw_all
    Y_train: np.ndarray,  # originally Yw_all
    cfg: Dict,
    loss_fn: nn.Module,
    device: str = 'cuda',
    val_ratio: float = 0.2,
    use_diffusion: bool = False,
    diffusion_augmentor = None
) -> Dict:
    """
    Train the Adapter (with a validation set and early stopping).
    1. split into train/validation sets
    2. no data augmentation
    3. early stopping
    4. returns the full training history
    Args:
        enc_src, enc_tgt: the concept encoders
        adapter: the Adapter module
        base_model: the PatchTST backbone
        X_train, Y_train: training data and labels
        cfg: config dict
        loss_fn: loss function
        device: device
        val_ratio: validation-set ratio
        use_diffusion: whether to use diffusion augmentation
        diffusion_augmentor: the diffusion augmentor

    Returns:
        history: training history
    """
    print("\n" + "="*70)
    print("Starting Adapter training (with validation)")
    print("="*70)

    # 1. Split into train/validation sets
    X_tr, X_val, Y_tr, Y_val = split_train_val(X_train, Y_train, val_ratio=val_ratio, shuffle=False)
    print(f"Training set size: {len(X_tr)}")
    print(f"Validation set size: {len(X_val)}")

    # 2. Move models to the device
    enc_src.to(device)
    enc_tgt.to(device)
    adapter.to(device)
    base_model.to(device).eval()

    # 3. Set the trainable parameters
    trainable_params = (
        list(enc_src.parameters()) +
        list(enc_tgt.parameters()) +
        list(adapter.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.get('lr', 5e-4))

    # 4. Build the training data loaders (no data augmentation)
    # training-set loader
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)),
        batch_size=cfg.get('batch_size', 64),
        shuffle=True,
        drop_last=False
    )

    # validation-set loader
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32) ),
        batch_size=cfg.get('batch_size', 64),
        shuffle=False,
        drop_last=False
    )

    # 5. Define a custom trainer (a special forward pass is needed here)
    class AdapterTrainer:
        def __init__(self, enc_src, enc_tgt, adapter, base_model, loss_fn, optimizer, device, reg_w=1e-4):
            self.enc_src = enc_src
            self.enc_tgt = enc_tgt
            self.adapter = adapter
            self.base_model = base_model
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.device = device
            self.reg_w = reg_w

            self.best_val_loss = float('inf')
            self.best_epoch = 0
            self.best_state = None
            self.epochs_no_improve = 0
            self.history = {'train_loss': [], 'val_loss': [], 'val_da': []}

        def train_epoch(self, train_loader):
            """Train for one epoch."""
            self.enc_src.train()
            self.enc_tgt.train()
            self.adapter.train()
            self.base_model.eval()

            total_loss = 0.0
            count = 0

            prev_Xb = None

            for Xb, Yb in train_loader:
                Xb = Xb.to(self.device)
                Yb = Yb.to(self.device)

                # target-domain encoding
                cj = self.enc_tgt(Xb)

                # source-domain encoding (uses the average of the previous batch)
                if prev_Xb is None:
                    prev_Xb = Xb.detach()

                with torch.no_grad():
                    ci_avg = self.enc_src(prev_Xb).mean(dim=0, keepdim=True)
                ci = ci_avg.expand_as(cj)


                # Adapter modulation
                delta = cj - ci
                mod = self.adapter(delta)

                # prediction
                pred = self.base_model(Xb, mod=mod)

                # loss
                loss_pred = self.loss_fn(
                    pred.view(pred.size(0), -1),
                    Yb.view(Yb.size(0), -1)
                )

                # regularization
                reg = 0.0
                for k, t in mod["patch_embed"].items():
                    reg = reg + t.pow(2).mean()
                for g in mod["enc"]:
                    for k, t in g.items():
                        reg = reg + t.pow(2).mean()
                for k, t in mod["head_in"].items():
                    reg = reg + t.pow(2).mean()
                for k, t in mod["fc_out"].items():
                    reg = reg + t.pow(2).mean()

                loss = loss_pred + self.reg_w * reg

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.enc_src.parameters()) +
                    list(self.enc_tgt.parameters()) +
                    list(self.adapter.parameters()),
                    1.0
                )
                self.optimizer.step()

                prev_Xb = Xb.detach()

                total_loss += loss.item() * Xb.size(0)
                count += Xb.size(0)

            return total_loss / max(1, count)

        def validate_epoch(self, val_loader):
            """Validate for one epoch."""
            self.enc_src.eval()
            self.enc_tgt.eval()
            self.adapter.eval()
            self.base_model.eval()

            total_loss = 0.0
            count = 0
            all_preds = []
            all_targets = []

            prev_Xb = None

            with torch.no_grad():
                for Xb, Yb in val_loader:
                    Xb = Xb.to(self.device)
                    Yb = Yb.to(self.device)

                    # same forward logic as training
                    cj = self.enc_tgt(Xb)

                    if prev_Xb is None:
                        prev_Xb = Xb

                    # ci uses the mean
                    ci_avg = self.enc_src(prev_Xb).mean(dim=0, keepdim=True)
                    ci = ci_avg.expand_as(cj)

                    delta = cj - ci
                    mod = self.adapter(delta)

                    pred = self.base_model(Xb, mod=mod)

                    loss = self.loss_fn(
                        pred.view(pred.size(0), -1),
                        Yb.view(Yb.size(0), -1)
                    )

                    total_loss += loss.item() * Xb.size(0)
                    count += Xb.size(0)

                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(Yb.cpu().numpy())

                    prev_Xb = Xb

            val_loss = total_loss / max(1, count)

            # compute DA
            all_preds = np.concatenate(all_preds, axis=0).flatten()
            all_targets = np.concatenate(all_targets, axis=0).flatten()
            val_da = np.mean(np.sign(all_preds) == np.sign(all_targets))

            return val_loss, val_da

        def fit(self, train_loader, val_loader, epochs, patience=10):
            """The full training loop."""
            for epoch in range(epochs):
                # train
                train_loss = self.train_epoch(train_loader)
                self.history['train_loss'].append(train_loss)

                # validate
                val_loss, val_da = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_da'].append(val_da)

                # check for improvement
                if val_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state = {
                        'enc_src': copy.deepcopy(self.enc_src.state_dict()),
                        'enc_tgt': copy.deepcopy(self.enc_tgt.state_dict()),
                        'adapter': copy.deepcopy(self.adapter.state_dict())
                    }
                    self.epochs_no_improve = 0
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | Val DA: {val_da:.4f} ⭐")
                else:
                    self.epochs_no_improve += 1
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | Val DA: {val_da:.4f}")

                # early stopping
                if self.epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            # restore the best model
            if self.best_state is not None:
                self.enc_src.load_state_dict(self.best_state['enc_src'])
                self.enc_tgt.load_state_dict(self.best_state['enc_tgt'])
                self.adapter.load_state_dict(self.best_state['adapter'])
                print(f"\n Restored best model from epoch {self.best_epoch+1}")

            self.history['best_epoch'] = self.best_epoch
            self.history['best_val_loss'] = self.best_val_loss

            return self.history

    # 6. Train
    adapter_trainer = AdapterTrainer(
        enc_src, enc_tgt, adapter, base_model,
        loss_fn, optimizer, device,
        reg_w=cfg.get('reg', 1e-4)
    )

    history = adapter_trainer.fit(
        train_loader, val_loader,
        epochs=cfg.get('epochs', 10),
        patience=cfg.get('patience', 10)
    )

    print("\nAdapter training complete!")
    print(f"Best epoch: {history['best_epoch']+1}")
    print(f"Best val loss: {history['best_val_loss']:.6f}")

    return history


# =================== Core function 3: online fine-tuning (with validation) ===================

def online_finetune_with_validation(
    online_model: nn.Module,
    params: List,
    X_ft: np.ndarray,
    Y_ft: np.ndarray,
    loss_fn: nn.Module,
    lr: float = 1e-4,
    steps: int = 1,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    min_samples: int = 10,
    device: str = 'cuda'
) -> bool:
    """
    Online fine-tuning (with a validation check).
    Steps:
    1. save the model state before fine-tuning
    2. split into train/validation sets
    3. after fine-tuning, check on the validation set whether it actually improved
    4. only accept the fine-tuning update if it improved; otherwise roll back

    Args:
        online_model: the online model (the one being fine-tuned)
        params: the model's trainable parameters
        X_ft, Y_ft: fine-tuning data
        loss_fn: loss function
        lr: learning rate
        steps: number of fine-tuning steps
        batch_size: batch size
        val_ratio: validation-set ratio
        min_samples: minimum sample count; skip fine-tuning if fewer samples than this
        device: device

    Returns:
        accepted: whether the fine-tuning update was accepted (True = improved, False = rolled back)
    """
    if len(X_ft) < min_samples:
        return False  # too few samples, skip fine-tuning

    # 1. Save the original model state
    original_state = copy.deepcopy(online_model.state_dict())

    # 2. Split into train/validation sets
    if len(X_ft) > min_samples * 2:
        X_tr, X_val, Y_tr, Y_val = split_train_val(
            X_ft, Y_ft,
            val_ratio=val_ratio,
            shuffle=False
        )
    else:
        # too few samples to hold out a validation set; use all the data directly
        X_tr, Y_tr = X_ft, Y_ft
        X_val, Y_val = None, None

    # 3. Compute the pre-fine-tuning validation loss (if a validation set exists)
    if X_val is not None:
        online_model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)
            pred_before = online_model(X_val_tensor)
            loss_before = loss_fn(
                pred_before.view(pred_before.size(0), -1),
                Y_val_tensor.view(Y_val_tensor.size(0), -1)
            ).item()
    else:
        loss_before = None

    # 4. Fine-tune
    optimizer = torch.optim.Adam(params, lr=lr)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(Y_tr, dtype=torch.float32)
        ),
        batch_size=min(batch_size, len(X_tr)),
        shuffle=True,
        drop_last=False
    )

    online_model.train()
    for _ in range(steps):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = online_model(xb)
            loss = loss_fn(
                pred.view(pred.size(0), -1),
                yb.view(yb.size(0), -1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 5. Compute the post-fine-tuning validation loss
    if X_val is not None:
        online_model.eval()
        with torch.no_grad():
            pred_after = online_model(X_val_tensor)
            loss_after = loss_fn(
                pred_after.view(pred_after.size(0), -1),
                Y_val_tensor.view(Y_val_tensor.size(0), -1)
            ).item()

        # 6. Decide whether it improved
        if loss_after < loss_before:
            # improved, accept the fine-tuning update

            return True
        else:
            # no improvement, roll back
            online_model.load_state_dict(original_state)

            return False
    else:
        # no validation set, accept unconditionally
        return True


