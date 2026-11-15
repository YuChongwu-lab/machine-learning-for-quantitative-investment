"""Training loop for the final E_DyGFormer model (objective: Spearman-IC)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Data

from metrics import _compute_signal, _spearman_corr, mse_ic_loss


def train_enhanced_model(model: nn.Module, train_dataset: List[Data],
                         val_dataset: List[Data], epochs: int = 20,
                         batch_size: int = 8, lr: float = 0.001, weight_decay: float = 1e-4,
                         scheduler_factor: float = 0.7, scheduler_patience: int = 5,
                         grad_clip_norm: float = 1.0, risk_reg_weight: float = 1e-3,
                         loss_alpha: float = 1.0, loss_beta: float = 0.5,
                         val_every: int = 5,
                         checkpoint_path: str = None) -> nn.Module:
    name = getattr(model, "model_name", model.__class__.__name__)
    safe_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in name).lower()
    ckpt_path = checkpoint_path or f'best_{safe_name}.pth'

    print(f"Training the {name} model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=scheduler_patience, factor=scheduler_factor)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i + batch_size]
            optim.zero_grad()
            try:
                outputs = model(batch)
                preds = outputs['predictions']                      # [B, A]
                risks = outputs.get('risks', None)
                tgts  = torch.stack([d.y for d in batch]).to(device)     # [B, A]

                # combined loss
                loss, logs = mse_ic_loss(preds, tgts, risks, alpha=loss_alpha, beta=loss_beta)
                if risks is not None:
                    loss = loss + risk_reg_weight * risks.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optim.step()
                total_loss += float(loss.item())
            except Exception:
                continue

        if val_dataset and (epoch % val_every == 0):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(val_dataset), batch_size):
                    batch = val_dataset[i:i + batch_size]
                    try:
                        outputs = model(batch)
                        preds = outputs['predictions']
                        risks = outputs.get('risks', None)
                        if risks is not None: risks = risks.detach().to(device)  # <- same convention as training
                        tgts  = torch.stack([d.y for d in batch]).to(device)
                        s = _compute_signal(preds, risks)
                        corr = _spearman_corr(s, tgts, dim=1)
                        val_loss += float((-corr).mean().item())

                    except Exception:
                        continue
            avg_val = val_loss / max(len(val_dataset) // batch_size, 1)
            scheduler.step(avg_val)
            print(f"Epoch {epoch}: Train(-SpearmanIC)={total_loss:.6f}, Val(-SpearmanIC)={avg_val:.6f}")
            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), ckpt_path)

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception:
        pass

    print(f"{name} training complete")
    return model
