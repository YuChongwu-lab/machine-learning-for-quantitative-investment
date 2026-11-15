
"""
Proceed + Diffusion data-augmentation integration module (with full validation).
═══════════════════════════════════════════════════════════════════════════
1. adds a train/validation split
2. validation set never uses data augmentation
3. early-stopping mechanism
4. saves and restores the best model
5. returns the full training history
6. monitors the validation-set DA metric


"""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Dict

from validation_framework import split_train_val
from proceed_diffusion_augmentation import ProceedDiffusionDataset


# ═══════════════════════════════════════════════════════════════════
# Core training function (with validation)
# ═══════════════════════════════════════════════════════════════════

def train_adapter_proceed_with_diffusion_validated(
    enc_src,
    enc_tgt,
    adapter,
    base_model,
    Xw_all,
    Yw_all,
    cfg,
    diffusion_augmentor,
    device,
    val_ratio: float = 0.2,
    save_best_path: Optional[str] = None
) -> Dict:
    """
    Proceed training with diffusion augmentation (with validation).
    Args:
        enc_src: source-domain concept encoder
        enc_tgt: target-domain concept encoder
        adapter: the Adapter module
        base_model: the PatchTST backbone
        Xw_all: all windowed data (N, T, F)
        Yw_all: all labels (N, P)
        cfg: config dict {'lr', 'batch_size', 'epochs', 'reg', 'synth_prob'}
        diffusion_augmentor: the diffusion augmentor
        device: training device
        val_ratio: validation-set ratio
        save_best_path: path to save the best model (optional)

    Returns:
        history: training history dict
            - 'train_loss': train loss per epoch
            - 'val_loss': validation loss per epoch
            - 'val_da': validation DA per epoch
            - 'best_epoch': the best epoch
            - 'best_val_loss': the best validation loss
            - 'best_val_da': the best validation DA
    """
    print("\n" + "="*70)
    print("Starting Adapter training (Proceed + Diffusion + Validation)")
    print("="*70)

    # =================== 1. Split the dataset ===================
    Xw_tr, Xw_val, Yw_tr, Yw_val = split_train_val(
        Xw_all, Yw_all,
        val_ratio=val_ratio,
        shuffle=False  # don't shuffle a time series
    )

    print(f"Training set size: {len(Xw_tr)}")
    print(f"Validation set size: {len(Xw_val)}")

    # =================== 2. Build the datasets ===================
    # Training set: with diffusion augmentation
    train_ds = ProceedDiffusionDataset(
        Xw_tr, Yw_tr,
        diffusion_augmentor=diffusion_augmentor,
        batch_size=cfg["batch_size"],
        augment_prob=cfg.get("synth_prob", 0.35),
        augment_ratio=1.0,
        use_concept_only=True
    )

    # Validation set: no augmentation
    val_ds = ProceedDiffusionDataset(
        Xw_val, Yw_val,
        diffusion_augmentor=None,  # no augmentation for validation
        batch_size=cfg["batch_size"],
        augment_prob=0.0,  # make sure augmentation is disabled
        augment_ratio=0.0,
        use_concept_only=True
    )

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)

    # =================== 3. Model and optimizer setup ===================
    enc_src.to(device)
    enc_tgt.to(device)
    adapter.to(device)
    base_model.to(device).eval()

    trainable_params = (
        list(enc_src.parameters()) +
        list(enc_tgt.parameters()) +
        list(adapter.parameters())
    )
    opt = torch.optim.Adam(trainable_params, lr=cfg["lr"])
    mse = nn.MSELoss()
    reg_w = cfg.get("reg", 1e-4)

    # =================== 4. Training-state tracking ===================
    best_val_loss = float('inf')
    best_val_da = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = cfg.get("patience", 10)

    best_state = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_da': []
    }

    # =================== 5. Define the train/validate functions ===================
    def train_epoch(dataloader, prev_Xb_buffer):
        """Train for one epoch."""
        enc_src.train()
        enc_tgt.train()
        adapter.train()
        base_model.eval()

        total_loss = 0.0
        nstep = 0

        for batch in dataloader:
            # handle the different batch formats
            if len(batch) == 3:  # a mixed batch (with augmentation)
                Xb_mixed, Yb, n_real = batch
                Xb_mixed = Xb_mixed.squeeze(0).to(device)
                Yb = Yb.squeeze(0).to(device)
                n_real = n_real.item()

                Xb_real = Xb_mixed[:n_real]
                Xb_all = Xb_mixed
            else:  # a plain batch
                Xb, Yb = batch
                Xb = Xb.squeeze(0).to(device)
                Yb = Yb.squeeze(0).to(device)
                Xb_real = Xb
                Xb_all = Xb

            # target concept (uses all data, including augmented)
            cj = enc_tgt(Xb_all).mean(dim=0, keepdim=True)

            # source concept (uses the buffer)
            if prev_Xb_buffer is None:
                prev_Xb_buffer = Xb_real[:min(len(Xb_real), 32)].detach().clone()

            with torch.no_grad():
                ci_avg = enc_src(prev_Xb_buffer).mean(dim=0, keepdim=True)

            ci = ci_avg.expand(len(Xb_real), -1)
            cj_expanded = cj.expand(len(Xb_real), -1)
            delta = cj_expanded - ci

            # Adapter
            mod = adapter(delta)

            # prediction (real data only)
            pred = base_model(Xb_real, mod=mod)

            # loss
            loss_pred = mse(pred.view(pred.size(0), -1), Yb.view(Yb.size(0), -1))

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

            loss = loss_pred + reg_w * reg

            # backward pass
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            # update the buffer
            prev_Xb_buffer = Xb_real[:min(len(Xb_real), 32)].detach().clone()

            total_loss += float(loss.item())
            nstep += 1

        avg_loss = total_loss / max(1, nstep)
        return avg_loss, prev_Xb_buffer

    def validate_epoch(dataloader, prev_Xb_buffer):
        """Validate for one epoch."""
        enc_src.eval()
        enc_tgt.eval()
        adapter.eval()
        base_model.eval()

        total_loss = 0.0
        nstep = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                # the validation set never uses augmentation, so the batch format is simple
                if len(batch) == 3:
                    Xb, Yb, _ = batch
                else:
                    Xb, Yb = batch

                Xb = Xb.squeeze(0).to(device)
                Yb = Yb.squeeze(0).to(device)

                # target concept
                cj = enc_tgt(Xb).mean(dim=0, keepdim=True)

                # source concept
                if prev_Xb_buffer is None:
                    prev_Xb_buffer = Xb[:min(len(Xb), 32)].detach().clone()

                ci_avg = enc_src(prev_Xb_buffer).mean(dim=0, keepdim=True)
                ci = ci_avg.expand(len(Xb), -1)
                cj_expanded = cj.expand(len(Xb), -1)
                delta = cj_expanded - ci

                # Adapter
                mod = adapter(delta)

                # prediction
                pred = base_model(Xb, mod=mod)

                # loss
                loss = mse(pred.view(pred.size(0), -1), Yb.view(Yb.size(0), -1))

                total_loss += float(loss.item())
                nstep += 1

                # collect predictions for computing DA
                all_preds.append(pred.cpu().numpy())
                all_targets.append(Yb.cpu().numpy())

                # update the buffer
                prev_Xb_buffer = Xb[:min(len(Xb), 32)].detach().clone()

        avg_loss = total_loss / max(1, nstep)

        # compute DA (direction accuracy)
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        val_da = np.mean(np.sign(all_preds) == np.sign(all_targets))

        return avg_loss, val_da

    # =================== 6. Training loop ===================
    prev_Xb = None

    for ep in range(cfg.get("epochs", 10)):
        # train
        train_loss, prev_Xb = train_epoch(train_dl, prev_Xb)
        history['train_loss'].append(train_loss)

        # validate
        val_loss, val_da = validate_epoch(val_dl, prev_Xb)
        history['val_loss'].append(val_loss)
        history['val_da'].append(val_da)

        # check for improvement
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_da = val_da
            best_epoch = ep
            epochs_no_improve = 0

            # save the best model state
            best_state = {
                'enc_src': copy.deepcopy(enc_src.state_dict()),
                'enc_tgt': copy.deepcopy(enc_tgt.state_dict()),
                'adapter': copy.deepcopy(adapter.state_dict())
            }

            # save to disk (if a path was given)
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path) if os.path.dirname(save_best_path) else '.', exist_ok=True)
                torch.save(best_state, save_best_path)

            print(f"Epoch {ep+1}/{cfg['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val DA: {val_da:.4f} ⭐")
        else:
            epochs_no_improve += 1
            print(f"Epoch {ep+1}/{cfg['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val DA: {val_da:.4f}")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n Early stopping at epoch {ep+1}")
            print(f"   No improvement for {patience} epochs")
            break

    # =================== 7. Restore the best model ===================
    if best_state is not None:
        enc_src.load_state_dict(best_state['enc_src'])
        enc_tgt.load_state_dict(best_state['enc_tgt'])
        adapter.load_state_dict(best_state['adapter'])
        print(f"\nRestored best model from epoch {best_epoch+1}")

    # =================== 8. Finalize the training history ===================
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    history['best_val_da'] = best_val_da

    print("\n" + "="*70)
    print(" Adapter training complete (validation version)")
    print("="*70)
    print(f"   Best epoch: {best_epoch+1}")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Best val DA: {best_val_da:.4f}")
    print("="*70 + "\n")

    return history


# ═══════════════════════════════════════════════════════════════════
# Convenience wrapper function
# ═══════════════════════════════════════════════════════════════════

def train_adapter_proceed_with_diffusion(
    enc_src, enc_tgt, adapter, base_model,
    Xw_all, Yw_all, cfg, diffusion_augmentor,
    device,
    use_validation=True,
    val_ratio=0.2,
    save_best_path=None
):
    """

    Args:
        use_validation: whether to use the validation version (default True)
        val_ratio: validation-set ratio
        save_best_path: path to save the best model

    """
    if use_validation:
        return train_adapter_proceed_with_diffusion_validated(
            enc_src, enc_tgt, adapter, base_model,
            Xw_all, Yw_all, cfg, diffusion_augmentor,
            device, val_ratio, save_best_path
        )
    else:
        print("The validation version is recommended! Using the simplified training path for now...")
        pass

