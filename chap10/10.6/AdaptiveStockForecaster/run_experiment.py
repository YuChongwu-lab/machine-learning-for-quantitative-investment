"""
Distribution-drift-adaptive stock-price forecasting system -- experiment entry point.
═══════════════════════════════════════════════════════════════════════════
Function: integrates the full forecasting system.
Notes:
1. seq_len must be a multiple of 4.
2. Online fine-tuning uses DirectionalLoss.
"""

import os, copy, collections
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =================== Import custom modules ===================
from TCVAE import TCVAEConceptEncoder
from losses import DirectionalLoss
from proceed_diffusion_augmentation import setup_diffusion_augmentor

from train_utils import (
    pretrain_backbone_with_validation,
    train_adapter_with_validation,
    online_finetune_with_validation
)
from proceed_diffusion_augmentation_with_validation import train_adapter_proceed_with_diffusion_validated

from data import load_real_data, make_windows_xy, set_seed
from models import PatchTSTBackbone, ProceedAdapter

# ------------------- Base config -------------------
from config import BASE_CFG
CFG = copy.deepcopy(BASE_CFG)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(CFG["diffusion"]["save_dir"], exist_ok=True)

set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== Online fine-tuning ==============
def build_online_model(base_model: PatchTSTBackbone, finetune_layers: List[str]):
    online_model = copy.deepcopy(base_model).to(DEVICE)
    for p in online_model.parameters(): p.requires_grad = False
    name2layer = {
        "head_in": online_model.head_in,
        "fc_out": online_model.fc_out,
    }
    params = []
    for k in finetune_layers:
        if k in name2layer:
            for p in name2layer[k].parameters():
                p.requires_grad = True
                params.append(p)
    return online_model, params


def online_finetune_step(online_model, params, X_ft, Y_ft, lr=1e-4, steps=1, batch_size=8):
    """Online fine-tuning (uses DirectionalLoss)."""
    if len(X_ft) == 0:
        return

    opt = torch.optim.Adam(params, lr=lr)

    # Use DirectionalLoss
    loss_cfg = CFG.get("loss", {})
    loss_fn = DirectionalLoss(
        alpha=loss_cfg.get("alpha", 1.0),
        beta=loss_cfg.get("beta", 2.0),
        use_soft=loss_cfg.get("use_soft", True)
    )

    ds = TensorDataset(torch.tensor(X_ft), torch.tensor(Y_ft))
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    online_model.train()
    for _ in range(steps):
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = online_model(xb)
            loss = loss_fn(pred.view(pred.size(0), -1), yb.view(yb.size(0), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    online_model.eval()


# ============== Run a single stock ==============
def run_stock(df_one, stock_id, cfg):
    """
    Run the full pipeline for a single stock.
    Args:
        df_one: the DataFrame for a single stock
        stock_id: the stock ID
        cfg: config dict
    Returns:
        metrics_dict: a dict with all metrics and training histories
    """
    # =================== Initialize history variables (avoids UnboundLocalError) ===================
    history_pretrain = {'best_epoch': 0, 'best_val_loss': 0.0, 'best_val_da': 0.0}
    history_adapter = {'best_epoch': 0, 'best_val_loss': 0.0, 'best_val_da': 0.0}

    print("\n" + "=" * 80)
    print(f"Processing stock: {stock_id}")
    print("=" * 80)

    seq_len = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    # =================== 1. Data preprocessing ===================
    print("\nStage 1: Data preprocessing")
    print("-" * 80)

    # Feature engineering
    feature_cols = [c for c in df_one.columns if c not in ["trade_date", "ts_code", "close"]]
    cont_cols = [c for c in feature_cols if df_one[c].nunique() > 20]#treated as continuous if it has more than 20 distinct values
    disc_cols = [c for c in feature_cols if c not in cont_cols]#everything else is treated as discrete

    X_cont=df_one[cont_cols] #continuous features keep their raw values
    # Discrete features: check for and filter out constants
    if len(disc_cols) > 0:
        disc_data = []
        for col in disc_cols:
            if df_one[col].nunique() > 1:
                disc_data.append(df_one[col].fillna(0).values.reshape(-1, 1))
        X_disc = np.concatenate(disc_data, axis=1) if disc_data else np.zeros((len(df_one), 0))
    else:
        X_disc = np.zeros((len(df_one), 0))

    X_all = np.concatenate([X_cont, X_disc], axis=1)
    y_full = np.log(df_one["close"]).diff().shift(-1).fillna(0).values

    # Train/test split
    n = len(X_all)
    if n < (seq_len + pred_len + 5):
        raise ValueError("Not enough data to build any windows.")
    n_train = int(n * 0.8)

    # Standardize
    mu = X_all[:n_train].mean(axis=0, keepdims=True)
    std = X_all[:n_train].std(axis=0, keepdims=True) + 1e-6
    X_all = (X_all - mu) / std

    X_train_all = X_all[:n_train]
    y_train_all = y_full[:n_train]
    X_test_all = X_all[n_train:]
    y_test_all = y_full[n_train:]

    # Build time windows
    Xw_tr, Yw_tr = make_windows_xy(X_train_all, y_train_all, seq_len, pred_len)
    Xw_test, Yw_test = make_windows_xy(X_test_all, y_test_all, seq_len, pred_len)

    if len(Xw_tr) == 0 or len(Xw_test) == 0:
        print("Not enough data for windows.")
        return None

    feature_dim = Xw_tr.shape[-1]

    print(f"Data preparation complete")
    print(f"Training set: {len(Xw_tr)} samples")
    print(f"Test set: {len(Xw_test)} samples")
    print(f"Feature dimension: {feature_dim}")

    # =================== 2. Build the model ===================
    print("Stage 2: Build the model")
    print("-" * 80)

    # Build the loss function
    loss_fn = nn.MSELoss()

    # Build the PatchTST backbone
    P = cfg["patchtst"]
    base_model = PatchTSTBackbone(
        seq_len=seq_len, feature_dim=feature_dim, pred_len=pred_len,
        d_model=P["d_model"], nhead=P["nhead"], nlayers=P["nlayers"],
        dropout=P["dropout"], patch_len=P["patch_len"], stride=P["stride"],
        head_hidden=P["head_hidden"]
    ).to(DEVICE)

    print(f"PatchTST backbone built successfully")
    print(f"Parameter count: {sum(p.numel() for p in base_model.parameters()):,}")

    # =================== 3. Pretrain the PatchTST backbone (with validation) ===================
    print("Stage 3: Pretrain the PatchTST backbone (with validation)")
    print("-" * 80)

    pretrain_save_path = f'checkpoints/best_pretrain_{stock_id}.pth'

    try:
        history_pretrain = pretrain_backbone_with_validation(
            model=base_model,
            X_train=Xw_tr,
            Y_train=Yw_tr,
            cfg=cfg["pretrain"],
            loss_fn=loss_fn,
            device=DEVICE,
            val_ratio=0.2,  # 20% held out for validation
            save_path=pretrain_save_path
        )

        print("Pretraining complete!")
        print(f"Best epoch: {history_pretrain['best_epoch'] + 1}")
        print(f"Best val loss: {history_pretrain['best_val_loss']:.6f}")
        if 'best_val_metric' in history_pretrain:
            print(f"Best val DA: {history_pretrain['best_val_metric']:.4f}")

    except Exception as e:
        print(f"\nPretraining failed: {e}")
        import traceback
        traceback.print_exc()
        history_pretrain = None

    # =================== 4. Train the concept encoders and adapter (with validation) ====================
    print("Stage 4: Train the concept encoders and adapter (with validation)")
    print("-" * 80)

    A = cfg["adapter_train"]

    tcvae_cfg = cfg.get("tcvae", {
        "latent_dim": 64,
        "hidden_dim": 128,
        "pooling": "mean"
    })
    pooling_mode = cfg.get("tcvae_pooling", tcvae_cfg.get("pooling", "mean"))

    print(f"\nTCVAE config:")
    print(f"  - feature_dim: {feature_dim}")
    print(f"  - latent_dim: {tcvae_cfg.get('latent_dim', 64)}")
    print(f"  - hidden_dim: {tcvae_cfg.get('hidden_dim', 128)}")
    print(f"  - pooling: {pooling_mode}")

    # Use the TCVAE encoder; pooling strategy is one of ('mean', 'max', 'attention', 'last')
    enc_src = TCVAEConceptEncoder(
        feature_dim,
        latent_dim=tcvae_cfg.get("latent_dim", 64),
        hidden_dim=tcvae_cfg.get("hidden_dim", 128),
        pooling=pooling_mode
    )
    enc_tgt = TCVAEConceptEncoder(
        feature_dim,
        latent_dim=tcvae_cfg.get("latent_dim", 64),
        hidden_dim=tcvae_cfg.get("hidden_dim", 128),
        pooling=pooling_mode
    )

    assert tcvae_cfg.get("latent_dim", 64) == A["concept_dim"], \
        f"TCVAE's latent_dim ({tcvae_cfg.get('latent_dim', 64)}) must equal the Adapter's concept_dim ({A['concept_dim']})"

    adapter = ProceedAdapter(
        concept_dim=A["concept_dim"],
        bottleneck_dim=A["bottleneck_dim"],
        layer_meta=base_model.get_layer_meta(),
        scale_mult=A.get("scale_mult", 0.05)
    )

    # Choose the augmentation method
    diff_cfg = cfg.get("diffusion", {})

    # Decide whether to use diffusion augmentation
    use_diffusion = diff_cfg.get("enable", True)

    try:
        if use_diffusion:
            print(f"\nUsing diffusion data augmentation")
            print(f"Mode: {diff_cfg.get('mode', 'mixed')}")
            print(f"Augmentation probability: {diff_cfg.get('augment_prob', 0.35)}")

            diffusion_save_path = os.path.join(
                diff_cfg.get("save_dir", "checkpoints"),
                f"diffusion_{stock_id}.pth"
            )

            augmentor = setup_diffusion_augmentor(
                Xw_tr,
                seq_len=cfg["seq_len"],
                feature_dim=feature_dim,
                mode=diff_cfg.get("mode", "mixed"),
                device=DEVICE,
                pretrain_epochs=diff_cfg.get("pretrain_epochs", 5),
                save_path=diffusion_save_path
            )

            A_modified = A.copy()
            A_modified["synth_prob"] = diff_cfg.get("augment_prob", 0.35)

            history_adapter = train_adapter_proceed_with_diffusion_validated(
                enc_src, enc_tgt, adapter, base_model,
                Xw_tr, Yw_tr, A_modified, augmentor, DEVICE,
                val_ratio=0.2,
                save_best_path='checkpoints/best.pth'
            )

            print(f"Best DA: {history_adapter['best_val_da']:.4f}")

        else:
            print(f"\nNo data augmentation (still with validation)")
            print(f"Augmentation probability: {A.get('synth_prob', 0.35)}")
            print(f"Augmentation modes: {A.get('synth_modes', [])}")

            # Use the validation-aware adapter trainer, but without data augmentation
            history_adapter = train_adapter_with_validation(
                enc_src=enc_src,
                enc_tgt=enc_tgt,
                adapter=adapter,
                base_model=base_model,
                X_train=Xw_tr,
                Y_train=Yw_tr,
                cfg=A,
                loss_fn=loss_fn,
                device=DEVICE,
                val_ratio=0.2,
                use_diffusion=False
            )

            if history_adapter:
                print("\n" + "=" * 80)
                print("Adapter training complete!")
                print("=" * 80)
                print(f"Best epoch: {history_adapter['best_epoch'] + 1}")
                print(f"Best val loss: {history_adapter['best_val_loss']:.6f}")

    except Exception as e:
        print(f"\nAdapter training failed: {e}")
        import traceback
        traceback.print_exc()
        history_adapter = None

    # =================== Online fine-tuning and inference (with validation) ===================
    print("\n" + "=" * 80)
    print("Stage 6: Online testing and inference")
    print("=" * 80)

    online_cfg = cfg["online_ft"]
    online_model, ft_params = build_online_model(base_model, online_cfg["finetune_layers"])

    preds = []
    trues = []
    rolling_anchor = collections.deque(maxlen=1)
    H = pred_len

    # Track online fine-tuning accept/reject counts
    finetune_accepted = 0
    finetune_rejected = 0
    finetune_skipped = 0

    with torch.no_grad():
        xi_anchor = torch.tensor(Xw_tr[-1:], dtype=torch.float32).to(DEVICE)
        c_anchor = enc_src(xi_anchor).detach()
        rolling_anchor.append(c_anchor.squeeze(0))

    print(f"Starting online testing ({len(Xw_test)} steps total)...")

    for i in range(len(Xw_test)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i + 1}/{len(Xw_test)}", end='\r')

        xj = torch.tensor(Xw_test[i:i + 1], dtype=torch.float32).to(DEVICE)
        y_true = Yw_test[i]

        # Online fine-tuning (with validation)
        if online_cfg["enable"]:
            X_ft_list, Y_ft_list = [], []
            safe_idx = i - H
            if safe_idx >= 0:
                X_ft_list.append(Xw_test[safe_idx:safe_idx + 1])
                Y_ft_list.append(Yw_test[safe_idx:safe_idx + 1])
            ut = online_cfg.get("use_train_tail", 0)
            if ut > 0:
                X_ft_list.append(Xw_tr[-min(ut, len(Xw_tr)):])
                Y_ft_list.append(Yw_tr[-min(ut, len(Yw_tr)):])

            if len(X_ft_list) > 0:
                X_ft = np.concatenate(X_ft_list, axis=0)
                Y_ft = np.concatenate(Y_ft_list, axis=0)

                # Fine-tune with validation
                accepted = online_finetune_with_validation(
                    online_model=online_model,
                    params=ft_params,
                    X_ft=X_ft,
                    Y_ft=Y_ft,
                    loss_fn=loss_fn,
                    lr=online_cfg["lr"],
                    steps=online_cfg["steps"],
                    batch_size=online_cfg["batch_size"],
                    val_ratio=0.2,
                    min_samples=10,
                    device=DEVICE
                )

                if accepted:
                    finetune_accepted += 1
                elif accepted is False:
                    finetune_rejected += 1
                else:
                    finetune_skipped += 1
            else:
                finetune_skipped += 1

        # Inference
        with torch.no_grad():
            if i >= (H - 1):
                xi_th = torch.tensor(Xw_test[i - (H - 1):i - (H - 1) + 1],
                                     dtype=torch.float32).to(DEVICE)
                c_th = enc_src(xi_th).detach().squeeze(0)
                rolling_anchor.clear()
                rolling_anchor.append(c_th)

            cj = enc_tgt(xj)
            ci = rolling_anchor[-1].unsqueeze(0).to(DEVICE)
            delta = cj - ci
            mod = adapter(delta)
            y_hat = online_model(xj, mod=mod).cpu().numpy().flatten()

        preds.append(y_hat)
        trues.append(y_true)

    print()  # newline

    if online_cfg["enable"]:
        print(f"\nOnline fine-tuning statistics:")
        print(f"Accepted: {finetune_accepted}")
        print(f"Rejected: {finetune_rejected}")
        print(f"Skipped: {finetune_skipped}")
        accept_rate = finetune_accepted / max(1, finetune_accepted + finetune_rejected) * 100
        print(f"Accept rate: {accept_rate:.1f}%")

    # =================== Metric computation ===================
    print("\n" + "=" * 80)
    print("Stage 7: Metric computation")
    print("=" * 80)

    preds_arr = np.array(preds).reshape(len(preds), -1)
    trues_arr = np.array(trues).reshape(len(trues), -1)
    Hh = preds_arr.shape[1]

    mae_all = float(np.mean(np.abs(preds_arr - trues_arr)))
    rmse_all = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))

    mae_h = np.mean(np.abs(preds_arr - trues_arr), axis=0)
    rmse_h = np.sqrt(np.mean((preds_arr - trues_arr) ** 2, axis=0))

    eps = 1e-8
    smape_h = 2.0 * np.mean(
        np.abs(preds_arr - trues_arr) / (np.abs(preds_arr) + np.abs(trues_arr) + eps),
        axis=0
    ) * 100.0
    smape_all = float(np.mean(
        2.0 * np.abs(preds_arr - trues_arr) / (np.abs(preds_arr) + np.abs(trues_arr) + eps)
    ) * 100.0)

    da = float(np.mean(np.sign(preds_arr) == np.sign(trues_arr)))
    r2 = float(1 - np.sum((preds_arr - trues_arr) ** 2) /
               (np.sum((trues_arr - trues_arr.mean()) ** 2) + 1e-12))

    # =================== Print the final results ===================
    method_tag = "Diffusion" if diff_cfg.get("enable", False) else "No Augment "

    print("\n" + "=" * 80)
    print(f"Final results - {stock_id}")
    print("=" * 80)
    print(f"Method: {method_tag} augmentation + DirectionalLoss")
    print(f"Training framework: Validation Framework (early stopping)")
    print()
    print(f"Overall metrics:")
    print(f"MAE:   {mae_all:.6f}")
    print(f"RMSE:  {rmse_all:.6f}")
    print(f"sMAPE: {smape_all:.2f}%")
    print(f"DAR:    {da:.3f}")
    print(f"R²:    {r2:.3f}")



    print(f"\nPer-horizon metrics:")
    for h in range(Hh):
        print(f"  H{h + 1:02d} MAE={mae_h[h]:.6f} RMSE={rmse_h[h]:.6f} "
              f"sMAPE={smape_h[h]:.2f}%")

    # =================== Return the detailed results ===================
    results = {
        # basic metrics
        "MAE": mae_all, "RMSE": rmse_all, "sMAPE": smape_all, "DA": da,
        "R2": r2, "MAE_H": mae_h, "RMSE_H": rmse_h, "sMAPE_H": smape_h,

        # method tag
        "method": method_tag,
        "loss_type": "DirectionalLoss",
        "use_validation": True,

        # training histories
        "history_pretrain": history_pretrain,
        "history_adapter": history_adapter,

        # online fine-tuning statistics
        "online_finetune_stats": {
            "accepted": finetune_accepted,
            "rejected": finetune_rejected,
            "skipped": finetune_skipped,
            "accept_rate": accept_rate if online_cfg["enable"] else 0
        }
    }

    print("\n" + "=" * 80)
    print(f"Stock {stock_id} processing complete!")
    print("=" * 80)

    return results


# ============== Main ==============
def parse_args():
    """Command-line arguments. Defaults match the original __main__'s
    hardcoded EXPERIMENT_CONFIG exactly, so running
    ``python run_experiment.py`` with no arguments behaves identically to
    before.

    Previously, switching between ablation experiments meant manually
    editing a dict in the source. Now you can simply run:

        python run_experiment.py --tcvae-pooling mean
        python run_experiment.py --diffusion-enable --diffusion-mode faithful
    """
    import argparse
    ap = argparse.ArgumentParser(description="Section 10.6: distribution-drift-adaptive stock-price forecasting experiment")
    ap.add_argument("--tcvae-pooling", default="attention",
                    choices=["mean", "max", "last", "attention"],
                    help="pooling strategy for the TCVAE concept encoder (default: attention)")
    ap.add_argument("--diffusion-enable", action="store_true",
                    help="enable diffusion-model data augmentation (default: off)")
    ap.add_argument("--diffusion-mode", default="mixed",
                    choices=["faithful", "adversarial", "mixed"],
                    help="diffusion augmentation mode, only used if --diffusion-enable is set (default: mixed)")
    return ap.parse_args()


if __name__ == "__main__":
    # =================== Experiment config ===================
    _args = parse_args()
    EXPERIMENT_CONFIG = {
        "tcvae_pooling": _args.tcvae_pooling,   # one of: 'mean', 'max', 'last', 'attention'
        "diffusion_enable": _args.diffusion_enable,
        "diffusion_mode": _args.diffusion_mode,  # one of: 'faithful', 'adversarial', 'mixed'
    }

    print("\n" + "=" * 80)
    print("Experiment configuration")
    print("=" * 80)
    print(f"TCVAE pooling mode: {EXPERIMENT_CONFIG['tcvae_pooling']}")
    print(f"Diffusion augmentation: {'enabled' if EXPERIMENT_CONFIG['diffusion_enable'] else 'disabled'}")
    if EXPERIMENT_CONFIG['diffusion_enable']:
        print(f"Diffusion mode: {EXPERIMENT_CONFIG['diffusion_mode']}")
    print("=" * 80 + "\n")

    # =================== Sync into the global config ===================
    # Propagate the experiment config into CFG
    CFG["tcvae"]["pooling"] = EXPERIMENT_CONFIG["tcvae_pooling"]  # update the default pooling
    CFG["tcvae_pooling"] = EXPERIMENT_CONFIG["tcvae_pooling"]  # takes priority in run_stock
    CFG["diffusion"]["enable"] = EXPERIMENT_CONFIG["diffusion_enable"]
    if EXPERIMENT_CONFIG["diffusion_enable"]:
        CFG["diffusion"]["mode"] = EXPERIMENT_CONFIG["diffusion_mode"]

    # =================== Load the data ===================
    price, bench, panel_data = load_real_data(
        CFG["data_path"],
        CFG["max_assets"],
        CFG["max_days"]
    )

    # =================== Run a single group of experiments ===================
    rows = []
    method_name = "Diffusion" if EXPERIMENT_CONFIG["diffusion_enable"] else "No_Augment"
    if EXPERIMENT_CONFIG["diffusion_enable"]:
        method_name += f"_{EXPERIMENT_CONFIG['diffusion_mode']}"
    method_name += f"_TCVAE_{EXPERIMENT_CONFIG['tcvae_pooling']}"

    print("\n" + "=" * 80)
    print(f"Starting experiment: {method_name}")
    print("=" * 80 + "\n")

    for stock_id, df_one in tqdm(
            panel_data.groupby("ts_code"),
            desc=f"Progress"
    ):
        try:
            res = run_stock(
                df_one.sort_values("trade_date"),
                stock_id,
                CFG  # pass the full CFG
            )

            if res is None:
                continue

            row = {
                "stock_id": stock_id,
                "method": method_name,
                "tcvae_pooling": EXPERIMENT_CONFIG["tcvae_pooling"],
                "tcvae_latent_dim": CFG["tcvae"]["latent_dim"],  # records the dimension config
                "tcvae_hidden_dim": CFG["tcvae"]["hidden_dim"],
                "diffusion_enable": EXPERIMENT_CONFIG["diffusion_enable"],
                "diffusion_mode": EXPERIMENT_CONFIG["diffusion_mode"] if EXPERIMENT_CONFIG[
                    "diffusion_enable"] else "N/A",
                "loss_type": res.get("loss_type", "DirectionalLoss"),
                "MAE": res["MAE"],
                "RMSE": res["RMSE"],
                "sMAPE": res["sMAPE"],
                "DA": res["DA"],
                "R2": res["R2"]
            }

            H = len(res["MAE_H"])
            for h in range(H):
                row[f"MAE@H{h + 1}"] = res["MAE_H"][h]
                row[f"RMSE@H{h + 1}"] = res["RMSE_H"][h]
                row[f"sMAPE@H{h + 1}"] = res["sMAPE_H"][h]

            rows.append(row)

        except Exception as e:
            print(f"\n[ERROR] Stock {stock_id} skipped, reason: {e}")
            import traceback

            traceback.print_exc()

    # =================== Save the results ===================
    if len(rows) == 0:
        print("\nWarning: no stock was processed successfully!")
        exit(0)

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("Experiment result statistics")
    print("=" * 80)
    print(f"\nExperiment config: {method_name}")
    print(out[['MAE', 'RMSE', 'sMAPE', 'DA', 'R2']].describe())

    result_filename = f"checkpoints/results_{method_name}.csv"
    out.to_csv(result_filename, index=False)

    print(f"\n✓ Results saved!")
    print(f"  - Results file: {result_filename}")
    print(f"  - Successfully processed: {len(rows)} stocks")
