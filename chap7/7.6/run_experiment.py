"""Unified entry point for the Section 7.6 experiments.

Usage:

    python run_experiment.py --experiment transformer
    python run_experiment.py --list
    python transformer_example.py          # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import importlib
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import EXPERIMENTS, get_config
from data_provider import TimeSeriesScaler, create_dataloader
from metrics import direction_accuracy
from paths import stock_file
from train_and_evaluate_model import train_model

#: Output root. 
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: dict, input_size: int, device: torch.device) -> nn.Module:
    """Dynamically import the model class named in the config and instantiate it.

    All 12 models in this section share the same constructor signature:
    ``(input_size, d_model, num_heads, num_layers, dropout, output_size)``.
    """
    module = importlib.import_module(cfg["module"])
    model_cls = getattr(module, cfg["cls"])
    return model_cls(
        input_size, cfg["d_model"], cfg["num_heads"],
        cfg["num_layers"], cfg["dropout"], cfg["output_size"],
    ).to(device)


def run(name: str, *, show: bool = False, output_root: Path | None = None,
        overrides: dict | None = None, run_dir_name: str | None = None) -> dict:
    """Run a single experiment and return its test-set metrics.

    ``overrides`` lets a caller substitute individual config values (e.g.
    ``data``, ``seq_len``, ``train_ratio``, ``val_ratio``) without touching
    the registered per-experiment defaults in ``config.py``. This exists
    for the grid sweeps in ``run_param_grid.py`` (book Tables 7-3/7-4).
    """
    cfg = get_config(name)
    if overrides:
        cfg.update(overrides)
    out_dir = (output_root or OUTPUT_ROOT) / (run_dir_name or name)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)

    df = pd.read_csv(stock_file(cfg["data"]), index_col="trade_date")

    # ===== 1. Parameters =====
    seq_len = cfg["seq_len"]              # lookback window length
    batch_size = cfg["batch_size"]        # batch size
    input_size = df.shape[1] - 1          # number of features
    output_size = cfg["output_size"]      # forecast horizon

    # ===== 2. Train/val/test split =====
    train_size = int(len(df) * cfg["train_ratio"])
    val_size = int(len(df) * cfg["val_ratio"])
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # ===== 3. Initialize the scaler =====
    scaler = TimeSeriesScaler()

    # ===== 4. DataLoaders =====
    train_loader = create_dataloader(train_df, seq_len, batch_size, scaler, is_train=True)
    val_loader = create_dataloader(val_df, seq_len, batch_size, scaler, is_train=False)
    test_loader = create_dataloader(test_df, seq_len, batch_size, scaler, is_train=False)

    # ===== 5. Device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 6. Model =====
    model = build_model(cfg, input_size, device)

    # ===== 7. Loss, optimizer, LR schedule =====
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    # ===== 8. Training =====
    ckpt_path = out_dir / "best_model.pth"
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg["epochs"], cfg["patience"], device, checkpoint_path=ckpt_path,
    )

    # ===== 9. Load the best checkpoint and evaluate =====
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            preds_real = scaler.inverse_transform_targets(preds)          # inverse-scale
            targets_real = scaler.inverse_transform_targets(y_batch.cpu().numpy())

            all_preds.extend(preds_real.flatten())
            all_targets.extend(targets_real.flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ===== 10. Compute metrics =====
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    da = direction_accuracy(all_targets, all_preds)

    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test Direction Accuracy: {da:.4f}")

    _plot(name, all_targets, all_preds, train_losses, val_losses, out_dir, show=show)

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "direction_accuracy": da}


def _plot(name, all_targets, all_preds, train_losses, val_losses, out_dir, *, show):
    """Plot the prediction comparison and loss curves.
    """
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ===== 11. Prediction vs. ground truth =====
    plt.figure(1, figsize=(12, 6))
    plt.plot(all_targets, label="True", alpha=0.7)
    plt.plot(all_preds, label="Predicted", alpha=0.7)
    plt.legend()
    plt.title("True vs Predicted on Test Set")
    plt.xlabel("Samples")
    plt.ylabel("Target Value")
    plt.grid(True)
    plt.savefig(out_dir / f"{name}_prediction.png", dpi=300, bbox_inches="tight")

    # ===== 12. Train/validation loss curves =====
    plt.figure(2, figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Train and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(out_dir / f"{name}_loss.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 7.6: Transformer-family time-series forecasting experiments")
    parser.add_argument("--experiment", "-e", default="transformer",
                        choices=sorted(EXPERIMENTS), help="experiment to run")
    parser.add_argument("--list", action="store_true", help="list all experiments and their settings")
    parser.add_argument("--show", action="store_true", help="also pop up the figure window in addition to saving")
    args = parser.parse_args()

    if args.list:
        print(f"{'experiment':22}{'model':22}{'data':16}{'seq_len':>8}{'train':>7}{'val':>6}")
        for key in EXPERIMENTS:
            c = get_config(key)
            print(f"{key:22}{c['cls']:22}{c['data']:16}"
                  f"{c['seq_len']:>8}{c['train_ratio']:>7}{c['val_ratio']:>6}")
        return 0

    run(args.experiment, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
