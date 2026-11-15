"""Unified entry point for the Section 5.6 experiments: recurrent networks
and temporal convolution for multi-step stock-price forecasting.

Takes 5 fields from Ping An Bank's daily price data (open/high/low/vol/close)
as input, looks back 120 days, and predicts the next day's closing price,
comparing LSTM, GRU, and TCN.

Usage:

    python run_experiment.py --experiment lstm
    python run_experiment.py --list
    python lstm_example.py                 # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import EXPERIMENTS, get_config
from data_provider import get_data, prepare_data
from evaluate_model import evaluate_model
from train_model import train_model

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(cfg: dict) -> nn.Module:
    """Construct the model from the config.

    LSTM/GRU share the signature ``(input_size, hidden_size, num_layers, output_size, dropout)``;
    TCN's signature is ``(input_size, output_size, seq_len, num_channels, kernel_size, dropout)``.
    """
    model_cls = getattr(importlib.import_module(cfg["module"]), cfg["cls"])
    if cfg["ctor"] == "tcn":
        return model_cls(cfg["input_size"], cfg["output_size"], cfg["seq_len"],
                         cfg["num_channels"], cfg["kernel_size"], cfg["dropout"])
    return model_cls(cfg["input_size"], cfg["hidden_size"], cfg["num_layers"],
                     cfg["output_size"], cfg["dropout"])


def run(name: str, *, show: bool = False, output_root: Path | None = None,
        overrides: dict | None = None, run_dir_name: str | None = None) -> dict:
    cfg = get_config(name)
    if overrides:
        cfg.update(overrides)
    out_dir = (output_root or OUTPUT_ROOT) / (run_dir_name or name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    set_seed(42)

    # Load data
    X, y = get_data(cfg["seq_len"], cfg["pred_horizon"])

    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_data(
        X, y, cfg["train_ratio"], cfg["val_ratio"], cfg["batch_size"])

    # Build the model 
    model = build_model(cfg)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), cfg["lr"])

    # Train the model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        cfg["epochs"], cfg["patience"], device=device)

    # Evaluate the model
    metrics, y_true, y_pred = evaluate_model(model, test_loader, scaler, device=device)

    # Print metrics
    print("\nEvaluation metrics:")
    for var, metrics_dict in metrics.items():
        print(
            f"{var}: MSE = {metrics_dict['MSE']:.6f}, MAE = {metrics_dict['MAE']:.6f}, "
            f"RMSE = {metrics_dict['RMSE']:.6f}, MAPE = {metrics_dict['MAPE']:.2f}%")

    _plot(cfg, y_true, y_pred, train_losses, val_losses, out_dir, show=show)
    return metrics


def _plot(cfg, y_true, y_pred, train_losses, val_losses, out_dir, *, show):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prediction plot
    plt.figure()
    plt.plot(y_true, label='y_true', linestyle='-', linewidth=2)
    plt.plot(y_pred, label='y_pred', linestyle=':', linewidth=2)
    plt.title(f'{cfg["title"]} test', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / f'{cfg["prefix"]}_test.png', dpi=300, bbox_inches='tight')

    # Train/validation loss curve
    plt.figure()
    plt.plot(train_losses, label='train_loss', linestyle=':', linewidth=2)
    plt.plot(val_losses, label='val_loss', linestyle='-', linewidth=2)
    plt.title(f'{cfg["title"]} loss curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / f'{cfg["prefix"]}_loss.png', dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close('all')


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 5.6: LSTM / GRU / TCN stock-price forecasting")
    parser.add_argument("--experiment", "-e", default="lstm",
                        choices=sorted(EXPERIMENTS), help="experiment to run")
    parser.add_argument("--list", action="store_true", help="list all experiments")
    parser.add_argument("--show", action="store_true", help="also pop up the figure window in addition to saving")
    args = parser.parse_args()

    if args.list:
        print(f"{'experiment':10}{'model':16}{'seq_len':>8}{'patience':>10}")
        for key in EXPERIMENTS:
            c = get_config(key)
            print(f"{key:10}{c['cls']:16}{c['seq_len']:>8}{c['patience']:>10}")
        return 0

    run(args.experiment, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
