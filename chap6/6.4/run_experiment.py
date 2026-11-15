"""Unified entry point for the Section 6.4 experiments: attention-enhanced CNNs.

Usage:

    python run_experiment.py --experiment SE_CNN
    python run_experiment.py --list
    python SE_CNN_example.py               # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config import EXPERIMENTS, get_config
from data_utils import create_dataloader, make_images, sort_features_by_correlation
from paths import data_file

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Dynamically import the model class named in the config and instantiate it.

    All 9 models in this section share the same constructor signature ``(num_classes)``.
    """
    model_cls = getattr(importlib.import_module(cfg["module"]), cfg["cls"])
    return model_cls(cfg["num_classes"]).to(device)


def load_data():
    """Load this section's 2017-11-30 cross-sectional data."""
    return pd.read_csv(data_file('data_ml_2017_11.csv'), index_col='stock_id')


def _build_loaders(cfg: dict, data):
    """Build the three DataLoaders from the cross-sectional data.

    Every experiment in this section first reorders features by
    correlation with the label, then reshapes them into a 9x11 image.
    ``train_test_split`` uses its own ``random_state``-seeded source and
    does not consume the global RNG.
    """
    sorted_data, _ = sort_features_by_correlation(data, label_col=-1, use_abs=cfg["use_abs"])
    images, labels = make_images(sorted_data, label_quantile=cfg["num_classes"])
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        images, labels, test_size=cfg["test_size"], random_state=cfg["random_state"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=cfg["val_size"], random_state=cfg["random_state"])

    return (create_dataloader(X_train, y_train),
            create_dataloader(X_val, y_val, shuffle=False),
            create_dataloader(X_test, y_test, shuffle=False))


def run(name: str, *, output_root: Path | None = None) -> tuple:
    """Run a single experiment and return its test-set metrics (loss, accuracy, f1)."""
    cfg = get_config(name)
    out_dir = (output_root or OUTPUT_ROOT) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # No popup window, to avoid blocking in a headless environment
    matplotlib.use("Agg")
    from train_utils import evaluate_model, plot_results, train_model

    set_seed(42)

    data = load_data()
    train_loader, val_loader, test_loader = _build_loaders(cfg, data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])

    criterion = nn.CrossEntropyLoss()

    history = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                          cfg["epochs"], cfg["patience"],
                          checkpoint_path=out_dir / "best_model.pt")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    plot_results(history, test_metrics, test_loader, model, device, output_dir=out_dir)
    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 6.4: attention-enhanced CNNs")
    parser.add_argument("--experiment", "-e", default="SE_CNN",
                        choices=sorted(EXPERIMENTS), help="experiment to run")
    parser.add_argument("--list", action="store_true", help="list all experiments")
    args = parser.parse_args()

    if args.list:
        print("This section's 9 experiments differ only in the model; hyperparameters are identical:")
        for key in EXPERIMENTS:
            print(f"  {key:16} <- {EXPERIMENTS[key]['module']}")
        return 0

    run(args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
