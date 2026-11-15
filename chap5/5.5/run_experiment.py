"""Unified entry point for the Section 5.5 experiments: CNN classification
of cross-sectional stock data.

Each stock's 93 contemporaneous features are zero-padded and reshaped into
a 9x11 single-channel image, then classified with a CNN. This section
compares four backbones (SimpleCNN / deformable convolution / ResNet18 /
ConvNeXt) and the effect of feature ordering (original order / reordered
by correlation with the label / reordered by absolute correlation).

Usage:

    python run_experiment.py --experiment SimpleCNN_Corr_Data
    python run_experiment.py --list
    python SimpleCNN_Raw_Data.py           # equivalent thin entry point
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

    Most models' constructor signature is ``(num_classes)``; ConvNeXt is
    the exception, constructed with keyword arguments via ``model_kwargs``.
    """
    module = importlib.import_module(cfg["module"])
    model_cls = getattr(module, cfg["cls"])
    if cfg["model_kwargs"] is not None:
        return model_cls(**cfg["model_kwargs"]).to(device)
    return model_cls(cfg["num_classes"]).to(device)


def load_data():
    """Load this section's 2017-11-30 cross-sectional data."""
    return pd.read_csv(data_file('data_ml_2017_11.csv'), index_col='stock_id')


def _build_loaders(cfg: dict, data):
    """Build the three DataLoaders from the cross-sectional data.

    The Raw variants use the original feature order directly; the Corr
    variants first reorder by correlation with the label.
    """
    if cfg["sort_features"]:
        features, _ = sort_features_by_correlation(data, label_col=-1, use_abs=cfg["use_abs"])
    else:
        features = data

    images, labels = make_images(features, label_quantile=cfg["num_classes"])
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

    matplotlib.use("Agg")   # no popup window, to avoid blocking in a headless environment
    from train_utils import evaluate_model, plot_results, train_model

    set_seed(42)


    data = load_data()
    train_loader, val_loader, test_loader = _build_loaders(cfg, data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    history = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                          cfg["epochs"], cfg["patience"],
                          checkpoint_path=out_dir / "best_model.pt")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    plot_results(history, test_metrics, test_loader, model, device, output_dir=out_dir)
    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 5.5: CNN classification of cross-sectional stock data")
    parser.add_argument("--experiment", "-e", default="SimpleCNN_Corr_Data",
                        choices=sorted(EXPERIMENTS), help="experiment to run")
    parser.add_argument("--list", action="store_true", help="list all experiments and their settings")
    args = parser.parse_args()

    if args.list:
        print(f"{'experiment':26}{'model':16}{'feature order':14}{'use_abs':9}{'lr':<9}{'val':<6}{'weight_decay'}")
        for key in EXPERIMENTS:
            c = get_config(key)
            sort = "by correlation" if c["sort_features"] else "original"
            print(f"{key:26}{c['cls']:16}{sort:14}{str(c['use_abs']):9}"
                  f"{c['lr']:<9}{c['val_size']:<6}{c['weight_decay']}")
        return 0

    run(args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
