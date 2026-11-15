"""Unified entry point for the Section 7.5 experiments: feeding tabular
data to vision models and a table Transformer.

This section has **two parallel data pipelines**:

* **Image pipeline**: encode the 93-dimensional tabular features into
  images with DeepInsight / MTF / etc., then classify with SimpleCNN or ViT.
* **Table pipeline**: feed features directly to a table-oriented
  Transformer (SAINT / SimpleSAINT), optionally with column-wise CutMix
  augmentation.

The two pipelines split data differently (the image pipeline uses
``train_test_split`` twice; the table pipeline uses ``split_raw`` for a
single three-way split), deliberately kept separate and not merged.

Usage:

    python run_experiment.py --experiment SAINT
    python run_experiment.py --list
    python SAINT_example.py                # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config import EXPERIMENTS, get_config
from data_utils import (create_dataloader, create_dataloader_raw, make_images,
                        split_raw)
from get_data import get_data

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Construct the model from the config. This section's four models each have a different constructor signature."""
    model_cls = getattr(importlib.import_module(cfg["module"]), cfg["cls"])
    kind = cfg["ctor"]
    if kind == "num_classes":
        return model_cls(cfg["num_classes"]).to(device)
    if kind == "vit":
        return model_cls(num_classes=cfg["num_classes"],
                         img_pool_size=cfg["img_pool_size"],
                         patch_size=cfg["patch_size"],
                         embed_dim=cfg["embed_dim"],
                         num_layers=cfg["num_layers"],
                         nheads=cfg["n_heads"],
                         mlp_ratio=cfg["mlp_ratio"],
                         dropout=cfg["dropout"]).to(device)
    if kind == "saint":
        # The original script never passed num_layers to SAINT (whose
        # n_blocks default happens to be 6); kept consistent here
        return model_cls(num_features=cfg["num_features"], num_classes=cfg["num_classes"],
                         d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         dropout=cfg["dropout"]).to(device)
    if kind == "simple_saint":
        return model_cls(num_features=cfg["num_features"], num_classes=cfg["num_classes"],
                         d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         n_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
    raise ValueError(f"unknown constructor kind: {kind}")


def load_data():
    """Load this section's full panel and arrange it into features + label (see get_data.py)."""
    return get_data()


def _build_loaders(cfg: dict, data):
    """Build the three DataLoaders according to the pipeline type."""
    bs = cfg["batch_size"]
    if cfg["pipeline"] == "image":
        images, labels = make_images(data, label_quantile=cfg["num_classes"],
                                     method=cfg["method"], image_shape=cfg["image_shape"],
                                     random_state=cfg["random_state"])
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            images, labels, test_size=cfg["test_size"], random_state=cfg["random_state"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=cfg["test_size"], random_state=cfg["random_state"])
        return (create_dataloader(X_train, y_train, batch_size=bs),
                create_dataloader(X_val, y_val, batch_size=bs, shuffle=False),
                create_dataloader(X_test, y_test, batch_size=bs, shuffle=False))

    X_train, X_val, X_test, y_train, y_val, y_test = split_raw(
        data, label_quantile=cfg["num_classes"], test_size=cfg["test_size"],
        val_size=cfg["val_size"], random_state=cfg["random_state"])
    return (create_dataloader_raw(X_train, y_train, batch_size=bs),
            create_dataloader_raw(X_val, y_val, batch_size=bs, shuffle=False),
            create_dataloader_raw(X_test, y_test, batch_size=bs, shuffle=False))


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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    history = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                          cfg["epochs"], cfg["patience"], **cfg["train_kwargs"],
                          checkpoint_path=out_dir / "best_model.pt")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    plot_results(history, test_metrics, test_loader, model, device, output_dir=out_dir)
    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 7.5: tabular-to-image encoding and table Transformers")
    parser.add_argument("--experiment", "-e", default="SAINT",
                        choices=sorted(EXPERIMENTS), help="experiment to run")
    parser.add_argument("--list", action="store_true", help="list all experiments and their settings")
    args = parser.parse_args()

    if args.list:
        print(f"{'experiment':16}{'pipeline':8}{'model':20}{'encoding/augmentation'}")
        for key in EXPERIMENTS:
            c = get_config(key)
            extra = c.get("method") or ("CutMix" if c["train_kwargs"] else "-")
            print(f"{key:16}{c['pipeline']:8}{c['cls']:20}{extra}")
        return 0

    run(args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
