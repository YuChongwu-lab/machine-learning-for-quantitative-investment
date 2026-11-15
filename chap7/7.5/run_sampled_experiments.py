"""Reconstruct the book's time-sampled SAINT comparison for Section 7.5.

Beyond the 3 full-data table-pipeline experiments (SAINT, SimpleSAINT,
SAINT_cutmix), the book additionally compares SAINT and SAINT+CutMix on
data sampled by time group.

Usage:

    python run_sampled_experiments.py
    python run_sampled_experiments.py --epochs 3   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from train_by_time_group import train_by_time_groups

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "sampled"

MODEL_KWARGS = dict(d_model=128, n_heads=8, n_blocks=6, dropout=0.1)


def run_one(name: str, *, use_cutmix: bool, epochs: int | None, output_root: Path) -> dict:
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")

    results = train_by_time_groups(
        time_col='date',
        num_classes=2,
        batch_size=64,
        lr=1e-3,
        epochs=epochs if epochs is not None else 100,
        patience=10,
        test_size=0.2,
        val_size=0.16,
        sample_time_groups=1,   # one sampled time group, not the book's two
        sample_seed=42,
        use_cutmix=use_cutmix,
        **MODEL_KWARGS,
    )
    results.to_csv(out_dir / "results.csv", index=False)
    metrics = results.iloc[0].to_dict() if len(results) else {}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs (e.g. for a quick smoke test)")
    args = ap.parse_args()

    for name, use_cutmix in (("SAINT_sampled", False), ("SAINT_cutmix_sampled", True)):
        print(f"\n=== {name} ===")
        metrics = run_one(name, use_cutmix=use_cutmix, epochs=args.epochs, output_root=OUTPUT_ROOT)
        print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
