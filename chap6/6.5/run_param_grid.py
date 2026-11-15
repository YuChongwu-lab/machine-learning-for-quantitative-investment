"""Construct the book's 11-experiment model/seq_len sensitivity analysis
for Section 6.5.

Usage:

    python run_param_grid.py
    python run_param_grid.py --epochs 3   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import EXTRA_CELLS, GRID_MODELS, SEQ_LEN_GRID
from run_experiment import OUTPUT_ROOT, run


def run_grid(output_root: Path, epochs: int | None) -> list[dict]:
    rows = []
    for seq_len in SEQ_LEN_GRID:
        for model in GRID_MODELS:
            run_dir_name = f"{model}_sl{seq_len}"
            overrides = {"seq_len": seq_len, "patience": 10}
            if epochs is not None:
                overrides["epochs"] = epochs
            print(f"\n=== {run_dir_name} ===")
            metrics = run(model, output_root=output_root, overrides=overrides,
                          run_dir_name=run_dir_name)
            for var, m in metrics.items():
                rows.append({"seq_len": seq_len, "model": model, "patience": 10,
                             "variable": var, **m})

    for cell in EXTRA_CELLS:
        model, seq_len, patience = cell["model"], cell["seq_len"], cell["patience"]
        run_dir_name = f"{model}_sl{seq_len}_patience{patience}"
        overrides = {"seq_len": seq_len, "patience": patience}
        if epochs is not None:
            overrides["epochs"] = epochs
        print(f"\n=== {run_dir_name} (extra) ===")
        metrics = run(model, output_root=output_root, overrides=overrides,
                      run_dir_name=run_dir_name)
        for var, m in metrics.items():
            rows.append({"seq_len": seq_len, "model": model, "patience": patience,
                         "variable": var, **m})

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs for all cells (e.g. for a quick smoke test)")
    args = ap.parse_args()

    output_root = OUTPUT_ROOT / "seq_len_grid"
    rows = run_grid(output_root, args.epochs)
    write_csv(rows, output_root / "seq_len_grid_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
