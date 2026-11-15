"""Table 7-3 / Table 7-4 parameter and model sweeps.

The book reports two grids:

* Table 7-3 ("PART2_GRID"): the base Transformer model swept over
  seq_len x split ratio x stock (27 combinations).
* Table 7-4 ("PART3_GRID"): all 12 models swept over 3 stocks, with
  seq_len and the split ratio held fixed (36 combinations).

Usage:

    python run_param_grid.py --part 2
    python run_param_grid.py --part 3
    python run_param_grid.py --part 2 --epochs 3   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import PART2_GRID, PART3_GRID, get_config
from run_experiment import OUTPUT_ROOT, run


def run_part2(output_root: Path, epochs: int | None) -> list[dict]:
    rows = []
    model = PART2_GRID["model"]
    for stock in PART2_GRID["stocks"]:
        for seq_len in PART2_GRID["seq_lens"]:
            for train_ratio, val_ratio in PART2_GRID["splits"]:
                run_dir_name = f"{stock}_sl{seq_len}_{train_ratio}_{val_ratio}"
                overrides = {
                    "data": f"{stock}.csv",
                    "seq_len": seq_len,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                }
                if epochs is not None:
                    overrides["epochs"] = epochs
                print(f"\n=== Part 2: {run_dir_name} ===")
                metrics = run(model, output_root=output_root, overrides=overrides,
                              run_dir_name=run_dir_name)
                rows.append({"stock": stock, "seq_len": seq_len,
                             "train_ratio": train_ratio, "val_ratio": val_ratio,
                             **metrics})
    return rows


def run_part3(output_root: Path, epochs: int | None) -> list[dict]:
    rows = []
    for model in PART3_GRID["models"]:
        for stock in PART3_GRID["stocks"]:
            run_dir_name = f"{model}_{stock}"
            overrides = {
                "data": f"{stock}.csv",
                "seq_len": PART3_GRID["seq_len"],
                "train_ratio": PART3_GRID["train_ratio"],
                "val_ratio": PART3_GRID["val_ratio"],
            }
            if epochs is not None:
                overrides["epochs"] = epochs
            print(f"\n=== Part 3: {run_dir_name} ===")
            metrics = run(model, output_root=output_root, overrides=overrides,
                          run_dir_name=run_dir_name)
            rows.append({"model": model, "stock": stock, **metrics})
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
    ap.add_argument("--part", type=int, choices=(2, 3), required=True,
                    help="which grid to run: 2 for Table 7-3, 3 for Table 7-4")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs for all cells (e.g. for a quick smoke test)")
    args = ap.parse_args()

    output_root = OUTPUT_ROOT / f"grid_part{args.part}"
    if args.part == 2:
        rows = run_part2(output_root, args.epochs)
        write_csv(rows, output_root / "table_7-3_results.csv")
    else:
        rows = run_part3(output_root, args.epochs)
        write_csv(rows, output_root / "table_7-4_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
