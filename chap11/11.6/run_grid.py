"""Table 11.2's 16-experiment ablation grid.

Table 11.2 runs 4 ablation groups (baseline encoder+PPO; +multi-objective
optimization; +curriculum learning; +both) x 2 encoder types (Vanilla,
iTransformer) x 2 asset-pool sizes (30, 120) = 16 experiments.This script 
constructs the full grid by invoking run_experiment.py as a **separate subprocess** 
per experiment and collecting each run's metrics into a summary table.

Usage:
    python run_grid.py
    python run_grid.py --quick   # drastically reduced settings, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config
from run_experiment import OUTPUT_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent


def _run_one(name: str, quick: bool) -> dict | None:
    cmd = [sys.executable, str(SCRIPT_DIR / "run_experiment.py"), "--experiment", name]
    if quick:
        cmd.append("--quick")
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(f"[ERROR] {name} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        return None

    result_path = OUTPUT_ROOT / name / "graph_rl_results.json"
    if not result_path.exists():
        print(f"[ERROR] {name}: {result_path} was not produced")
        return None
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_grid(*, quick: bool = False) -> pd.DataFrame:
    rows = []
    for name, exp in config.EXPERIMENTS.items():
        print(f"\n{'#' * 80}\n# {name}  (group={exp['group']}, encoder={exp['encoder_type']}, max_assets={exp['max_assets']})\n{'#' * 80}")
        result = _run_one(name, quick)
        if result is None:
            continue

        row = {
            "experiment": name,
            "group": exp["group"],
            "encoder_type": exp["encoder_type"],
            "max_assets": exp["max_assets"],
            "val_annual_return": result["val"]["annual_return"],
            "val_sharpe_ratio": result["val"]["sharpe_ratio"],
            "val_information_ratio": result["val"]["information_ratio"],
            "val_max_drawdown": result["val"]["max_drawdown"],
            "test_annual_return": result["test"]["annual_return"],
            "test_sharpe_ratio": result["test"]["sharpe_ratio"],
            "test_information_ratio": result["test"]["information_ratio"],
            "test_max_drawdown": result["test"]["max_drawdown"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = OUTPUT_ROOT / "ablation_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_grid_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved the 16-experiment ablation grid results to: {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description="Section 11.6 ablation grid (group x encoder x asset-pool size)")
    ap.add_argument("--quick", action="store_true", help="drastically reduce settings for a fast smoke test")
    args = ap.parse_args()
    run_grid(quick=args.quick)


if __name__ == "__main__":
    main()
