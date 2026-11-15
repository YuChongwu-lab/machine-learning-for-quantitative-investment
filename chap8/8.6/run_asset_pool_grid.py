"""Asset-pool-size sensitivity analysis.

Usage:
    python run_asset_pool_grid.py
    python run_asset_pool_grid.py --quick   # drastically reduced settings, for a smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from run_experiment import run, OUTPUT_ROOT


def run_grid(*, quick: bool = False) -> pd.DataFrame:
    if quick:
        config.SHARED["max_days"] = 120
        config.SHARED["train"]["epochs"] = 1
        config.SHARED["search"]["max_trials"] = 1
        config.SHARED["search"]["fast_train_epochs"] = 1
        config.SHARED["explain"]["n_samples"] = 1
        config.SHARED["backtest"]["test_limit"] = 10

    rows = []
    for max_assets in config.ASSET_POOL_GRID:
        print(f"\n{'#' * 80}\n# max_assets = {max_assets}\n{'#' * 80}")
        try:
            result = run(max_assets=max_assets)
        except Exception as e:
            print(f"[ERROR] max_assets={max_assets} failed: {e}")
            continue
        if result is None:
            print(f"[SKIPPED] max_assets={max_assets}: dataset construction failed")
            continue

        bt = result["backtest_results"]
        row = {
            "max_assets": max_assets,
            "total_return": bt["total_return"],
            "annual_return": bt["annual_return"],
            "annual_volatility": bt["annual_volatility"],
            "sharpe_ratio": bt["sharpe_ratio"],
            "max_drawdown": bt["max_drawdown"],
            "win_rate": bt["win_rate"],
            "information_ratio": bt["information_ratio"],
            "excess_total_return": bt["excess_total_return"],
            "val_ic": result["val_ic"]["ic_mean"],
            "test_ic": result["test_ic"]["ic_mean"],
            "best_corr_window": result["search_results"]["best_params"].get("corr_window"),
            "best_corr_threshold": result["search_results"]["best_params"].get("corr_threshold"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = OUTPUT_ROOT / "asset_pool_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "asset_pool_grid_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved the asset-pool-size grid results to: {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description="Section 8.6 asset-pool-size sensitivity grid")
    ap.add_argument("--quick", action="store_true", help="drastically reduce settings for a fast smoke test")
    args = ap.parse_args()
    run_grid(quick=args.quick)


if __name__ == "__main__":
    main()
