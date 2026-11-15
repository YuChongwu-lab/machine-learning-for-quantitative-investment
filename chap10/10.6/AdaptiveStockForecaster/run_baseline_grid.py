"""The book's 16-experiment baseline grid comparison for Section 10.6.

The book's baseline comparison runs the first 10 stocks under 4
augmentation conditions (no augmentation, faithful, adversarial, mixed)
crossed with 4 TCVAE pooling strategies (mean, max, last, attention) --
4x4 = 16 experiments. ``run_experiment.py``'s CLI only runs one
(augmentation, pooling) combination per invocation; this script loops
over all 16, reusing ``run_stock`` unchanged.

Usage:

    python run_baseline_grid.py
    python run_baseline_grid.py --max-assets 2 --quick   # fast smoke test
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from config import BASE_CFG
from data import load_real_data
from run_experiment import run_stock

OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "baseline_grid"

#: (augmentation condition, diffusion kwargs) -- "none" leaves diffusion disabled
AUGMENTATION_CONDITIONS = ["none", "faithful", "adversarial", "mixed"]
POOLING_STRATEGIES = ["mean", "max", "last", "attention"]


def build_cfg(augmentation: str, pooling: str, *, quick: bool) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    cfg["tcvae"]["pooling"] = pooling
    cfg["tcvae_pooling"] = pooling
    if augmentation == "none":
        cfg["diffusion"]["enable"] = False
    else:
        cfg["diffusion"]["enable"] = True
        cfg["diffusion"]["mode"] = augmentation
    if quick:
        cfg["pretrain"]["epochs"] = 1
        cfg["adapter_train"]["epochs"] = 1
        cfg["diffusion"]["pretrain_epochs"] = 1
        cfg["max_days"] = 300
    return cfg


def run_grid(max_assets: int, *, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cfg = copy.deepcopy(BASE_CFG)
    if quick:
        base_cfg["max_days"] = 300
    price, bench, panel_data = load_real_data(
        base_cfg["data_path"], max_assets, base_cfg["max_days"]
    )
    stocks = list(panel_data.groupby("ts_code"))
    print(f"Loaded {len(stocks)} stocks for the baseline grid.")

    per_stock_rows = []
    for augmentation in AUGMENTATION_CONDITIONS:
        for pooling in POOLING_STRATEGIES:
            method_name = f"{augmentation}_{pooling}"
            print(f"\n{'=' * 80}\n=== {method_name} ===\n{'=' * 80}")
            cfg = build_cfg(augmentation, pooling, quick=quick)
            for stock_id, df_one in stocks:
                try:
                    res = run_stock(df_one.sort_values("trade_date"), stock_id, cfg)
                except Exception as e:
                    print(f"\n[ERROR] stock {stock_id} skipped: {e}")
                    continue
                if res is None:
                    continue
                per_stock_rows.append({
                    "augmentation": augmentation,
                    "pooling": pooling,
                    "stock_id": stock_id,
                    "MAE": res["MAE"], "RMSE": res["RMSE"],
                    "sMAPE": res["sMAPE"], "DA": res["DA"], "R2": res["R2"],
                })

    detail_df = pd.DataFrame(per_stock_rows)
    summary_df = (
        detail_df.groupby(["augmentation", "pooling"])[["MAE", "RMSE", "sMAPE", "DA", "R2"]]
        .mean()
        .reset_index()
    )
    return detail_df, summary_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-assets", type=int, default=10,
                    help="number of (alphabetically-first) stocks to use (default 10, matching the book)")
    ap.add_argument("--quick", action="store_true",
                    help="drastically reduce epochs/days for a fast smoke test")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df, summary_df = run_grid(args.max_assets, quick=args.quick)
    detail_df.to_csv(OUTPUT_DIR / "baseline_grid_per_stock.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "baseline_grid_summary.csv", index=False)

    print(f"\n{'=' * 80}\nSummary (mean over {args.max_assets} stocks)\n{'=' * 80}")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {len(detail_df)} per-stock rows and {len(summary_df)} summary rows to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
