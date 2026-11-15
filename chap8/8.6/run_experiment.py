"""Unified entry point for the Section 8.6 experiment.

Usage:

    python run_experiment.py                      # default: max_assets=180 (matches the book's original script)
    python run_experiment.py --max-assets 30
    python dynamic_graph_transformer.py            # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from backtest import EnhancedBacktestEngine
from config import SHARED
from data_utils import create_enhanced_dataset, load_real_data
from explain import ModelExplainer
from hypersearch import HyperparameterSearchEngine
from model import E_DyGFormer
from metrics import evaluate_ic, _eval_ic_with_risk
from paths import data_file
from train_utils import train_enhanced_model

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(s: int = 42) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(SHARED["seed"])

#: Output root. 
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def run(max_assets: Optional[int] = None, max_days: Optional[int] = None) -> dict:
    """
    Run the full pipeline: data hyperparameter search, training, Spearman-IC
    consistency evaluation, explainability analysis, and backtest (formerly
    this section's module-level ``enhanced_main()``).

    Parameters:
    - max_assets: number of stocks to use; defaults to config.py's SHARED value (180)
    - max_days: number of most-recent trading days to use; defaults to config.py's SHARED value (500)
    """
    cfg = copy.deepcopy(SHARED)
    if max_assets is None:
        max_assets = cfg["max_assets"]
    if max_days is None:
        max_days = cfg["max_days"]

    run_name = f"max_assets_{max_assets}"
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        print("=" * 80)
        print("Edge-aware Dynamic Graph Transformer quantitative-investing system (E-DyGFormer)")
        print("Features: edge awareness, transaction costs, explainability, hyperparameter search (incl. corr_window/corr_threshold)")
        print("=" * 80)

        # 1) Data
        default_file = str(data_file(cfg["data_file"]))
        if not os.path.exists(default_file):
            raise FileNotFoundError(f"Data file not found: {default_file}")

        print(f"Asset pool size: {max_assets}, lookback days: {max_days}")
        price_data, benchmark_data, panel_data = load_real_data(
            default_file,
            max_assets=max_assets,
            max_days=max_days
        )

        # 2) Build the dataset once with the default corr hyperparameters, to get basic info
        default_corr_window = cfg["default_corr_window"]
        default_corr_threshold = cfg["default_corr_threshold"]
        seq_len = cfg["seq_len"]
        dataset, feat_names = create_enhanced_dataset(
            panel_data, seq_len=seq_len,
            corr_window=default_corr_window,
            corr_threshold=default_corr_threshold
        )
        if len(dataset) == 0:
            print("Dataset construction failed")
            return None

        # Initial split
        n_all = len(dataset)
        train_size = int(cfg["train_ratio"] * n_all)
        val_size = int(cfg["val_ratio"] * n_all)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
        print(f"Split: train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}")

        # 3) Hyperparameter search (corr_window/corr_threshold only; model-structure params are fixed)
        feat_dim = train_dataset[0].x.shape[1]
        num_assets = len(price_data.columns)
        search_cfg = cfg["search"]
        search_space = search_cfg["search_space"]
        fixed_params = {
            "node_feat_dim": feat_dim,  # overridden by the actual data dimension at evaluation time
            "num_assets": num_assets,
            "seq_len": seq_len,
            **cfg["model"],
        }
        searcher = HyperparameterSearchEngine(E_DyGFormer, search_space, search_method=search_cfg["search_method"])
        search_results = searcher.search(
            train_dataset, val_dataset, fixed_params, max_trials=search_cfg["max_trials"],
            panel_data=panel_data, seq_len=seq_len, feature_cols=feat_names,
            fast_epochs=search_cfg["fast_train_epochs"], batch_size=search_cfg["fast_train_batch_size"],
            lr=search_cfg["fast_train_lr"],
        )

        # 4) Rebuild the dataset with the best data hyperparameters & re-split
        best_corr_window = int(search_results["best_params"].get("corr_window", default_corr_window))
        best_corr_threshold = float(search_results["best_params"].get("corr_threshold", default_corr_threshold))
        dataset_best, _ = create_enhanced_dataset(
            panel_data, seq_len=seq_len,
            corr_window=best_corr_window,
            corr_threshold=best_corr_threshold,
            feature_cols=feat_names
        )
        n_all = len(dataset_best)
        train_size = int(cfg["train_ratio"] * n_all)
        val_size = int(cfg["val_ratio"] * n_all)
        train_dataset = dataset_best[:train_size]
        val_dataset = dataset_best[train_size:train_size + val_size]
        test_dataset = dataset_best[train_size + val_size:]
        print(f"[Best data hyperparameters] corr_window={best_corr_window}, corr_threshold={best_corr_threshold}")
        print(f"Re-split: train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}")

        # 5) Train the best model
        best_config = dict(fixed_params)
        best_config["node_feat_dim"] = train_dataset[0].x.shape[1]  # overridden by the new data dimension
        model = E_DyGFormer(**best_config)
        train_cfg = cfg["train"]
        model = train_enhanced_model(
            model, train_dataset, val_dataset, epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"], lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
            scheduler_factor=train_cfg["scheduler_factor"], scheduler_patience=train_cfg["scheduler_patience"],
            grad_clip_norm=train_cfg["grad_clip_norm"], risk_reg_weight=train_cfg["risk_reg_weight"],
            loss_alpha=train_cfg["loss_alpha"], loss_beta=train_cfg["loss_beta"], val_every=train_cfg["val_every"],
        )

        # 6) Evaluate consistency (Spearman-IC): val/test sets (both raw-prediction and risk-adjusted conventions)
        val_ic = evaluate_ic(model, val_dataset)
        test_ic = evaluate_ic(model, test_dataset)
        val_ic_ra = _eval_ic_with_risk(model, val_dataset)
        test_ic_ra = _eval_ic_with_risk(model, test_dataset)
        print(f"[Spearman-IC] val (raw) mean={val_ic['ic_mean']:.4f} (n={val_ic['n']}) | test (raw) mean={test_ic['ic_mean']:.4f} (n={test_ic['n']})")
        print(f"[Spearman-IC-RA] val (risk-adjusted) mean={val_ic_ra['ic_mean']:.4f} (n={val_ic_ra['n']}) | test (risk-adjusted) mean={test_ic_ra['ic_mean']:.4f} (n={test_ic_ra['n']})")

        # 7) Explainability
        explain_cfg = cfg["explain"]
        explainer = ModelExplainer(model, feature_names=feat_names)
        explanations = explainer.analyze_model_behavior(val_dataset[:explain_cfg["n_samples"]], n_samples=explain_cfg["n_samples"])
        explainer.visualize_explanations(explanations, save_path="model_interpretability.png")
        sample_for_explain = val_dataset[0] if len(val_dataset) > 0 else train_dataset[0]
        _ = explainer.visualize_heatmap_and_bars(
            sample_for_explain, agg=explain_cfg["heatmap_agg"], feature_idx=None,
            top_k=explain_cfg["heatmap_top_k"], time_idx=explain_cfg["heatmap_time_idx"],
            asset_idx=explain_cfg["heatmap_asset_idx"],
            save_path="explain_heatmap_bars.png"
        )

        # 8) Backtest
        backtest_cfg = cfg["backtest"]
        backtest_engine = EnhancedBacktestEngine(
            initial_capital=backtest_cfg["initial_capital"],
            transaction_cost_rate=backtest_cfg["transaction_cost_rate"],
            impact_cost_factor=backtest_cfg["impact_cost_factor"],
            min_trade_amount=backtest_cfg["min_trade_amount"],
            max_position_ratio=backtest_cfg["max_position_ratio"],
        )
        test_limited = test_dataset[:min(backtest_cfg["test_limit"], len(test_dataset))]
        price_test = price_data.iloc[-len(test_limited):]
        bench_test = benchmark_data.iloc[-len(test_limited):]
        results = backtest_engine.run_backtest(model, test_limited, price_test, bench_test)

        # 9) Print the results
        print("\n" + "=" * 80)
        print("Enhanced backtest results (including transaction costs)")
        print("=" * 80)
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Benchmark return: {results['benchmark_total_return']:.2%}")
        print(f"Excess return: {results['excess_total_return']:.2%}")
        print(f"Annualized return: {results['annual_return']:.2%}")
        print(f"Annualized volatility: {results['annual_volatility']:.2%}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        print(f"Win rate: {results['win_rate']:.2%}")
        print(f"Transaction cost ratio: {results['transaction_cost_ratio']:.3%}")
        print(f"Average turnover rate: {results['avg_turnover_rate']:.2%}")
        print(f"Information ratio: {results['information_ratio']:.3f}")
        print(f"Average holdings (nonzero value): {results['avg_holdings']:.2f}")
        print(f"Average effective holdings (>= min trade amount): {results['avg_holdings_effective']:.2f}")
        print(f"[IC summary, raw] val mean={val_ic['ic_mean']:.4f} (n={val_ic['n']}), test mean={test_ic['ic_mean']:.4f} (n={test_ic['n']})")
        print(f"[IC summary, risk-adjusted] val mean={val_ic_ra['ic_mean']:.4f} (n={val_ic_ra['n']}), test mean={test_ic_ra['ic_mean']:.4f} (n={test_ic_ra['n']})")
        print(f"\nBest parameters (including data hyperparameters): {search_results['best_params']}")
        print(f"Best validation score (IC): {search_results['best_score']:.4f}")

        return {
            "model": model,
            "backtest_results": results,
            "search_results": search_results,
            "explanations": explanations,
            "backtest_engine": backtest_engine,
            "val_ic": val_ic,
            "test_ic": test_ic,
        }
    finally:
        os.chdir(prev_cwd)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Section 8.6: edge-aware dynamic graph Transformer for quantitative investing",
    )
    ap.add_argument("--max-assets", type=int, default=None,
                    help=f"number of stocks to use (default {SHARED['max_assets']}, matching the book)")
    ap.add_argument("--max-days", type=int, default=None,
                    help=f"number of most-recent trading days to use (default {SHARED['max_days']})")
    return ap.parse_args()


if __name__ == "__main__":
    from visualize import plot_enhanced_results

    args = parse_args()
    print("E-DyGFormer (Edge-aware Dynamic Graph Transformer)......")
    print("Features: transaction costs, explainability (gradient-based), hyperparameter search (incl. corr_window/corr_threshold)")
    results = run(max_assets=args.max_assets, max_days=args.max_days)
    if isinstance(results, dict):
        print("\nGenerating the analysis plots...")
        plot_enhanced_results(
            results['backtest_results'],
            results.get('search_results'),
            portfolio_values=getattr(results.get('backtest_engine', None), 'portfolio_value', None),
            benchmark_levels=getattr(results.get('backtest_engine', None), 'benchmark_levels', None),
            save_path='enhanced_analysis_results.png'
        )
        print("\nDone!")
    print("\nProgram finished.")
