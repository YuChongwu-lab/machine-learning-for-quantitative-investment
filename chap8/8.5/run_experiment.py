"""Unified entry point for the Section 8.5 experiments.

Usage:

    python run_experiment.py --experiment all-softmax
    python run_experiment.py --list
    python hierarchical_multi_edge_gnn.py       # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

from backtest import BacktestEngine
from config import EXPERIMENTS, DEFAULT_EXPERIMENT, get_config
from data_utils import FeatureEngineer, load_real_data
from model import HierarchicalGNN
from paths import data_file
from portfolio import EnhancedPortfolioOptimizer
from train_utils import train_model
from visualize import debug_edge_construction, plot_backtest_results, plot_temporal_edge_dynamics, visualize_edge_types

#: Output root. 
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def run(name: str,
        data_file_path: Optional[str] = None,
        benchmark_file_path: Optional[str] = None,
        edge_types: Optional[Set[str]] = None,
        optimization_method: Optional[str] = None):
    """
    Run one named experiment from config.EXPERIMENTS (formerly this
    section's module-level ``main()``).

    Parameters:
    - name: experiment name in config.EXPERIMENTS
    - data_file_path: path to the real data file (defaults to config.py's data_file, resolved under data/)
    - benchmark_file_path: path to the benchmark data file (optional)
    - edge_types: edge types to use; defaults to the experiment's config
    - optimization_method: portfolio-construction method; defaults to the experiment's config
    """
    cfg = get_config(name)
    if edge_types is None:
        edge_types = cfg["edge_types"]
    if optimization_method is None:
        optimization_method = cfg["optimization_method"]
    if data_file_path is None:
        data_file_path = str(data_file(cfg["data_file"]))

    print("=" * 60)
    print("Hierarchical Multi-Edge-Type Dynamic Graph Neural Network for Quantitative Investing")
    print("=" * 60)
    print(f"Experiment: {name}")
    print(f"Edge types in use: {edge_types}")
    print(f"Optimization method: {optimization_method}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the data
    print("\n1. Loading the real data...")
    price_data, benchmark_data = load_real_data(data_file_path, benchmark_file_path)

    if data_file_path.endswith('.pkl'):
        panel_data = pd.read_pickle(data_file_path)
    elif data_file_path.endswith('.csv'):
        panel_data = pd.read_csv(data_file_path)
    else:
        panel_data = pd.read_excel(data_file_path)

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])

    # 2. Feature engineering and configurable-edge-type graph-dataset construction
    print("\n2. Feature engineering and configurable multi-edge-type graph-dataset construction...")
    feature_engineer = FeatureEngineer()
    dataset = feature_engineer.create_dataset_with_configurable_edges(
        price_data=price_data,
        panel_data=panel_data,
        seq_len=cfg["seq_len"],
        edge_types=edge_types,
        max_stocks=cfg["max_stocks"],
        lookback_days=cfg["lookback_days"],
        sample_interval=cfg["sample_interval"],
        correlation_threshold=cfg["correlation_threshold"],
        knn_k=cfg["knn_k"],
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty, exiting")
        return None, None, None

    # 3. Edge-type visualization
    debug_edge_construction(dataset, sample_idx=0)
    print("\n3. Visualizing the edge-type structure...")
    sector_mapping = {}
    if 'industry' in panel_data.columns:
        sector_info = panel_data.groupby('ts_code')['industry'].first()
        sector_mapping = sector_info.to_dict()

    visualize_edge_types(dataset, price_data, sector_mapping, sample_idx=0)
    plot_temporal_edge_dynamics(dataset, n_samples=10)

    # 4. Split the dataset
    train_size = int(cfg["train_ratio"] * len(dataset))
    val_size = int(cfg["val_ratio"] * len(dataset))

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    print(f"Train set: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}")

    # 5. Build the model
    print("\n4. Building the hierarchical multi-edge-type model...")
    sample_nodes = train_dataset[0].x.shape[0]
    seq_len = cfg["seq_len"]
    actual_num_assets = sample_nodes // seq_len

    print(f"Number of stocks inferred from the dataset: {actual_num_assets}")

    model_config = {
        'feature_dim': train_dataset[0].x.shape[1],
        'hidden_dim': cfg["model"]["hidden_dim"],
        'num_layers': cfg["model"]["num_layers"],
        'num_assets': actual_num_assets,
        'seq_len': seq_len,
        'conv_type': cfg["model"]["conv_type"],
        'use_edge_weights': cfg["model"]["use_edge_weights"],
    }

    model = HierarchicalGNN(model_config)
    model = model.to(device)

    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train the model
    print("\n5. Training the hierarchical model...")
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = train_model(
        model, train_dataset, val_dataset,
        epochs=cfg["train"]["epochs"],
        patience=cfg["train"]["patience"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        scheduler_factor=cfg["train"]["scheduler_factor"],
        scheduler_patience=cfg["train"]["scheduler_patience"],
        batch_size=cfg["train"]["batch_size"],
        grad_clip_norm=cfg["train"]["grad_clip_norm"],
        checkpoint_path=str(out_dir / "best_hierarchical_model.pth"),
    )

    # 7. Backtest (supports several portfolio-construction methods)
    print(f"\n6. Running the backtest ({optimization_method})...")
    optimizer = EnhancedPortfolioOptimizer(risk_aversion=cfg["portfolio"]["risk_aversion"])
    backtest_engine = BacktestEngine(
        initial_capital=cfg["backtest"]["initial_capital"],
        transaction_cost=cfg["backtest"]["transaction_cost"],
    )

    # Make sure the test-set indices line up
    actual_price_data = getattr(feature_engineer, 'price_data_used', price_data)

    test_start_idx = max(0, len(actual_price_data) - len(test_dataset))
    test_price_data = actual_price_data.iloc[test_start_idx:]
    test_benchmark_data = benchmark_data.iloc[test_start_idx:test_start_idx + len(test_price_data)]

    # Make sure all three have the same length
    min_len = min(len(test_price_data), len(test_benchmark_data), len(test_dataset))
    test_price_data = test_price_data.iloc[:min_len]
    test_benchmark_data = test_benchmark_data.iloc[:min_len]
    test_dataset = test_dataset[:min_len]

    results = backtest_engine.run_backtest(
        model=model,
        test_dataset=test_dataset,
        price_data=test_price_data,
        benchmark_data=test_benchmark_data,
        optimizer=optimizer,
        optimization_method=optimization_method
    )

    # 8. Show the results
    print("\n" + "=" * 60)
    print(f"Hierarchical Gated Dynamic GNN Backtest Results ({optimization_method.upper()})")
    print("=" * 60)

    print(f"\nTotal return: {results['total_return']:.2%}")
    print(f"Annualized return: {results['annual_return']:.2%}")
    print(f"Annualized volatility: {results['annual_volatility']:.2%}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {results['max_drawdown']:.2%}")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Final portfolio value: ${results['final_portfolio_value']:,.2f}")

    # 9. Generate the visualization
    print("\n7. Generating the visualization...")
    plot_backtest_results(
        backtest_results=results,
        portfolio_values=backtest_engine.portfolio_value,
        benchmark_data=test_benchmark_data,
        dates=test_price_data.index,
        optimization_method=optimization_method
    )

    print("\n" + "=" * 60)
    print("Hierarchical gated dynamic GNN analysis complete!")
    print("=" * 60)

    return model, results, backtest_engine


def parse_args():
    ap = argparse.ArgumentParser(
        description="Section 8.5: hierarchical multi-edge-type dynamic graph neural network for quantitative investing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available experiments:\n" + "\n".join(
            f"  {k:20} edges={sorted(v['edge_types'])} optimization={v['optimization_method']}"
            for k, v in EXPERIMENTS.items()))
    ap.add_argument("--experiment", "-e", default=DEFAULT_EXPERIMENT,
                    choices=list(EXPERIMENTS),
                    help=f"experiment to run (default {DEFAULT_EXPERIMENT}, the only example block "
                         f"originally left uncommented)")
    ap.add_argument("--list", action="store_true", help="list all experiments and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print(f"{'Experiment':22}{'Edge types':42}{'Optimization method'}")
        for name, exp_cfg in EXPERIMENTS.items():
            mark = "  <- default" if name == DEFAULT_EXPERIMENT else ""
            print(f"{name:22}{str(sorted(exp_cfg['edge_types'])):42}{exp_cfg['optimization_method']}{mark}")
        raise SystemExit(0)

    print(f"Running experiment: {args.experiment}  edge_types={sorted(EXPERIMENTS[args.experiment]['edge_types'])}  "
          f"optimization_method={EXPERIMENTS[args.experiment]['optimization_method']}")

    model, results, backtest_engine = run(args.experiment)
