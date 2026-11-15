"""Unified configuration for Section 8.5's hierarchical multi-edge GNN experiments.

Usage:
    from config import SHARED, EXPERIMENTS, get_config
    cfg = get_config("all-softmax")
"""

from __future__ import annotations

import copy
from typing import Dict

#: Parameters shared by every experiment (data, model, training, backtest).
#: Written as plain literals, not assembled via dict-unpacking, so every
#: value is visible in one place.
SHARED: Dict = {
    "seed": 42,
    "data_file": "CSI100_stock_data_with_factors.pkl",

    # ---- dataset construction (FeatureEngineer.create_dataset_with_configurable_edges) ----
    "seq_len": 20,
    "max_stocks": 190,          # keep only the 190 stocks with the most data points
    "lookback_days": 2500,      # trailing window of calendar days to use
    "sample_interval": 5,       # stride between consecutive samples
    "correlation_threshold": 0.4,
    "knn_k": 8,

    # ---- train/val/test split ----
    "train_ratio": 0.6,
    "val_ratio": 0.2,           # remainder goes to the test set

    # ---- model (HierarchicalGNN) ----
    "model": {
        "hidden_dim": 64,
        "num_layers": 3,
        "conv_type": "GAT",
        "use_edge_weights": True,
    },

    # ---- training (ModelTrainer / train_model) ----
    "train": {
        "epochs": 50,
        "patience": 20,
        "batch_size": 16,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.7,
        "scheduler_patience": 8,
        "grad_clip_norm": 0.5,
    },

    # ---- portfolio construction (EnhancedPortfolioOptimizer) ----
    "portfolio": {
        "risk_aversion": 2.0,
    },

    # ---- backtest (BacktestEngine) ----
    "backtest": {
        "initial_capital": 1_000_000,
        "transaction_cost": 0.001,
    },
}

#: All four edge types, used by the "all edges" experiments below.
ALL_EDGES = {"correlation", "sector", "knn", "temporal"}

# ============================================================================
# Experiment registry
# ============================================================================

EXPERIMENTS: Dict[str, Dict] = {
    # ---- single edge type (ablation: each edge structure's standalone contribution) ----
    "sector":            {"edge_types": {"sector"},      "optimization_method": "softmax_long_only"},
    "correlation":       {"edge_types": {"correlation"}, "optimization_method": "softmax_long_only"},
    "knn":               {"edge_types": {"knn"},         "optimization_method": "softmax_long_only"},
    "temporal":          {"edge_types": {"temporal"},    "optimization_method": "softmax_long_only"},
    # ---- edge types stacked incrementally ----
    "corr-sector":       {"edge_types": {"correlation", "sector"},        "optimization_method": "softmax_long_only"},
    "corr-sector-knn":   {"edge_types": {"correlation", "sector", "knn"}, "optimization_method": "softmax_long_only"},
    # ---- all edge types x four portfolio-construction methods ----
    "all-softmax":       {"edge_types": ALL_EDGES, "optimization_method": "softmax_long_only"},
    "all-min-variance":  {"edge_types": ALL_EDGES, "optimization_method": "min_variance"},
    "all-risk-parity":   {"edge_types": ALL_EDGES, "optimization_method": "risk_parity"},
    "all-mean-variance": {"edge_types": ALL_EDGES, "optimization_method": "mean_variance"},
}

DEFAULT_EXPERIMENT = "all-softmax"


def get_config(name: str) -> Dict:
    """Return a deep copy of SHARED merged with EXPERIMENTS[name]."""
    cfg = copy.deepcopy(SHARED)
    cfg.update(copy.deepcopy(EXPERIMENTS[name]))
    cfg["experiment"] = name
    return cfg
