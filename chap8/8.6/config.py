"""Unified configuration for Section 8.6's edge-aware dynamic graph Transformer.

Usage:
    from config import SHARED, ASSET_POOL_GRID, EXPLAIN_MAX_ASSETS
    import copy
    cfg = copy.deepcopy(SHARED)
"""

from __future__ import annotations

from typing import Dict

#: Parameters shared by every run, written as plain literals so every
#: value is visible in one place.
SHARED: Dict = {
    "seed": 42,
    "data_file": "CSI100_stock_data_with_factors.pkl",

    # ---- data loading (load_real_data) ----
    "max_assets": 180,     #
    "max_days": 500,

    # ---- dataset construction (create_enhanced_dataset) ----
    "seq_len": 20,
    "default_corr_window": 5,
    "default_corr_threshold": 0.3,

    # ---- train/val/test split ----
    "train_ratio": 0.6,
    "val_ratio": 0.2,           # remainder goes to the test set

    # ---- model (E_DyGFormer) ----
    "model": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "edge_bias_scale": 0.3,
        "edge_bias_temp": 2.0,
        "use_signed_agg": False,
    },

    # ---- data-hyperparameter search (HyperparameterSearchEngine) ----
    # Only corr_window/corr_threshold are searched; the model-structure
    # parameters above stay fixed during the search.
    "search": {
        "search_space": {
            "corr_window": [5, 10, 15],
            "corr_threshold": [0.3, 0.5, 0.7],
        },
        "search_method": "grid",
        "max_trials": 32,
        "fast_train_epochs": 5,   # epochs used inside the search loop to score each combo
        "fast_train_batch_size": 8,
        "fast_train_lr": 0.001,
    },

    # ---- final-model training (train_enhanced_model) ----
    "train": {
        "epochs": 15,
        "batch_size": 8,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "scheduler_factor": 0.7,
        "scheduler_patience": 5,
        "grad_clip_norm": 1.0,
        "risk_reg_weight": 1e-3,     # weight on risks.mean() added to the loss
        "loss_alpha": 1.0,           # mse_ic_loss weight on MSE
        "loss_beta": 0.5,            # mse_ic_loss weight on -IC
        "val_every": 5,              # validate every N epochs
    },

    # ---- explainability (ModelExplainer) ----
    "explain": {
        "n_samples": 5,
        "heatmap_agg": "sum",
        "heatmap_top_k": 10,
        "heatmap_time_idx": "last",
        "heatmap_asset_idx": 1,
    },

    # ---- backtest (EnhancedBacktestEngine) ----
    "backtest": {
        "initial_capital": 1_000_000,
        "transaction_cost_rate": 0.002,
        "impact_cost_factor": 0.001,
        "min_trade_amount": 5_000,
        "max_position_ratio": 0.2,
        "test_limit": 50,     # at most this many days of the test set are backtested
    },
}


ASSET_POOL_GRID = [30, 60, 90, 120, 150, 180]

# Gradient-based explainability analysis, specifically at max_assets=30. 
EXPLAIN_MAX_ASSETS = 30
