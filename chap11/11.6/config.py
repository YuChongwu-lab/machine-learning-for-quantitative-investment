"""Unified configuration for Section 11.6's curriculum + multi-objective PPO
graph-RL experiments.

Usage:
    from config import SHARED, EXPERIMENTS, get_config
    cfg = get_config("morl_curriculum-itransformer-30")
"""

from __future__ import annotations

import copy
from typing import Dict

#: Parameters shared by every experiment, written as plain literals so
#: every value is visible in one place.
SHARED: Dict = {
    "seed": 42,
    "data_file": "CSI100_stock_data_with_factors.pkl",
    "max_days": 500,
    "seq_len": 20,

    "dataset": {
        "corr_window": 10,
        "corr_threshold": 0.3,
    },

    "train_ratio": 0.7,
    "val_ratio": 0.2,           # remainder goes to the test set

    "encoder": {
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.1,
        "edge_bias_scale": 0.3,
        "edge_bias_temp": 2.0,
        "use_signed_agg": False,
    },

    "policy": {
        "hidden": 256,
        "alpha_scale": 0.5,
        "alpha_min": 1e-3,
        "temperature": 10,
        "max_pos": None,
        "use_cls_for_value": True,
        "freeze_encoder": False,
    },

    "engine": {
        "initial_capital": 1_000_000,
        "transaction_cost_rate": 0.002,
        "impact_cost_factor": 0.001,
        "min_trade_amount": 5_000,
        "max_position_ratio": 0.2,
    },

    "reward": {
        "lambda_turnover": 0.05,
    },

    "morl": {
        "morl_aux_weight": 0.5,
        "morl_alpha": 1.0,
        "morl_beta": 0.5,
    },

    "trainer": {
        "gamma": 0.99,
        "lam": 0.95,
        "clip_ratio": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "lr": 3e-4,
        "minibatch_size": 16,
        "update_epochs": 4,
    },

    "horizon": 128,
    "flat_epochs": 30,  # total PPO epochs when use_curriculum=False (matches the curriculum groups' 8+10+12=30)

    "adapter": {
        "smooth_weights": True,
        "smooth_rho": 0.3,
    },
}

# Table 11.2 ablation groups

GROUPS: Dict[str, Dict] = {
    "baseline":        {"use_curriculum": False, "use_morl": False},
    "morl":            {"use_curriculum": False, "use_morl": True},
    "curriculum":      {"use_curriculum": True,  "use_morl": False},
    "morl_curriculum": {"use_curriculum": True,  "use_morl": True},
}

ENCODER_TYPES = ["vanilla", "itransformer"]
MAX_ASSETS_GRID = [30, 120]

EXPERIMENTS: Dict[str, Dict] = {
    f"{group}-{encoder}-{max_assets}": {
        "group": group,
        "encoder_type": encoder,
        "max_assets": max_assets,
    }
    for group in GROUPS
    for encoder in ENCODER_TYPES
    for max_assets in MAX_ASSETS_GRID
}

DEFAULT_EXPERIMENT = "morl_curriculum-itransformer-30"


def get_config(name: str) -> Dict:
    """Return a deep copy of SHARED merged with the named experiment's group/encoder/max_assets."""
    exp = EXPERIMENTS[name]
    cfg = copy.deepcopy(SHARED)
    cfg["use_curriculum"] = GROUPS[exp["group"]]["use_curriculum"]
    cfg["use_morl"] = GROUPS[exp["group"]]["use_morl"]
    cfg["encoder_type"] = exp["encoder_type"]
    cfg["max_assets"] = exp["max_assets"]
    cfg["experiment"] = name
    cfg["group"] = exp["group"]
    return cfg
