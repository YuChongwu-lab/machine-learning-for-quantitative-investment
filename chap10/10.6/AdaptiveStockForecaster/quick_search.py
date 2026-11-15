
"""
Quick hyperparameter search (simplified version).
========================================
For quickly testing and validating the search workflow.Search space:
1. diffusion.mode: ['faithful', 'adversarial', 'mixed']
2. tcvae.pooling: ['mean', 'max', 'last', 'attention']
3. seq_len: [24, 48, 60, 96]
"""

import warnings
import numpy as np
import optuna
from data import load_real_data
from run_experiment import run_stock

warnings.filterwarnings('ignore')

# =================== Simplified config ===================
import copy
from config import BASE_CFG
BASE_CFG = copy.deepcopy(BASE_CFG)
BASE_CFG["max_assets"] = 1  # only run 1 stock for the quick search
# Note: seq_len / tcvae.pooling / diffusion.mode are overridden during the search


# =================== Reduced search space (only 3 parameters) ===================
def simple_search_space(trial):
    """
    Searches only 3 key parameters:
    1. diffusion.mode
    2. tcvae.pooling
    3. seq_len
    """
    # Search diffusion.mode
    diffusion_mode = trial.suggest_categorical(
        "diffusion_mode",
        ["faithful", "adversarial", "mixed"]
    )

    # Search tcvae.pooling
    tcvae_pooling = trial.suggest_categorical(
        "tcvae_pooling",
        ["mean", "max", "last", "attention"]
    )

    # Search seq_len
    seq_len = trial.suggest_categorical(
        "seq_len",
        [24, 48, 60, 96]
    )

    return {
        "seq_len": seq_len,  # update seq_len
        "tcvae": {
            "latent_dim": 64,
            "hidden_dim": 128,
            "pooling": tcvae_pooling,  # update pooling
        },
        "diffusion": {
            "enable": True,
            "mode": diffusion_mode,  #update mode
            "pretrain_epochs": 5,
            "augment_prob": 0.35,
            "augment_ratio": 1.0,
            "adversarial_strength": 0.3,
            "hidden_dim": 64,
            "context_dim": 64,
            "save_dir": "checkpoints/diffusion_models",
        },
    }


# =================== Objective function ===================
def objective(trial):
    """The quick-test objective function."""
    #  Deep-copy BASE_CFG to avoid contaminating the original config
    import copy
    cfg = copy.deepcopy(BASE_CFG)

    # Get the search parameters
    search_params = simple_search_space(trial)

    # Update the config (deep update)
    cfg["seq_len"] = search_params["seq_len"]
    cfg["tcvae"].update(search_params["tcvae"])
    cfg["diffusion"].update(search_params["diffusion"])

    # Sync tcvae_pooling to the top level
    cfg["tcvae_pooling"] = search_params["tcvae"]["pooling"]

    try:
        price, bench, panel_data = load_real_data(
            cfg["data_path"], cfg["max_assets"], cfg["max_days"]
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        return 0.0

    da_scores = []
    for stock_id, df_one in panel_data.groupby("ts_code"):
        try:
            res = run_stock(df_one.sort_values("trade_date"), stock_id, cfg)
            if res is not None:
                da_scores.append(res["DA"])
        except Exception as e:
            print(f"Stock {stock_id} failed: {e}")
            continue

    return float(np.mean(da_scores)) if da_scores else 0.0


# =================== Quick search ===================
def quick_search(n_trials=10):
    """Run a quick search (10 trials, roughly 10-15 minutes)."""

    print("=" * 60)
    print("Quick hyperparameter search")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Search parameters:")
    print(f"  1. diffusion.mode: ['faithful', 'adversarial', 'mixed']")
    print(f"  2. tcvae.pooling: ['mean', 'max', 'last', 'attention']")
    print(f"  3. seq_len: [24, 48, 60, 96]")
    print(f"Number of stocks: {BASE_CFG['max_assets']} (for a faster test)")
    print("=" * 60 + "\n")

    study = optuna.create_study(
        study_name="quick_search",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print the results
    print("\n" + "=" * 60)
    print("Search complete!")
    print("=" * 60)
    print(f"\nBest DA: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save the results
    df = study.trials_dataframe()
    df.to_csv("checkpoints/quick_search_results.csv", index=False)
    print(f"\nResults saved: checkpoints/quick_search_results.csv")

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    study = quick_search(args.trials)
