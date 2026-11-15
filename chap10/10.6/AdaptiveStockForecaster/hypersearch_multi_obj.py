
"""
===============================================================================
Multi-objective hyperparameter search.
===============================================================================
Functionality:
1. jointly optimizes MSE (minimize) and DA (maximize)
2. search parameters: diffusion.mode, tcvae.pooling, seq_len
3. automatically generates the Pareto front and recommendation strategies
4. optional visualization
===============================================================================
"""
import warnings
import numpy as np
import pandas as pd
import optuna
import os
import sys
from pathlib import Path

# Make sure the checkpoints directory exists
Path("checkpoints").mkdir(exist_ok=True)

warnings.filterwarnings('ignore')

# =================== Import the main system ===================
try:
    from data import load_real_data
    from run_experiment import run_stock

    print("✓ Successfully imported the main system modules")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Make sure run_experiment.py and data.py are in the current directory")
    sys.exit(1)

# =================== Global config ===================
import copy
from config import BASE_CFG
BASE_CFG = copy.deepcopy(BASE_CFG)
BASE_CFG["max_assets"] = 1  # only run 1 stock during the search, for speed


# =================== Search-space definition ===================
def simple_search_space(trial):
    """
    Define the hyperparameter search space.
    Search parameters:
    1. diffusion_mode: the data-augmentation strategy
    2. tcvae_pooling: the concept-encoder pooling strategy
    3. seq_len: the input sequence length
    """
    diffusion_mode = trial.suggest_categorical(
        "diffusion_mode",
        ["faithful", "adversarial", "mixed"]
    )

    tcvae_pooling = trial.suggest_categorical(
        "tcvae_pooling",
        ["mean", "max", "last", "attention"]
    )

    seq_len = trial.suggest_categorical(
        "seq_len",
        [24, 48, 60, 96]
    )

    return {
        "seq_len": seq_len,
        "tcvae": {
            "latent_dim": 64,
            "hidden_dim": 128,
            "pooling": tcvae_pooling,
        },
        "diffusion": {
            "enable": True,
            "mode": diffusion_mode,
            "pretrain_epochs": 5,
            "augment_prob": 0.35,
            "augment_ratio": 1.0,
            "adversarial_strength": 0.3,
            "hidden_dim": 64,
            "context_dim": 64,
            "save_dir": "checkpoints/diffusion_models",
        },
    }


# =================== Multi-objective function ===================
def objective_multi(trial):
    """
    The multi-objective optimization function.
    Returns:
        tuple: (MSE, -DA)
            - MSE: mean squared error (lower is better)
            - -DA: negative direction accuracy (negated so both objectives
              are minimized uniformly; the original DA is better when higher)
    """
    import copy

    # Deep-copy the config to avoid cross-trial contamination
    cfg = copy.deepcopy(BASE_CFG)

    # Get this trial's hyperparameters
    search_params = simple_search_space(trial)

    # Update the config
    cfg["seq_len"] = search_params["seq_len"]
    cfg["tcvae"].update(search_params["tcvae"])
    cfg["diffusion"].update(search_params["diffusion"])
    cfg["tcvae_pooling"] = search_params["tcvae"]["pooling"]  # for compatibility with the main system

    # Load the data
    try:
        price, bench, panel_data = load_real_data(
            cfg["data_path"],
            cfg["max_assets"],
            cfg["max_days"]
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        return float('inf'), float('inf')

    # Collect MSE and DA across stocks
    mse_scores = []
    da_scores = []

    for stock_id, df_one in panel_data.groupby("ts_code"):
        try:
            # run a single stock
            res = run_stock(df_one.sort_values("trade_date"), stock_id, cfg)

            if res is not None:
                # extract RMSE and compute MSE
                if "RMSE" in res and not np.isnan(res["RMSE"]):
                    mse = res["RMSE"] ** 2
                    mse_scores.append(mse)

                # extract DA
                if "DA" in res and not np.isnan(res["DA"]):
                    da_scores.append(res["DA"])

        except Exception as e:
            print(f"Stock {stock_id} failed: {e}")
            continue

    # Check whether there are any valid results
    if not mse_scores or not da_scores:
        print(f" Trial {trial.number}: no valid results")
        return float('inf'), float('inf')

    # Make sure the lengths match (truncate to the shorter one)
    min_len = min(len(mse_scores), len(da_scores))
    mse_scores = mse_scores[:min_len]
    da_scores = da_scores[:min_len]

    # Compute the averages
    avg_mse = float(np.mean(mse_scores))
    avg_da = float(np.mean(da_scores))

    # Live output
    print(f" Trial {trial.number}: MSE={avg_mse:.6f}, DA={avg_da:.4f} "
          f"[{search_params['diffusion']['mode']}, "
          f"{search_params['tcvae']['pooling']}, "
          f"len={search_params['seq_len']}]")

    # Return both objectives (Optuna always minimizes, so DA is negated)
    return avg_mse, -avg_da


# =================== Helper: save all trials ===================
def _save_all_trials(study, pareto_trials):
    """
    Save all trial results to a CSV file.
    Args:
        study: the Optuna study object
        pareto_trials: the list of Pareto-optimal trials
    """
    all_trials_data = []# holds the data for every trial
    pareto_numbers = {t.number for t in pareto_trials}# the trial numbers of the Pareto-optimal solutions

    for trial in study.trials:
        if trial.values is not None:
            all_trials_data.append({
                "trial_number": trial.number,
                "mse": trial.values[0],
                "da": -trial.values[1],  # restore the positive sign
                "is_pareto": trial.number in pareto_numbers,# whether this is a Pareto-optimal solution
                "state": trial.state.name,
                **trial.params
            })

    if all_trials_data:
        all_df = pd.DataFrame(all_trials_data)
        all_df.to_csv("checkpoints/multi_objective_all_trials.csv", index=False)
        print(f"All trials saved: checkpoints/multi_objective_all_trials.csv")
        return all_df
    else:
        print("No valid trial results to save")
        return None


# =================== Helper: print recommendation strategies ===================
def _print_recommendations(pareto_trials):
    """Print the three recommendation strategies."""
    if not pareto_trials:# return directly if there are no Pareto-optimal solutions
        return

    print("\n" + "=" * 70)
    print("Recommendation strategies")
    print("=" * 70)

    # Strategy 1: MSE-optimal
    try:
        best_mse_trial = min(pareto_trials, key=lambda t: t.values[0])# the Pareto solution with the lowest MSE
        print("\n[Strategy 1] MSE-optimal - precise prediction")
        print(f"  Use case: quantitative backtesting, precise prediction")
        print(f"  MSE: {best_mse_trial.values[0]:.6f}")
        print(f"  DA:  {-best_mse_trial.values[1]:.4f}")
        print(f"  Parameters:")
        for k, v in best_mse_trial.params.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"\nCould not compute the MSE-optimal strategy: {e}")

    # Strategy 2: DA-optimal
    try:
        best_da_trial = max(pareto_trials, key=lambda t: -t.values[1]) # the Pareto solution with the highest DA
        print("\n[Strategy 2] DA-optimal - trading signals")
        print(f"  Use case: live trading, up/down direction calls")
        print(f"  MSE: {best_da_trial.values[0]:.6f}")
        print(f"  DA:  {-best_da_trial.values[1]:.4f}")
        print(f"  Parameters:")
        for k, v in best_da_trial.params.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"\nCould not compute the DA-optimal strategy: {e}")

    # Strategy 3: balanced strategy
    if len(pareto_trials) > 1: # if there is more than one Pareto solution
        try:
            mse_values = np.array([t.values[0] for t in pareto_trials])
            da_values = np.array([-t.values[1] for t in pareto_trials])

            mse_range = mse_values.max() - mse_values.min()
            da_range = da_values.max() - da_values.min()

            if mse_range > 1e-8 and da_range > 1e-8: # normalize only if MSE and DA vary enough
                mse_norm = (mse_values - mse_values.min()) / mse_range
                da_norm = (da_values.max() - da_values) / da_range

                # combined score (equal weighting)
                combined_score = mse_norm + da_norm # the combined score
                best_balanced_idx = np.argmin(combined_score) # pick the solution with the lowest combined score
                best_balanced_trial = pareto_trials[best_balanced_idx]

                print("\n[Strategy 3] Balanced - overall best")
                print(f"  Use case: general-purpose applications, research reports")
                print(f"  MSE: {best_balanced_trial.values[0]:.6f}")
                print(f"  DA:  {-best_balanced_trial.values[1]:.4f}")
                print(f"  Parameters:")
                for k, v in best_balanced_trial.params.items():
                    print(f"    {k}: {v}")
            else:
                print("\nThe Pareto solutions differ too little to compute a balanced strategy")
        except Exception as e:
            print(f"\nFailed to compute the balanced strategy: {e}")
    elif len(pareto_trials) == 1:
        print("\nOnly 1 Pareto solution; all strategies are the same")


# =================== Helper: print the visualization guide ===================
def _print_visualization_guide():
    """Print a usage guide for the visualization."""
    print("\n" + "=" * 70)
    print("Visualization guide")
    print("=" * 70)
    print("""
Method 1: pass --visualize when running
    python hypersearch_multi_obj.py --trials 20 --visualize

Method 2: call it from Python
    from hypersearch_multi_obj import plot_pareto_front
    plot_pareto_front()

Method 3: plot it manually
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('checkpoints/multi_objective_all_trials.csv')
    pareto = df[df['is_pareto'] == True]

    plt.scatter(df['mse'], df['da'], alpha=0.3, label='All')
    plt.scatter(pareto['mse'], pareto['da'], color='red', label='Pareto')
    plt.xlabel('MSE')
    plt.ylabel('DA')
    plt.legend()
    plt.show()
    """)


# =================== Visualization function ===================
def plot_pareto_front(csv_path="checkpoints/multi_objective_all_trials.csv",
                      save_path="checkpoints/pareto_front.png"):
    """
    Plot the Pareto front.

    Args:
        csv_path: path to the trial-results CSV
        save_path: path to save the figure
    """
    # Check for matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # headless environment
    except ImportError:
        print("matplotlib is not installed; cannot generate the visualization")
        print("Install with: pip install matplotlib")
        return

    # Check the file exists
    if not os.path.exists(csv_path):
        print(f" File not found: {csv_path}")
        return

    # Load the data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f" Failed to read the CSV: {e}")
        return

    # Check the columns
    required_cols = ['mse', 'da', 'is_pareto']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f" CSV is missing required columns: {missing_cols}")
        return

    # Split into Pareto and non-Pareto solutions
    pareto_df = df[df['is_pareto'] == True]
    non_pareto_df = df[df['is_pareto'] == False]

    if len(pareto_df) == 0:
        print("No Pareto solutions to plot")
        return

    # Build the figure
    plt.figure(figsize=(12, 7))

    # Plot the non-Pareto solutions
    if len(non_pareto_df) > 0:
        plt.scatter(non_pareto_df['mse'], non_pareto_df['da'],
                    alpha=0.4, s=80, label='Non-Pareto Solutions',
                    color='skyblue', edgecolors='steelblue', linewidths=0.5)

    # Plot the Pareto solutions
    plt.scatter(pareto_df['mse'], pareto_df['da'],
                color='red', label='Pareto Front', s=200, marker='*',
                edgecolors='darkred', linewidths=1.5, zorder=5)

    # Connect the Pareto front with a line
    if len(pareto_df) > 1:
        pareto_sorted = pareto_df.sort_values('mse')
        plt.plot(pareto_sorted['mse'], pareto_sorted['da'],
                 'r--', alpha=0.5, linewidth=1.5, zorder=4,
                 label='Pareto Boundary')

    # Annotate the Pareto solutions
    for idx, row in pareto_df.iterrows():
        label = f"#{row.get('trial_number', idx)}"
        plt.annotate(label,
                     xy=(row['mse'], row['da']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, color='darkred', fontweight='bold')

    # Figure styling
    plt.xlabel('MSE (Mean Squared Error)', fontsize=12, fontweight='bold')
    plt.ylabel('DA (Directional Accuracy)', fontsize=12, fontweight='bold')
    plt.title('Pareto Front: MSE vs DA Trade-off',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add summary stats
    stats_text = (
        f"Total Trials: {len(df)}\n"
        f"Pareto Solutions: {len(pareto_df)}\n"
        f"MSE Range: [{pareto_df['mse'].min():.6f}, {pareto_df['mse'].max():.6f}]\n"
        f"DA Range: [{pareto_df['da'].min():.4f}, {pareto_df['da'].max():.4f}]"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n Pareto-front figure saved: {save_path}")
    except Exception as e:
        print(f" Failed to save the figure: {e}")
    finally:
        plt.close()


# =================== Main search function ===================
def multi_objective_search(n_trials=20):
    """
    The main multi-objective hyperparameter search function.

    Args:
        n_trials: number of search trials (>=20 recommended)

    Returns:
        tuple: (study, pareto_df)
            - study: the Optuna study object
            - pareto_df: a DataFrame of the Pareto solutions
    """
    print("=" * 70)
    print(" Multi-objective hyperparameter search (MSE + DA)")
    print("=" * 70)
    print(f"Number of trials: {n_trials}")
    print(f"Data path: {BASE_CFG['data_path']}")
    print(f"Number of stocks: {BASE_CFG['max_assets']}")
    print(f"Number of days: {BASE_CFG['max_days']}")
    print(f"\nSearch parameters:")
    print(f"  1. diffusion.mode: ['faithful', 'adversarial', 'mixed']")
    print(f"  2. tcvae.pooling: ['mean', 'max', 'last', 'attention']")
    print(f"  3. seq_len: [24, 48, 60, 96]")
    print(f"\nOptimization objectives:")
    print(f"  Objective 0: MSE (minimize)")
    print(f"  Objective 1: DA (maximize)")
    print("=" * 70 + "\n")

    # Create the multi-objective study
    study = optuna.create_study(
        study_name="multi_objective_search",
        directions=["minimize", "minimize"],  # both minimized (DA has already been negated)
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    # Run the optimization
    try:
        study.optimize(objective_multi, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nSearch interrupted by the user (Ctrl+C)")
    except Exception as e:
        print(f"\n Error during the search: {e}")
        import traceback
        traceback.print_exc()

    # =================== Result analysis ===================
    print("\n" + "=" * 70)
    print("Search complete!")
    print("=" * 70)

    # Check whether there are any valid trials
    valid_trials = [t for t in study.trials if t.values is not None]
    if not valid_trials:
        print("\n No trials completed successfully!")
        return study, None

    print(f"\nValid trials: {len(valid_trials)} / {len(study.trials)}")

    # Get the Pareto front
    try:
        pareto_trials = study.best_trials
    except Exception as e:
        print(f"\nCould not get the Pareto front: {e}")
        pareto_trials = []

    if not pareto_trials:
        print("\n No Pareto-optimal solutions found")
        all_df = _save_all_trials(study, [])
        return study, all_df

    print(f"\n Number of Pareto-optimal solutions: {len(pareto_trials)}")
    print("\n" + "=" * 70)
    print("Pareto-front solutions")
    print("=" * 70)

    # Save the Pareto solution details
    pareto_data = []
    for i, trial in enumerate(pareto_trials, 1):
        mse = trial.values[0]
        da = -trial.values[1]  # restore the positive sign
        params = trial.params

        print(f"\nSolution {i} (Trial #{trial.number}):")
        print(f"  MSE: {mse:.6f}")
        print(f"  DA:  {da:.4f}")
        print(f"  Parameters:")
        for k, v in params.items():
            print(f"    {k}: {v}")

        pareto_data.append({
            "solution_id": i,
            "trial_number": trial.number,
            "mse": mse,
            "da": da,
            **params
        })

    # Save the Pareto solutions
    pareto_df = pd.DataFrame(pareto_data)
    pareto_df.to_csv("checkpoints/pareto_solutions.csv", index=False)
    print(f"\n Pareto solutions saved: checkpoints/pareto_solutions.csv")

    # Save all trials
    all_df = _save_all_trials(study, pareto_trials)

    # Print the recommendation strategies
    _print_recommendations(pareto_trials)

    # Print the visualization guide
    _print_visualization_guide()

    return study, pareto_df


# =================== Main entry point ===================
def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-objective hyperparameter search (MSE + DA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  Basic search (20 trials):
    python hypersearch_multi_obj.py --trials 20

  Quick test (5 trials):
    python hypersearch_multi_obj.py --trials 5

  Full search + visualization (30 trials):
    python hypersearch_multi_obj.py --trials 30 --visualize

  Custom data and stock count:
    python hypersearch_multi_obj.py --trials 20 --max_assets 20
        """
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="number of search trials (>=20 recommended, default 20)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="whether to generate the Pareto-front visualization"
    )

    parser.add_argument(
        "--max_assets",
        type=int,
        default=None,
        help="max number of stocks (defaults to the config-file setting)"
    )

    parser.add_argument(
        "--max_days",
        type=int,
        default=None,
        help="max number of days of data (defaults to the config-file setting)"
    )

    args = parser.parse_args()

    # Update the config
    if args.max_assets is not None:
        BASE_CFG["max_assets"] = args.max_assets
        print(f" Number of stocks set to: {args.max_assets}")

    if args.max_days is not None:
        BASE_CFG["max_days"] = args.max_days
        print(f" Number of days set to: {args.max_days}")

    # Run the search
    print(f"\nStarting the search...\n")
    study, pareto_df = multi_objective_search(args.trials)

    # Visualize
    if args.visualize:
        print("\nGenerating the visualization...")
        plot_pareto_front()

    # Completion message
    print("\n" + "=" * 70)
    print(" All tasks complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. checkpoints/pareto_solutions.csv - the Pareto-optimal solutions")
    print("  2. checkpoints/multi_objective_all_trials.csv - all trial results")
    if args.visualize:
        print("  3. checkpoints/pareto_front.png - the Pareto-front figure")
    print("\nNext steps:")
    print("  1. Review the recommendation strategies and pick a parameter configuration")
    print("  2. Use the selected parameters for a full training run in the main system")
    print("  3. For visualization, run: plot_pareto_front()")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
