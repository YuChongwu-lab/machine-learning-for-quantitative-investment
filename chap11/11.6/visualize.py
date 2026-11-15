"""Plotting of the backtest results (transaction costs + NAV curve)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_enhanced_results(results: Dict,
                          portfolio_values: List,
                          benchmark_levels: List,
                          save_path: Optional[str] = None):
    """
    Plots the transaction-cost chart and the NAV curve.
    - results: the return value of EnhancedBacktestEngine.calculate_enhanced_metrics
    - portfolio_values: the engine's cumulative portfolio-value series (length T+1)
    - benchmark_levels: the engine's recorded benchmark NAV/price level (length T)
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.suptitle('Graph-RL Backtest Results', fontsize=14, fontweight='bold')

        # --- transaction costs ---
        ax_cost = axes[0]
        cost_data = {
            'Total Cost': float(results.get('total_transaction_costs', 0.0)),
            'Avg Cost': float(results.get('avg_transaction_cost', 0.0)),
            'Cost Ratio (%)': float(results.get('transaction_cost_ratio', 0.0)) * 100.0
        }
        bars = ax_cost.bar(list(cost_data.keys()), list(cost_data.values()))
        for b, v in zip(bars, cost_data.values()):
            ax_cost.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center', va='bottom')
        ax_cost.set_title('Transaction Costs')
        ax_cost.grid(True, alpha=0.3)

        # --- NAV curve (strategy is T+1, the benchmark T is aligned by prepending a point) ---
        ax_nav = axes[1]
        if (not portfolio_values) or (not benchmark_levels):
            ax_nav.text(0.5, 0.5, 'No NAV data', ha='center', va='center', transform=ax_nav.transAxes)
        else:
            vals = np.asarray(portfolio_values, dtype=float)  # [T+1]
            bl = np.asarray(benchmark_levels, dtype=float)  # [T]

            # normalize so both the strategy and the benchmark start at 1.0
            if len(vals) > len(bl):
                # the strategy has one extra initial point; use it to normalize
                strat_nav = vals / vals[0]  # [T+1], starts at 1.0
                # prepend a starting point of 1.0 for the benchmark
                bench_nav = np.concatenate([[1.0], bl / bl[0]])  # [T+1], starts at 1.0
            elif len(vals) == len(bl):
                # same length: normalize directly
                strat_nav = vals / vals[0]
                bench_nav = bl / bl[0]
            else:
                # the benchmark is longer (unlikely): truncate
                strat_nav = vals / vals[0]
                bench_nav = (bl[:len(vals)] / bl[0])

            # make sure the lengths match
            L = min(len(strat_nav), len(bench_nav))
            strat_nav = strat_nav[:L]
            bench_nav = bench_nav[:L]

            # plot (keeping the original style)
            ax_nav.plot(strat_nav, linewidth=2, label='Strategy NAV')
            ax_nav.plot(bench_nav, linewidth=1.8, label='Benchmark NAV')
            ax_nav.set_title('NAV Curve'); ax_nav.legend();ax_nav.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Plot] Saved image to: {save_path}")
        plt.show()
    except Exception as e:
        print(f"[Plot] Plotting failed: {e}")
