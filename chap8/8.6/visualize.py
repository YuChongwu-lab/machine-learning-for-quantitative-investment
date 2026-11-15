"""Plotting of the backtest results."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot_enhanced_results(results: Dict, search_results: Dict = None,
                          portfolio_values: List = None,
                          benchmark_levels: List = None,
                          save_path: str = None):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.suptitle('E_DyGFormer Quantitative Investment System -- Backtest Results', fontsize=14, fontweight='bold')

        # --- transaction costs ---
        ax_cost = axes[0]
        cost_data = {
            'Total Cost': float(results.get('total_transaction_costs', 0.0)),
            'Avg Cost': float(results.get('avg_transaction_cost', 0.0)),
            'Cost Ratio (%)': float(results.get('transaction_cost_ratio', 0.0)) * 100.0
        }
        bars = ax_cost.bar(cost_data.keys(), cost_data.values())
        for b, v in zip(bars, cost_data.values()):
            ax_cost.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center', va='bottom')
        ax_cost.set_title('Transaction Costs'); ax_cost.grid(True, alpha=0.3)

        # --- NAV curve (unified first/last-value convention & matched length = T+1) ---
        ax_nav = axes[1]
        if not portfolio_values or not benchmark_levels:
            ax_nav.text(0.5, 0.5, 'No NAV data', ha='center', va='center', transform=ax_nav.transAxes)
        else:
            # strategy: length T+1
            vals = np.asarray(portfolio_values, dtype=float)
            strat_nav = vals / max(vals[0], 1e-8)

            # benchmark: prepend a starting point so its length also becomes T+1
            bl = np.asarray(benchmark_levels, dtype=float)                   # length T
            bl_aligned = np.concatenate([[bl[0]], bl])                       # -> length T+1
            bench_nav = bl_aligned / max(bl_aligned[0], 1e-8)

            if len(strat_nav) != len(bench_nav):
                L = min(len(strat_nav), len(bench_nav))
                strat_nav = strat_nav[:L]; bench_nav = bench_nav[:L]

            ax_nav.plot(strat_nav, linewidth=2, label='Strategy NAV')
            ax_nav.plot(bench_nav, linewidth=1.8, label='Benchmark NAV')
            ax_nav.set_title('NAV Curve'); ax_nav.legend(); ax_nav.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Backtest-results plot saved to: {save_path}")
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")
