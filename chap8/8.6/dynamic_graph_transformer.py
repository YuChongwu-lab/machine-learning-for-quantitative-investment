"""Thin entry point, kept under the book-matching filename.

See run_experiment.py for the actual implementation; this just runs the
default configuration (max_assets=180, max_days=500).
"""

from config import SHARED
from run_experiment import run
from visualize import plot_enhanced_results

if __name__ == "__main__":
    results = run(max_assets=SHARED["max_assets"], max_days=SHARED["max_days"])
    if isinstance(results, dict):
        plot_enhanced_results(
            results['backtest_results'],
            results.get('search_results'),
            portfolio_values=getattr(results.get('backtest_engine', None), 'portfolio_value', None),
            benchmark_levels=getattr(results.get('backtest_engine', None), 'benchmark_levels', None),
            save_path='enhanced_analysis_results.png'
        )
