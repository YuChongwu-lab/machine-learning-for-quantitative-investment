"""Section 5.5 experiment: SimpleCNN + features reordered by absolute correlation.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment SimpleCNN_Corr_abs_data``
"""

from run_experiment import run

if __name__ == "__main__":
    run("SimpleCNN_Corr_abs_data")
