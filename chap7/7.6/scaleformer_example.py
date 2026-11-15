"""Section 7.6 experiment: Scaleformer.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment scaleformer``
"""

from run_experiment import run

if __name__ == "__main__":
    run("scaleformer")
