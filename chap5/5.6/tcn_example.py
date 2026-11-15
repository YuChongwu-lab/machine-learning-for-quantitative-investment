"""Section 5.6 experiment: TCN (Temporal Convolutional Network).

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment tcn``
"""

from run_experiment import run

if __name__ == "__main__":
    run("tcn")
