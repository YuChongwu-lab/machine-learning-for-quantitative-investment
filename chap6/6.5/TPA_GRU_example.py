"""Section 6.5 experiment: TPA-GRU (Temporal Pattern Attention).

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment TPA_GRU``
"""

from run_experiment import run

if __name__ == "__main__":
    run("TPA_GRU")
