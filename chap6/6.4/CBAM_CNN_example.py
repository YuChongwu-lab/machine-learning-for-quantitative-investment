"""Section 6.4 experiment: CBAM (Convolutional Block Attention Module) + CNN.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment CBAM_CNN``
"""

from run_experiment import run

if __name__ == "__main__":
    run("CBAM_CNN")
