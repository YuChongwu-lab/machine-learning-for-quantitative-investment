"""Section 6.4 experiment: SE (Squeeze-and-Excitation) attention + CNN.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment SE_CNN``
"""

from run_experiment import run

if __name__ == "__main__":
    run("SE_CNN")
