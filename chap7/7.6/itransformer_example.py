"""Section 7.6 experiment: iTransformer.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment itransformer``
"""

from run_experiment import run

if __name__ == "__main__":
    run("itransformer")
