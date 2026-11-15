"""Section 7.5 experiment: ViT + Markov Transition Field image encoding.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment VIT``
"""

from run_experiment import run

if __name__ == "__main__":
    run("VIT")
