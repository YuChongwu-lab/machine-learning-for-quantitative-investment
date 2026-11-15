"""Section 5.5 experiment: ResNet18 + features reordered by correlation.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment Resnet_Corr_Data``
"""

from run_experiment import run

if __name__ == "__main__":
    run("Resnet_Corr_Data")
