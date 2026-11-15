"""Section 5.5 experiment: deformable convolution + original feature order.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment DCN_Raw_Data``
"""

from run_experiment import run

if __name__ == "__main__":
    run("DCN_Raw_Data")
