"""Section 6.4 experiment: BAM attention + deformable convolution backbone.

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment BAM_DCN``
"""

from run_experiment import run

if __name__ == "__main__":
    run("BAM_DCN")
