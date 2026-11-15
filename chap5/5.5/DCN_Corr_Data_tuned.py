"""Section 5.5 experiment: deformable convolution (lower learning rate, larger validation set, added weight decay).

This is a thin entry point; the implementation lives in run_experiment.py
and the hyperparameters live in config.py.
Equivalent command: ``python run_experiment.py --experiment DCN_Corr_Data_tuned``
"""

from run_experiment import run

if __name__ == "__main__":
    run("DCN_Corr_Data_tuned")
