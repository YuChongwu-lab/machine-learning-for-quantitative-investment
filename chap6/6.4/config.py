"""Section 6.4 experiment configuration.
"""

#: Hyperparameters shared by every experiment. 
SHARED = {
    "num_classes": 2,      # number of label classes: the response is factorized by quantile
    "lr": 1e-4,            # learning rate
    "weight_decay": 1e-4,  # weight decay
    "test_size": 0.2,      # test-set ratio
    "val_size": 0.3,       # validation-set ratio (split again within trainval)
    "patience": 10,        # early-stopping patience
    "epochs": 500,         # number of training epochs
    "use_abs": False,      # whether feature reordering by correlation uses the absolute value
    "random_state": 42,    # random_state for train_test_split
}

#: Per-experiment settings. 
EXPERIMENTS = {
    "SE_CNN":         {"module": "model.SE_CNN",         "cls": "SE_CNN"},
    "AG_CNN":         {"module": "model.AG_CNN",         "cls": "AG_CNN"},
    "BAM_CNN":        {"module": "model.BAM_CNN",        "cls": "BAM_CNN"},
    "CBAM_CNN":       {"module": "model.CBAM_CNN",       "cls": "CBAM_CNN"},
    "FCA_CNN":        {"module": "model.FCA_CNN",        "cls": "FCA_CNN"},
    "SelfAttn_CNN":   {"module": "model.SelfAttn_CNN",   "cls": "SelfAttn_CNN"},
    "DeformAttn_CNN": {"module": "model.DeformAttn_CNN", "cls": "DeformAttn_CNN"},
    "BAM_DCN":        {"module": "model.BAM_DCN",        "cls": "BAM_DCN"},
    "DeformAttn_DCN": {"module": "model.DeformAttn_DCN", "cls": "DeformAttn_DCN"},
}


def get_config(name: str) -> dict:
    """Return the full config for an experiment (shared settings + its own)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}. Choices: {', '.join(EXPERIMENTS)}")
    cfg = dict(SHARED)
    cfg.update(EXPERIMENTS[name])
    return cfg
