#Section 5.5 experiment configuration.


#: Hyperparameters shared by every experiment
SHARED = {
    "num_classes": 2,     # number of label classes: the response is factorized by quantile
    "lr": 1e-3,           # learning rate
    "weight_decay": 0.0,  # weight decay (nonzero only for the tuned variant)
    "test_size": 0.2,     # test-set ratio
    "val_size": 0.2,      # validation-set ratio (split again within trainval)
    "patience": 10,       # early-stopping patience
    "epochs": 500,        # number of training epochs
    "random_state": 42,   # random_state for train_test_split
    "sort_features": True,  # whether to reorder features by correlation with the label
    "use_abs": False,       # whether reordering uses the absolute correlation
    "model_kwargs": None,   # when not None, the model is constructed with these keyword args
}

#: Per-experiment settings. Keys double as both the --experiment CLI value
#: and the corresponding thin entry point's filename.
EXPERIMENTS = {
    # ---- SimpleCNN: original order / correlation-reordered / absolute-correlation-reordered ----
    "SimpleCNN_Raw_Data": {
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "sort_features": False,
    },
    "SimpleCNN_Corr_Data": {
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "sort_features": True, "use_abs": False,
    },
    "SimpleCNN_Corr_abs_data": {
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "sort_features": True, "use_abs": True,
    },
    # ---- Deformable convolution ----
    "DCN_Raw_Data": {
        "module": "model.DCN", "cls": "DeformableCNN",
        "sort_features": False,
    },
    "DCN_Corr_Data": {
        "module": "model.DCN", "cls": "DeformableCNN",
        "sort_features": True, "use_abs": False,
    },
    # Deliberately-tuned variant: lower learning rate, larger validation set, added weight decay
    "DCN_Corr_Data_tuned": {
        "module": "model.DCN", "cls": "DeformableCNN",
        "sort_features": True, "use_abs": False,
        "lr": 1e-4, "val_size": 0.3, "weight_decay": 1e-4,
    },
    # ---- Deeper backbones ----
    "Resnet_Corr_Data": {
        "module": "model.Resnet", "cls": "ResNet18",
        "sort_features": True, "use_abs": False,
    },
    "ConvNext_Corr_Data": {
        "module": "model.ConvNext", "cls": "ConvNeXt",
        "sort_features": True, "use_abs": False,
        # ConvNeXt's constructor signature differs from the other models
        "model_kwargs": {"in_channels": 1, "dim": 64, "num_classes": 2, "num_blocks": 4},
    },
}


def get_config(name: str) -> dict:
    """Return the full config for an experiment (shared settings + its own)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}. Choices: {', '.join(EXPERIMENTS)}")
    cfg = dict(SHARED)
    cfg.update(EXPERIMENTS[name])
    return cfg
