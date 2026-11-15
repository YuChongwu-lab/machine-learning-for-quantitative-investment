"""Section 7.5 experiment configuration.

This section has **two parallel data pipelines**, kept separate:

1. **Image pipeline**: encode tabular features into images (Raw /
   DeepInsight / MTF / IGTD), then classify with a vision model
   (SimpleCNN / VIT) -- 2x4=8 experiments.
   Goes through ``make_images`` -> ``create_dataloader``.
2. **Table pipeline**: feed features directly to a table-oriented
   Transformer (SAINT / SimpleSAINT) -- 3 experiments on the full
   dataset (SAINT, SimpleSAINT, SAINT_cutmix), plus 2 experiments on
   data sampled by time group (see ``train_by_time_group.py``), for 5
   in total.
"""

#: Hyperparameters shared by all experiments
SHARED = {
    "num_classes": 2,      # number of label classes
    "lr": 1e-3,            # learning rate
    "test_size": 0.2,      # test-set ratio
    "patience": 10,        # early-stopping patience
    "epochs": 100,         # number of training epochs
    "batch_size": 256,     # batch size
    "random_state": 42,    # random state for the data split
    "pipeline": "image",   # image or raw
    "train_kwargs": {},    # extra kwargs passed to train_model (cutmix, etc.)
}

#: Per-experiment settings
EXPERIMENTS = {
    # ---- Image pipeline: SimpleCNN x 4 encoding methods ----
    "SimpleCNN": {
        "pipeline": "image",
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "method": "deepinsight",       # tabular-to-image encoding method
        "image_shape": None,
        "ctor": "num_classes",
    },
    "SimpleCNN_raw": {
        "pipeline": "image",
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "method": "raw",
        "image_shape": None,
        "ctor": "num_classes",
    },
    "SimpleCNN_mtf": {
        "pipeline": "image",
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "method": "mtf",
        "image_shape": None,
        "ctor": "num_classes",
    },
    "SimpleCNN_igtd": {
        "pipeline": "image",
        "module": "model.SimpleCNN", "cls": "SimpleCNN",
        "method": "igtd",
        "image_shape": None,
        "ctor": "num_classes",
    },
    # ---- Image pipeline: VIT x 4 encoding methods ----
    "VIT": {
        "pipeline": "image",
        "module": "model.VIT", "cls": "ViTClassification",
        "method": "mtf",               # Markov Transition Field encoding
        "image_shape": None,
        "ctor": "vit",
        "img_pool_size": (8, 8), "patch_size": 2, "embed_dim": 128,
        "num_layers": 6, "n_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
    },
    "VIT_raw": {
        "pipeline": "image",
        "module": "model.VIT", "cls": "ViTClassification",
        "method": "raw",
        "image_shape": None,
        "ctor": "vit",
        "img_pool_size": (8, 8), "patch_size": 2, "embed_dim": 128,
        "num_layers": 6, "n_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
    },
    "VIT_deepinsight": {
        "pipeline": "image",
        "module": "model.VIT", "cls": "ViTClassification",
        "method": "deepinsight",
        "image_shape": None,
        "ctor": "vit",
        "img_pool_size": (8, 8), "patch_size": 2, "embed_dim": 128,
        "num_layers": 6, "n_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
    },
    "VIT_igtd": {
        "pipeline": "image",
        "module": "model.VIT", "cls": "ViTClassification",
        "method": "igtd",
        "image_shape": None,
        "ctor": "vit",
        "img_pool_size": (8, 8), "patch_size": 2, "embed_dim": 128,
        "num_layers": 6, "n_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
    },
    # ---- Table pipeline ----
    "SAINT": {
        "pipeline": "raw",
        "module": "model.SAINT", "cls": "SAINT",
        "ctor": "saint",
        "val_size": 0.16,
        "num_features": 93, "d_model": 128, "n_heads": 8,
        "dropout": 0.1,
    },
    "SimpleSAINT": {
        "pipeline": "raw",
        "module": "model.SimpleSAINT", "cls": "SimpleSAINT",
        "ctor": "simple_saint",
        "val_size": 0.16,
        "num_features": 93, "d_model": 128, "n_heads": 8,
        "num_layers": 6, "dropout": 0.1,
    },
    "SAINT_cutmix": {
        "pipeline": "raw",
        "module": "model.SAINT", "cls": "SAINT",
        "ctor": "saint",
        "val_size": 0.16,
        "num_features": 93, "d_model": 128, "n_heads": 8,
        "num_layers": 6, "dropout": 0.1,
        # CutMix augmentation: mixes two samples column-wise
        "train_kwargs": {"use_cutmix": True, "cutmix_prob": 0.7,
                         "cutmix_alpha": 1.0, "cutmix_col_frac": (0.3, 0.7)},
    },
}


def get_config(name: str) -> dict:
    """Return the full config for an experiment (shared settings + its own)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}. Choices: {', '.join(EXPERIMENTS)}")
    cfg = dict(SHARED)
    cfg.update(EXPERIMENTS[name])
    return cfg
