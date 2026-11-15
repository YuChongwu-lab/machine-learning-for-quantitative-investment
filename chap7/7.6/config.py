"""Section 7.6 experiment configuration.

"""

#: Hyperparameters shared by all 12 experiments 
SHARED = {
    'batch_size': 32,
    'output_size': 1,
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.1,
    'lr': 0.0001,
    'epochs': 100,
    'patience': 20,
}

#: Per-experiment settings. Keys are experiment names, which also double
#: as the valid values for --experiment on the command line.
EXPERIMENTS = {
    "transformer": {
        "module": "model.transformer",
        "cls": "Transformer",
        "data": "688256.SH.csv",
        "seq_len": 90,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
    },
    "informer": {
        "module": "model.informer",
        "cls": "Informer",
        "data": "000538.SZ.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "autoformer": {
        "module": "model.autoformer",
        "cls": "Autoformer",
        "data": "000538.SZ.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "fedformer": {
        "module": "model.fedformer",
        "cls": "FEDformer",
        "data": "688256.SH.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "crossformer": {
        "module": "model.crossformer",
        "cls": "Crossformer",
        "data": "688256.SH.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "itransformer": {
        "module": "model.itransformer",
        "cls": "iTransformer",
        "data": "000538.SZ.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "ns_transformer": {
        "module": "model.ns_transformer",
        "cls": "NS_Transformer",
        "data": "688256.SH.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "scaleformer": {
        "module": "model.scaleformer",
        "cls": "Scaleformer",
        "data": "000538.SZ.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "transformerRevIN": {
        "module": "model.transformerRevIN",
        "cls": "TransformerRevIN",
        "data": "000538.SZ.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "informerRevIN": {
        "module": "model.informerRevIN",
        "cls": "InformerRevIN",
        "data": "688256.SH.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "crossformerRevIN": {
        "module": "model.crossformerRevIN",
        "cls": "CrossformerRevIN",
        "data": "000538.SZ.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
    "itransformerRevIN": {
        "module": "model.itransformerRevIN",
        "cls": "iTransformerRevIN",
        "data": "688256.SH.csv",
        "seq_len": 60,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
    },
}


#: Table 7-3: base Transformer model, sweeping seq_len x split ratio x stock
#: (3 x 3 x 3 = 27 runs). Ratios are (train_ratio, val_ratio) pairs; test is
#: the remainder.
PART2_GRID = {
    "model": "transformer",
    "stocks": ["688256.SH", "600111.SH", "000538.SZ"],
    "seq_lens": [30, 60, 90],
    "splits": [(0.6, 0.2), (0.7, 0.2), (0.8, 0.1)],
}

#: Table 7-4: all 12 models x 3 stocks (12 x 3 = 36 runs).
PART3_GRID = {
    "models": list(EXPERIMENTS.keys()),
    "stocks": ["688256.SH", "600111.SH", "000538.SZ"],
    "seq_len": 60,
    "train_ratio": 0.7,
    "val_ratio": 0.2,
}


def get_config(name: str) -> dict:
    """Return the full config for an experiment (shared settings + its own)."""
    if name not in EXPERIMENTS:
        raise KeyError(
            f"unknown experiment {name!r}. Choices: {', '.join(EXPERIMENTS)}"
        )
    cfg = dict(SHARED)
    cfg.update(EXPERIMENTS[name])
    return cfg
