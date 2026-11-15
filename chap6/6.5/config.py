"""Section 6.5 experiment configuration.
"""

#: Hyperparameters shared by every experiment
SHARED = {
    "seq_len": 120,        # lookback window length (days)
    "pred_horizon": 1,     # forecast horizon (days)
    "input_size": 5,       # number of input features: open/high/low/vol/close
    "output_size": 1,      # number of output features
    "hidden_size": 64,     # hidden layer size
    "num_layers": 3,       # number of recurrent layers
    "dropout": 0.2,        # dropout rate
    "train_ratio": 0.7,    # training-set ratio
    "val_ratio": 0.15,     # validation-set ratio
    "batch_size": 32,      # batch size
    "lr": 0.001,           # learning rate
    "epochs": 100,         # number of training epochs
    "patience": 10,        # early-stopping patience
}

EXPERIMENTS = {
    "TPA_GRU": {
        "module": "model.TPA_GRU", "cls": "TPA_GRU",
        "title": "TPA_GRU", "prefix": "TPA_GRU",
        "patience": 20,          # this experiment uses a larger early-stopping patience
    },
    "EleAttG_GRU": {
        "module": "model.EleAttG_GRU", "cls": "EleAttG_GRU",
        "title": "EleAttG_GRU", "prefix": "EleAttG_GRU",
        "seq_len": 60,           # this experiment uses a shorter lookback window
    },
    "EleAttG_GRU_Back": {
        "module": "model.EleAttG_GRU_Back", "cls": "EleAttG_GRU_Back",
        "title": "EleAttG_GRU_Back", "prefix": "EleAttG_GRU_Back",
    },
    "TPA_GRU_NoRes": {
        "module": "model.TPA_GRU_NoRes", "cls": "TPA_GRU_NoRes",
        "title": "TPA_GRU_NoRes", "prefix": "TPA_GRU_NoRes",
        "seq_len": 60,
    },
}


SEQ_LEN_GRID = [20, 60, 90, 120]
GRID_MODELS = ["EleAttG_GRU_Back", "TPA_GRU"]  # swept across all of SEQ_LEN_GRID
EXTRA_CELLS = [
    {"model": "TPA_GRU_NoRes", "seq_len": 60, "patience": 10},
    {"model": "TPA_GRU_NoRes", "seq_len": 60, "patience": 20},
]


def get_config(name: str) -> dict:
    """Return the full config for an experiment (shared settings + its own)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}. Choices: {', '.join(EXPERIMENTS)}")
    cfg = dict(SHARED)
    cfg.update(EXPERIMENTS[name])
    return cfg
