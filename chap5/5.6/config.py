"""Section 5.6 experiment configuration.
"""

#: Hyperparameters shared by all experiments
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
    "ctor": "rnn",         # constructor signature: rnn or tcn
}

#: Per-experiment settings. ``title`` and ``prefix`` only affect plot
#: titles and output filenames, not the numeric results.
EXPERIMENTS = {
    "lstm": {
        "module": "model.lstm", "cls": "LSTM",
        "title": "LSTM", "prefix": "LSTM",
    },
    "gru": {
        "module": "model.gru", "cls": "GRU",
        "title": "GRU", "prefix": "GRU",
    },
    "tcn": {
        "module": "model.tcn", "cls": "TCNPredictor",
        "title": "tcn", "prefix": "tcn",
        "ctor": "tcn",
        "num_channels": [64, 64, 64],   # channel count per TCN layer
        "kernel_size": 3,               # TCN convolution kernel size
    },
}


SEQ_LEN_GRID = [20, 60, 90, 120]


def get_config(name: str) -> dict:
    """Return the full config for an experiment (shared settings + its own)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}. Choices: {', '.join(EXPERIMENTS)}")
    cfg = dict(SHARED)
    cfg.update(EXPERIMENTS[name])
    return cfg
