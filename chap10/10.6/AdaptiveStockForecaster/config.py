# config.py
"""
Unified configuration file.

Usage:
  from config import BASE_CFG
  import copy
  cfg = copy.deepcopy(BASE_CFG)   
"""

BASE_CFG = {
    "seq_len": 48,
    "pred_len": 1,

    "pretrain": {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 10,
    },

    "adapter_train": {
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 10,
        "bottleneck_dim": 96,
        "concept_dim": 64,
        "d_model": 128,
        "reg": 1e-4,
        "scale_mult": 0.05,
        "synth_prob": 0.35,
        "synth_modes": ["noise", "scale", "shift"],
        "synth_strength": 0.02,
    },

    "tcvae": {
        "latent_dim": 64,      # latent space dimension (concept vector dimension)
        "hidden_dim": 128,     # hidden layer dimension
        "pooling": "mean",     # pooling strategy: 'mean', 'max', 'last', 'attention'
    },

    "diffusion": {
        "enable": True,                              # set to False to disable augmentation
        "mode": "faithful",                          # 'faithful', 'adversarial', 'mixed'
        "pretrain_epochs": 5,
        "augment_prob": 0.35,
        "augment_ratio": 1.0,
        "adversarial_strength": 0.3,
        "hidden_dim": 64,
        "context_dim": 64,
        "save_dir": "checkpoints/diffusion_models",
    },

    "patchtst": {
        "d_model": 256,
        "nhead": 8,
        "nlayers": 4,
        "dropout": 0.1,
        "patch_len": 6,
        "stride": 3,
        "head_hidden": 128,
    },

    "online_ft": {
        "enable": True,
        "lr": 1e-4,
        "steps": 1,
        "batch_size": 8,
        "use_train_tail": 32,
        "finetune_layers": ["head_in", "fc_out"],
    },

    "loss": {
        "type": "directional",  # uses DirectionalLoss
        "alpha": 1.0,           # weight when the direction is correct
        "beta": 2.0,            # weight when the direction is wrong (penalized more heavily)
        "use_soft": True,       # use soft directional matching
    },

    "data_path": "CSI100_Top30.pkl",   # filename only; resolved to the top-level data/ by data.load_real_data
    "max_assets": 2,
    "max_days": 1000,
    "seed": 42,
}
