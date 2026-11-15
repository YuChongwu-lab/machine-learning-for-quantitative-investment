# 7.5 Image encoding of tabular data and table Transformers

This section explores two routes for feeding 93-dimensional stock
cross-sectional features to deep models:

1. **Image pipeline** -- encode tabular features into images with
   DeepInsight, Markov Transition Field, etc., then classify with a
   vision model (SimpleCNN, ViT);
2. **Table pipeline** -- feed features directly to a table-oriented
   Transformer (SAINT), optionally with column-wise CutMix augmentation.

## Running

```bash
python run_experiment.py --list                # list all experiments
python run_experiment.py --experiment SAINT
python SAINT_example.py                        # equivalent thin entry point
```

Artifacts are written to `outputs/<experiment>/`: `best_model.pt`,
`loss_curve.png`, `val_metrics.png`, `confusion_matrix.png`.

## Experiment list

This section's actual design has **13 experiments**: 8 in the image
pipeline (2 models x 4 encoding methods) and 5 in the table pipeline (3
on the full dataset + 2 on data sampled by time group). 

| Experiment | Pipeline | Model | Encoding / augmentation | Split |
|---|---|---|---|---|
| `SimpleCNN` | image | SimpleCNN | DeepInsight | two `train_test_split` calls (0.2 / 0.2) |
| `SimpleCNN_raw` | image | SimpleCNN | Raw | same as above |
| `SimpleCNN_mtf` | image | SimpleCNN | Markov Transition Field | same as above |
| `SimpleCNN_igtd` | image | SimpleCNN | IGTD | same as above |
| `VIT` | image | ViTClassification | Markov Transition Field | same as above |
| `VIT_raw` | image | ViTClassification | Raw | same as above |
| `VIT_deepinsight` | image | ViTClassification | DeepInsight | same as above |
| `VIT_igtd` | image | ViTClassification | IGTD | same as above |
| `SAINT` | table | SAINT | -- | `split_raw` (test 0.2 / val 0.16) |
| `SimpleSAINT` | table | SimpleSAINT | -- | same as above |
| `SAINT_cutmix` | table | SAINT | column-wise CutMix | same as above |
| `SAINT_sampled` (not in the CLI registry; see below) | table | SAINT | -- | one time-sampled group |
| `SAINT_cutmix_sampled` (ditto) | table | SAINT | column-wise CutMix | same sampled group |

Shared settings: `num_classes=2`, `lr=1e-3`, `test_size=0.2`,
`patience=10`, `epochs=100`, `batch_size=256`, `random_state=42`.
The table pipeline additionally has `num_features=93`, `d_model=128`,
`n_heads=8`, `dropout=0.1`; ViT additionally has `img_pool_size=(8,8)`,
`patch_size=2`, `embed_dim=128`, `num_layers=6`, `mlp_ratio=4.0`;
CutMix parameters are `cutmix_prob=0.7`, `cutmix_alpha=1.0`,
`cutmix_col_frac=(0.3, 0.7)`.

## File layout

```
config.py                   # SHARED + EXPERIMENTS (13 entries across both pipelines)
run_experiment.py           # the single implementation: both pipelines + model registry + CLI
run_sampled_experiments.py  # reconstructs the two time-sampled SAINT experiments
data_utils.py                # make_images (4 encodings), split_raw, two kinds of DataLoader
train_utils.py               # training loop (including CutMix), evaluation, result visualization
get_data.py                  # loads data_ml.csv and arranges it into features + label
model/                       # SimpleCNN / ViT / SAINT / SimpleSAINT
*_example.py                  # 5 thin entry points (the originally-named experiments)
train_by_time_group.py       # rolling training grouped by time (standalone; also used by run_sampled_experiments.py)
visualize_images.py           # visualizes the effect of each image encoding (standalone)
```


