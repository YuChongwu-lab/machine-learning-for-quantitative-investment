# 7.6 Stock-price time-series forecasting with Transformer-family models

This section compares 12 Transformer variants on single-step forecasting
for individual stocks' daily price data, 4 of which include RevIN
(Reversible Instance Normalization).

## Running

```bash
python run_experiment.py --list                    # list all experiments and their settings
python run_experiment.py --experiment transformer  # run a specific experiment
python transformer_example.py                      # equivalent thin entry point
```

Artifacts are written to `outputs/<experiment>/`: the best checkpoint
`best_model.pth`, the prediction-comparison plot, and the loss curve.

## Experiment list

| Experiment | Model | Data | seq_len | train/val |
|---|---|---|---|---|
| `transformer` | Transformer | 688256.SH | **90** | **0.8 / 0.1** |
| `informer` | Informer | 000538.SZ | 60 | 0.7 / 0.2 |
| `autoformer` | Autoformer | 000538.SZ | 60 | 0.7 / 0.2 |
| `fedformer` | FEDformer | 688256.SH | 60 | 0.7 / 0.2 |
| `crossformer` | Crossformer | 688256.SH | 60 | 0.7 / 0.2 |
| `itransformer` | iTransformer | 000538.SZ | 60 | 0.7 / 0.2 |
| `ns_transformer` | Non-stationary Transformer | 688256.SH | 60 | 0.7 / 0.2 |
| `scaleformer` | Scaleformer | 000538.SZ | 60 | 0.7 / 0.2 ※ |
| `transformerRevIN` | Transformer + RevIN | 000538.SZ | 60 | 0.7 / 0.2 |
| `informerRevIN` | Informer + RevIN | 688256.SH | 60 | 0.7 / 0.2 |
| `crossformerRevIN` | Crossformer + RevIN | 000538.SZ | 60 | 0.7 / 0.2 |
| `itransformerRevIN` | iTransformer + RevIN | 688256.SH | 60 | 0.7 / 0.2 |

Shared hyperparameters: `batch_size=32`,
`d_model=128`, `num_heads=4`, `num_layers=2`, `dropout=0.1`, `lr=1e-4`,
`epochs=100`, `patience=20`.

## Param grids in tables 7-3 and 7-4

The book reports two sweeps in tables 7-3 and 7-4:

* **Table 7-3**: the base Transformer model, swept over seq_len x
  train/val/test split x stock (3 x 3 x 3 = 27 combinations).
* **Table 7-4**: all 12 models compared across 3 stocks (12 x 3 = 36
  combinations), with seq_len and split ratio fixed.

Param grid constructs the sweep structure (reusing the exact
same model code, data loading, and training pipeline as the 12 named
experiments, just parameterized over the full grid) and writes a results
CSV comparable to the published tables:

```bash
python run_param_grid.py --part 2   # Table 7-3: outputs/grid_part2/table_7-3_results.csv
python run_param_grid.py --part 3   # Table 7-4: outputs/grid_part3/table_7-4_results.csv
```

## File layout

```
config.py                  # config for all 12 experiments (shared + per-experiment settings)
run_experiment.py          # the single implementation: registry + train/eval pipeline + CLI
run_param_grid.py          # reconstructs the book's Table 7-3 / Table 7-4 sweeps
data_provider.py           # TimeSeriesScaler and create_dataloader
train_and_evaluate_model.py# training loop (gradient clipping, LR schedule, early stopping)
metrics.py                 # direction_accuracy
model/                     # the 12 model definitions
*_example.py               # 12 thin entry points, keeping the book's filenames
drift_detection.py          # KL-divergence distribution-drift detection (standalone)
rolling_window_train.py    # rolling vs. fixed window training comparison (standalone)
revin_transformer.py       # standalone RevIN demo (standalone)
```

