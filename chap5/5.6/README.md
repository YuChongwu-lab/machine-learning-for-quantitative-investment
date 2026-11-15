# 5.6 Stock-price forecasting with recurrent networks and temporal convolution

Takes 5 fields from Ping An Bank's (000001.SZ) daily price data
(`open`/`high`/`low`/`vol`/`close`) as input, looks back 120 days, and
predicts the next day's closing price, comparing three architectures:
LSTM, GRU, and TCN.

## Running

```bash
python run_experiment.py --list              # list all experiments
python run_experiment.py --experiment lstm
python lstm_example.py                       # equivalent thin entry point
```

Artifacts are written to `outputs/<experiment>/`: the prediction-comparison
plot and the loss curve.

## Experiment list

| Experiment | Model | Constructor signature |
|---|---|---|
| `lstm` | LSTM | `(input_size, hidden_size, num_layers, output_size, dropout)` |
| `gru` | GRU | same as above |
| `tcn` | TCNPredictor | `(input_size, output_size, seq_len, num_channels, kernel_size, dropout)` |

Shared settings: `seq_len=120`, `pred_horizon=1`, `input_size=5`,
`hidden_size=64`, `num_layers=3`, `dropout=0.2`, `train_ratio=0.7`,
`val_ratio=0.15`, `batch_size=32`, `lr=0.001`, `epochs=100`, `patience=10`.
TCN additionally has `num_channels=[64, 64, 64]`, `kernel_size=3`.

## The seq_len x model grid

The book doesn't just compare LSTM/GRU/TCN at `seq_len=120` -- holding
every other setting fixed, it sweeps `seq_len` over 20/60/90/120 and
compares the 3 models at each value (4 x 3 = 12 experiments in total).
`run_param_grid.py` constructs the full grid, reusing the exact same
model/pipeline code as the 3 named experiments, parameterized only over
`seq_len`:

```bash
python run_param_grid.py   # outputs/seq_len_grid/seq_len_grid_results.csv
```

## File layout

```
config.py          # SHARED + EXPERIMENTS (3 entries) + SEQ_LEN_GRID
run_experiment.py   # the single implementation: model registry + train/eval pipeline + CLI
run_param_grid.py  # reconstructs the book's seq_len x model sweep
data_provider.py    # loads price data, builds sliding windows, splits chronologically, standardizes
train_model.py      # training loop (early stopping)
evaluate_model.py   # MSE / MAE / RMSE / MAPE
model/              # LSTM / GRU / TCN
*_example.py        # 3 thin entry points
```
