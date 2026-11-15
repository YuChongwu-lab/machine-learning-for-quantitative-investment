# 6.5 Stock-price forecasting with attention-enhanced GRUs

Building on Section 5.6's GRU, this section introduces two attention
mechanisms and compares their effect on forecast accuracy: **TPA**
(Temporal Pattern Attention) and **EleAttG** (Element-wise Attention
Gate). EleAttG comes in two placements: front (applied to the input) and
back (applied to the GRU output, with a residual added).

## Running

```bash
python run_experiment.py --list                 # list all experiments
python run_experiment.py --experiment TPA_GRU
python TPA_GRU_example.py                       # equivalent thin entry point
```

Artifacts are written to `outputs/<experiment>/`: the prediction-comparison
plot and the loss curve.

## Experiment list

| Experiment | Model | seq_len | patience |
|---|---|---|---|
| `TPA_GRU` | Temporal Pattern Attention + GRU | 120 | **20** |
| `EleAttG_GRU` | Element-wise Attention Gate (front) | **60** | 10 |
| `EleAttG_GRU_Back` | Element-wise Attention Gate (back + residual) | 120 | 10 |

## Model/seq_len grid

`run_param_grid.py` constructs this grid, reusing the exact same
model/pipeline code as the named experiments, parameterized over
`seq_len` and `patience`:

```bash
python run_param_grid.py   # outputs/seq_len_grid/seq_len_grid_results.csv
```

## File layout

```
config.py          # SHARED + EXPERIMENTS (4 entries) + SEQ_LEN_GRID / GRID_MODELS / EXTRA_CELLS
run_experiment.py  # the single implementation: model registry + train/eval pipeline + CLI
run_param_grid.py  # reconstructs the book's 11-experiment model/seq_len analysis
data_provider.py   # loads price data, builds sliding windows, splits chronologically, standardizes
train_model.py     # training loop (early stopping)
evaluate_model.py  # MSE / MAE / RMSE / MAPE
model/             # TPA_GRU / TPA_GRU_NoRes / EleAttG_GRU / EleAttG_GRU_Back
*_example.py       # 3 thin entry points
```

