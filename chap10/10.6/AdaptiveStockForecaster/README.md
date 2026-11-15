# Section 10.6 experiments: deep generative models for an adaptive online stock-forecasting system

A stock-forecasting system built on the **Proceed** framework: the core idea is to detect market distribution drift and dynamically modulate the backbone network, letting the model keep adapting to regime changes without retraining.

This project is an improved version of the PROCEED framework from Zhao and Shen (KDD 2025). The
main improvements are: (1) a TCVAE concept encoder; (2) optional DDPM-based data augmentation;
(3) an optional validation split.


## Core design

```
Market data → TCVAE concept encoding → drift detection (delta) → ProceedAdapter gated modulation → PatchTST forecast → online fine-tuning (with validation rollback)
```

- **PatchTST backbone**: patches the time series and extracts features via a Transformer
- **TCVAE concept encoder**: compresses a time series of arbitrary length into a fixed-dimension
  "market-state vector", supporting mean / max / last / attention pooling
- **ProceedAdapter**: takes the concept difference `delta` between source and target domains and
  generates multiplicative gating parameters covering every backbone layer, giving lightweight
  distribution adaptation
- **Diffusion-based augmentation**: optional DDPM data augmentation, supporting faithful /
  adversarial / mixed modes
- **Online fine-tuning**: one fine-tuning step on the prediction head before each inference    step, with validation-based rollback to prevent negative transfer


## File layout

```
.
├── config.py                                      # shared hyperparameter config for all modules
├── data.py                                        # data loading and sliding-window construction
├── models.py                                      # model definitions (TransformerBlock / PatchTSTBackbone / ProceedAdapter)
├── losses.py                                      # DirectionalLoss (direction-aware weighted loss)
├── TCVAE.py                                       # temporal conditional VAE and concept encoder
├── train_utils.py                                 # training functions (pretraining / adapter training / online fine-tuning)
├── validation_framework.py                        # generic Trainer, EarlyStopping, split_train_val
├── proceed_diffusion_augmentation.py              # diffusion model definitions and DiffusionAugmentor
├── proceed_diffusion_augmentation_with_validation.py  # diffusion-augmented trainer with a validation split
├── run_experiment.py                              # main experiment entry point (run_stock + __main__)
├── run_baseline_grid.py                           # baseline grid: 4 augmentation conditions x 4 pooling strategies, reusing run_stock
├── hypersearch_multi_obj.py                       # multi-objective hyperparameter search (Optuna, MSE + DA Pareto front)
├── quick_search.py                                # quick hyperparameter search (few trials, for pipeline sanity checks)
├── drift_analysis.py                              # concept-representation and distribution-drift visualization
├── requirements.txt
├── .gitignore
└── checkpoints/                                   # training checkpoints (created automatically, contents not committed)
```

---

## Quick start

```bash
python run_experiment.py                              # default config
python run_experiment.py --tcvae-pooling mean         # switch pooling strategy
python run_experiment.py --diffusion-enable --diffusion-mode faithful
python run_experiment.py --help                       # see all flags
```


## Baseline grid (augmentation x pooling)

The book reports a 16-cell grid over 4 augmentation conditions (`none`, `faithful`, `adversarial`,`mixed`) x 4 TCVAE pooling strategies (`mean`, `max`, `last`, `attention`), run across the first 10 stocks. 

```bash
python run_baseline_grid.py --max-assets 10           # full grid, first 10 stocks
python run_baseline_grid.py --max-assets 10 --quick   # fewer epochs, for a quick sanity check
```

It writes a per-stock detail table and an aggregated summary table to
`checkpoints/baseline_grid/baseline_grid_per_stock.csv` and `baseline_grid_summary.csv`.

## Hyperparameter search

```bash
# full multi-objective search (MSE + DA Pareto front)
python hypersearch_multi_obj.py --trials 30

# search + generate visualization plots
python hypersearch_multi_obj.py --trials 30 --visualize

# quick sanity-check pipeline (few trials)
python quick_search.py --trials 5
```

## Drift analysis

```bash
python drift_analysis.py
```

---

## Key hyperparameters

All parameters live in `config.py`; the key ones are listed below:

| Parameter | Default | Description |
|---|---|---|
| `seq_len` | 48 | input sequence length (must be a multiple of 4) |
| `pred_len` | 1 | forecast horizon |
| `tcvae.latent_dim` | 64 | concept-vector dimension |
| `tcvae.pooling` | `"mean"` | pooling strategy |
| `patchtst.d_model` | 256 | Transformer hidden dimension |
| `patchtst.nlayers` | 4 | number of Transformer layers |
| `adapter_train.concept_dim` | 64 | must equal `tcvae.latent_dim` |
| `adapter_train.scale_mult` | 0.05 | gating-magnitude cap (tanh x scale_mult) |
| `diffusion.mode` | `"faithful"` | diffusion augmentation mode |
| `online_ft.lr` | 1e-4 | online fine-tuning learning rate |

---

## Evaluation metrics

| Metric | Description |
|---|---|
| MAE | mean absolute error |
| RMSE | root mean squared error |
| sMAPE | symmetric mean absolute percentage error |
| DA | direction accuracy, the core metric |
| R² | coefficient of determination |

---

