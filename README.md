# Machine Learning for Quantitative Investment — Companion Experiment Code

This repository is the companion experiment code for the book *Machine
Learning for Quantitative Investment*.

---

## Directory Structure

```
.
├── data/                  # The single location where all of the book's data lives (see data/README.md)
├── R/                     # R dependency installation script and data-path resolution
├── tools/                 # Master copy of the data-path-resolution helper duplicated into each section
├── chap3/  3.6  3.7       # R: linear models and regularization
├── chap4/  4.6  4.7       # R: tree models
├── chap5/  5.5  5.6       # CNN, RNN/TCN
├── chap6/  6.4  6.5       # Attention mechanisms
├── chap7/  7.5  7.6       # Transformers
├── chap8/  8.5  8.6       # Dynamic graph neural networks
├── chap10/ 10.6           # Deep generative models and online adaptation
└── chap11/ 11.6           # Reinforcement learning
```

A typical section directory looks like:

```
chapN/N.M/
├── README.md          # This section's experiment list and usage notes
├── config.py          # Per-experiment configuration 
├── paths.py           # Data-path resolution
├── run_experiment.py  # The single implementation: model registry + train/eval pipeline + CLI
├── model/             # Model definitions
├── <experiment>_example.py # A thin entry point
└── outputs/           # Run outputs, one subdirectory per experiment
```
---

## Experiment Index

| Section | Topic | Models | Data | Language |
|---|---|---|---|---|
| [3.6](chap3/3.6/) | PB relative valuation | OLS, FGLS, 2SLS, quantile regression | `data_ml.RData` | R |
| [3.7](chap3/3.7/) | Predictive regression | Ridge, Lasso, ElasticNet, quantile lasso, multi-layer group lasso | `data_ml.RData` | R |
| [4.6](chap4/4.6/) | Structural breaks in PB valuation | CART, MOB | `data_ml.RData` | R |
| [4.7](chap4/4.7/) | Tree-model comparison | CART, CTree, random forest, LightGBM | the analyst-synchronicity panel | R |
| [5.5](chap5/5.5/) | Cross-sectional data as images for classification | SimpleCNN, deformable convolution, ResNet, ConvNeXt | `data_ml_2017_11.csv` | Python |
| [5.6](chap5/5.6/) | Stock-price time-series forecasting | LSTM, GRU, TCN | `000001SZ.xlsx` | Python |
| [6.4](chap6/6.4/) | Attention-augmented CNNs | SE, AG, BAM, CBAM, FCA, SelfAttn, DeformAttn | `data_ml_2017_11.csv` | Python |
| [6.5](chap6/6.5/) | Attention-augmented GRU | TPA-GRU, EleAttG-GRU | `000001SZ.xlsx` | Python |
| [7.5](chap7/7.5/) | Tabular-to-image encoding nmm| SAINT, ViT, CutMix | `data_ml.csv` | Python |
| [7.6](chap7/7.6/) | Transformer-family time-series forecasting | Transformer, Informer, Autoformer, FEDformer, Crossformer, iTransformer, Non-stationary Transformer, Scaleformer, plus 4 RevIN variants | individual-stock daily data | Python |
| [8.5](chap8/8.5/) | Hierarchical dynamic graph neural network | 4 edge types + a hierarchical gated GNN | `CSI100_*.pkl` | Python |
| [8.6](chap8/8.6/) | Dynamic graph Transformer | E-DyGFormer + explainability analysis | `CSI100_*.pkl` | Python |
| [10.6](chap10/10.6/) | Online-adaptive forecasting | PatchTST + TCVAE + diffusion augmentation (Proceed framework) | `CSI100_Top30.pkl` | Python |
| [11.6](chap11/11.6/) | RL-based portfolio management | Dynamic-graph PPO + curriculum learning + multi-objective optimization | `CSI100_*.pkl` | Python |

---

## Environment

### Python

```bash
pip install -r requirements.txt
```

### R

```bash
Rscript R/install_packages.R
```

---

## Data

All of the book's data lives in one place, [`data/`](data/); see
[data/README.md](data/README.md) for its structure, scale, and provenance.
Every experiment directory has a self-contained
`paths.py` that locates the repo root starting from the file's own
location:

```python
from paths import data_file, stock_file

data = pd.read_csv(data_file("data_ml_2017_11.csv"), index_col="stock_id")
df   = pd.read_csv(stock_file("688256.SH.csv"), index_col="trade_date")
```

So scripts run from any working directory:

```bash
python chap7/7.6/transformer_example.py     # from the repo root
cd chap7/7.6 && python transformer_example.py   # or from the section directory
```

---

## Known Issues and Errata

The issues below **genuinely exist in the code that produced the book's
published numbers**. Since correcting them would change those
already-published results, this codebase **deliberately preserves the
original behavior** and only records them here, for readers who extend
these experiments on their own.

| Location | Issue | Impact |
|---|---|---|
| `chap4/4.6` | The pruning step takes `cptable[11, ]`, and the 8 node numbers passed to `summary(pb_mob, node = N)` are all hardcoded | Valid only for the current data and the current tree shape; using different data will raise an error |
| `chap5/5.6` vs. `chap7/7.6` | The training loss uses a different normalization convention (per-batch vs. per-sample) | The two chapters' loss curves are not on the same scale and cannot be compared directly |
| `chap6/6.4` | The book's text says `weight_decay` is `1e-2`, it should be `1e-4`, and this repository keeps the `1e-4`.
| `chap7/7.6/data_provider.py` | The test-set `DataLoader` uses `shuffle=True` | The test set's order gets shuffled. Since `TensorDataset` keeps X and y shuffled as pairs, the 5 metrics the book reports (MSE/MAE/RMSE/R², and `direction_accuracy`, which compares signs element-wise) are **all order-independent, so their values are unaffected**; but the "actual vs. predicted" curve's x-axis no longer represents chronological order and should not be read as a time series |
| `chap8/8.5` | `main()` applies a different cleaning convention to the price matrix vs. the panel data: `price_data` comes from `load_real_data`'s **filtered** data (drops stocks by missing-data ratio, clips `ret` to +/-0.5, drops stocks with fewer than 100 samples), while the feature engineering downstream uses `panel_data`, the **unfiltered** raw panel | The two paths cover different sets of stocks and different sample ranges. The code already has a comment warning not to merge this "duplicate read" away -- doing so would change the published results |
| `chap10/10.6` | Cross-process non-determinism from `cudnn.benchmark=True`**: `chap10/10.6/AdaptiveStockForecaster/data.py`'s `set_seed()` sets `torch.backends.cudnn.benchmark = True`, rather than the `deterministic = True` used uniformly elsewhere. cuDNN automatically picks a convolution algorithm based on runtime timing, and the choice depends on the GPU's state at the moment the process starts | Unlike the `atomicAdd`-type non-determinism: results are stable across repeated calls **within one Python process** (differences appear only in the 7th-8th decimal digit); but **restarting the process** can produce a more noticeable shift (a single-stock probe's `MAE` shifted as far as the 4th decimal digit). |
| `chap11/11.6` | The benchmark NAV series has its first element set to 1000, while every other element is around 1.0 | A 1000x scale discontinuity exists, which affects `information_ratio` and `benchmark_total_return`. Kept as-is: changing it would alter already-published numbers |
| **Experiments that are inherently non-deterministic on GPU**: 5.5's `DCN_*`, 6.4's `BAM_DCN`/`DeformAttn_DCN` (`torchvision.ops.DeformConv2d`), and 7.5's `SAINT`/`SAINT_cutmix` (a block-wise `nn.TransformerEncoder` with row/column-transposed attention) | These operators accumulate results on CUDA via `atomicAdd` or shape-dependent kernel selection, neither of which `torch.backends.cudnn.deterministic = True` covers | **The same code, with the same seed, gives a different result on two consecutive GPU runs.** Measured: `DCN_Corr_Data`'s own epoch-2 validation loss was 0.6234 vs. 0.6241 across two runs; `SAINT`'s epoch-1 training loss was 0.6945 vs. 0.6946 across two runs. This is a property of the underlying CUDA kernels, not an artifact of this codebase. |
| **Cross-environment training sensitivity in deep-learning experiments** (not specific to any one chapter -- any experiment involving many training iterations can show this) | The specific implementation of random-number generation, the hardware platform, and software library versions all differ across machines; the more training epochs, the more these small floating-point errors tend to accumulate and amplify, especially affecting threshold-based metrics derived from a sign comparison (such as direction accuracy) | Every model architecture and hyperparameter matches the configuration that produced the book's published numbers; repeating a run **on the same machine, in the same environment** gives bit-for-bit identical results. But running on a different machine, or switching between GPU and CPU, produces an observable deviation in later-stage training metrics. |

---

## License and Citation

This code is provided solely for the study and research use of readers of
*Machine Learning for Quantitative Investment*. See
[data/README.md](data/README.md) for data provenance.
