# Section 8.6: an edge-aware dynamic graph Transformer with explainability

A graph-Transformer model for cross-sectional stock return prediction, combining:

- **E-DyGFormer**: a Transformer encoder over (asset x time) tokens plus a CLS
  token, with an additive edge bias derived from rolling return correlations
  (temporal edges get a zero bias; correlation edges get a tempered,
  atanh-mapped bias), and DropEdge during training
- **Data hyperparameter search**: a grid search over `corr_window` and
  `corr_threshold` (the model's own architecture hyperparameters are held
  fixed), scored by validation-set Spearman-IC
- **Gradient-based explainability**: per-sample feature-importance via
  input gradients, visualized as a time x asset heatmap and bar charts
- **Backtest with realistic costs**: linear transaction cost + a
  turnover-scaled market-impact cost, a minimum-trade-size cutoff, and
  information-ratio / tracking-error reporting against an equal-weight benchmark

## File layout

```
.
├── config.py                  # SHARED hyperparameters + ASSET_POOL_GRID / EXPLAIN_MAX_ASSETS
├── model.py                   # PositionalEncoding, EdgeAwareMHSA, EdgeAwareEncoderLayer, E_DyGFormer
├── metrics.py                 # cross-sectional signal, rank/Spearman/Pearson correlation, mse_ic_loss
├── data_utils.py               # load_real_data, create_enhanced_dataset
├── hypersearch.py              # HyperparameterSearchEngine (corr_window / corr_threshold)
├── train_utils.py              # train_enhanced_model
├── explain.py                  # ModelExplainer (gradient-based explainability)
├── backtest.py                 # EnhancedBacktestEngine
├── visualize.py                 # plot_enhanced_results
├── run_experiment.py           # run(max_assets=..., max_days=...) + CLI
├── run_asset_pool_grid.py      # asset-pool-size sensitivity grid (hidden-grid reconstruction, part 1)
├── dynamic_graph_transformer.py  # thin entry point, kept under the book-matching filename
├── paths.py
└── outputs/<run_name>/         # per-run checkpoint + plot outputs (gitignored)
```

## Quick start

```bash
python run_experiment.py                      # default: max_assets=180, max_days=500 (matches the book)
python run_experiment.py --max-assets 30       # explainability analysis setting
python dynamic_graph_transformer.py            # equivalent thin entry point
python run_asset_pool_grid.py                 # asset-pool-size sensitivity
```

