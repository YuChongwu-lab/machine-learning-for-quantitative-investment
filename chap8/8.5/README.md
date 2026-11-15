# Section 8.5: a hierarchical gated multi-edge-type dynamic GNN for quantitative investing

A graph neural network for cross-sectional stock return prediction and portfolio
construction. Four independently selectable edge types connect stocks at
each point in a rolling time window:

- **correlation edges** -- pairs whose trailing-return correlation exceeds a threshold
- **sector edges** -- pairs in the same industry
- **KNN edges** -- pairs that are nearest neighbors in feature space
- **temporal edges** -- a stock's own consecutive time steps

A gating mechanism learns how much weight to give each edge type's
message-passing output, on top of a shared feature encoder, LSTM, and
temporal attention. Predicted returns and risks feed into one of four
portfolio-construction methods (softmax long-only, mean-variance,
risk-parity, min-variance), which are then backtested with a cash +
position ledger.

## File layout

```
.
├── config.py            # SHARED hyperparameters + the EXPERIMENTS registry + get_config(name)
├── model.py              # HierarchicalGNN, TemporalGraphAttention
├── data_utils.py         # load_real_data, MultiEdgeGraphConstructor, FeatureEngineer
├── portfolio.py          # EnhancedPortfolioOptimizer (the 4 portfolio-construction methods)
├── backtest.py           # BacktestEngine (cash + position ledger, mark-to-market)
├── train_utils.py        # ModelTrainer, train_model
├── visualize.py          # edge-type / backtest plotting and debugging helpers
├── run_experiment.py     # run(name) + CLI (--experiment / --list)
├── hierarchical_multi_edge_gnn.py  # thin entry point, kept under the book-matching filename
├── paths.py
└── outputs/<experiment>/ # per-experiment checkpoint + plot outputs (gitignored)
```

## Quick start

```bash
python run_experiment.py --list
python run_experiment.py --experiment all-softmax
python hierarchical_multi_edge_gnn.py        # equivalent thin entry point, runs the default experiment
```

## Experiments

| Experiment | Edge types | Portfolio method |
|---|---|---|
| `sector` | sector | softmax long-only |
| `correlation` | correlation | softmax long-only |
| `knn` | knn | softmax long-only |
| `temporal` | temporal | softmax long-only |
| `corr-sector` | correlation + sector | softmax long-only |
| `corr-sector-knn` | correlation + sector + knn | softmax long-only |
| `all-softmax` (default) | all four | softmax long-only |
| `all-min-variance` | all four | min-variance |
| `all-risk-parity` | all four | risk-parity |
| `all-mean-variance` | all four | mean-variance |
