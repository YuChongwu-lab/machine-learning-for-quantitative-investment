# Chapter 8: Dynamic Graph Neural Networks (Python)

Both sections take the CSI 100 daily panel,
`data/CSI100_stock_data_with_factors.pkl` (437,656 rows, 190 stocks,
2015-01 through 2025-08), build a dynamic graph of relationships between
stocks, and use it for portfolio construction and backtesting.

## 8.5 A hierarchical, multi-edge-type dynamic GNN

```bash
python chap8/8.5/run_experiment.py --list
python chap8/8.5/run_experiment.py --experiment all-softmax
python chap8/8.5/hierarchical_multi_edge_gnn.py   # equivalent thin entry point, runs the default experiment
```

This section builds **four edge types** -- correlation, sector, KNN, and
temporal -- fuses them with a hierarchical gated GNN, and compares four
portfolio-construction methods.

| Experiment | Edge types | Portfolio method |
|---|---|---|
| `sector` / `correlation` / `knn` / `temporal` | a single edge type (ablation) | softmax long-only |
| `corr-sector` | correlation + sector | softmax long-only |
| `corr-sector-knn` | correlation + sector + knn | softmax long-only |
| **`all-softmax` (default)** | all four | softmax long-only |
| `all-min-variance` | all four | min-variance |
| `all-risk-parity` | all four | risk-parity |
| `all-mean-variance` | all four | mean-variance |

## 8.6 Dynamic graph Transformer and explainability

```bash
python chap8/8.6/run_experiment.py                    # default max_assets=180 (matches the book)
python chap8/8.6/run_experiment.py --max-assets 30
python chap8/8.6/run_asset_pool_grid.py                # asset-pool-size sensitivity grid (30/60/90/120/150/180)
python chap8/8.6/dynamic_graph_transformer.py          # equivalent thin entry point
```

E-DyGFormer (an edge-aware dynamic graph Transformer) + a data-hyperparameter
search (`corr_window`/`corr_threshold`) + gradient-based model explainability
(attention heatmaps, feature attribution) + backtesting with trading and
market-impact costs.

## Dependencies

Beyond the core dependencies in the top-level `requirements.txt`, this
chapter also needs `torch-geometric` (8.5 uses `GATConv`/`TransformerConv`;
8.6 uses the `Data` container) and `networkx` (for 8.5's edge-structure
visualization).
