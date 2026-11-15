# Chapter 10: Deep Generative Models (Python)

| Section | Topic | Notes |
|---|---|---|
| [10.6](10.6/AdaptiveStockForecaster/) | Deep generative models in an adaptive online stock-forecasting system | Based on the Proceed framework: TCVAE concept encoding + distribution-drift detection + a gated adapter that modulates a PatchTST backbone, with optional DDPM data augmentation and online fine-tuning |

This section has its own `README.md`, with its experiment list, usage
instructions, and implementation notes.

## Running

```bash
python chap10/10.6/AdaptiveStockForecaster/run_experiment.py
python chap10/10.6/AdaptiveStockForecaster/run_baseline_grid.py --max-assets 10
python chap10/10.6/AdaptiveStockForecaster/hypersearch_multi_obj.py --trials 30
python chap10/10.6/AdaptiveStockForecaster/drift_analysis.py
```

Scripts run from any working directory; all data lives in the top-level
`data/` directory.
