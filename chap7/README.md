# Chapter 7: Transformers (Python)

| Section | Topic | Notes |
|---|---|---|
| [7.5](7.5/) | Image encodings of tabular data, and tabular Transformers | Two parallel pipelines: DeepInsight/MTF image encoding + a vision model; and a SAINT tabular Transformer (with optional CutMix) |
| [7.6](7.6/) | Transformer-family time-series forecasting | 12 variants: Transformer, Informer, Autoformer, FEDformer, Crossformer, iTransformer, Non-stationary Transformer, Scaleformer, plus 4 RevIN variants |

Each section has its own `README.md`, with its experiment list, usage
instructions, and implementation notes.

## Running

```bash
python chap7/7.6/run_experiment.py --list
python chap7/7.6/run_experiment.py --experiment informer
python chap7/7.6/informer_example.py               # equivalent thin entry point
```

## Standalone scripts in each section

Beyond the main experiments in the table above, both sections also include
a few standalone scripts:

- `7.5/train_by_time_group.py`: rolling training grouped by time period
- `7.5/visualize_images.py`: visualizes the effect of the tabular-to-image encoding
- `7.6/drift_detection.py`: KL-divergence-based distribution-drift detection
- `7.6/rolling_window_train.py`: compares rolling-window vs. fixed-window training
- `7.6/revin_transformer.py`: a standalone demonstration of RevIN
