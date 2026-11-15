# 6.4 Attention-enhanced convolutional neural networks

This section attaches 8 attention modules to a CNN / deformable
convolution (DCN) backbone, classifying 9x11 images converted from
cross-sectional stock data, comparing the effect of each attention
mechanism.

## Running

```bash
python run_experiment.py --list                 # list all experiments
python run_experiment.py --experiment SE_CNN    # run a specific experiment
python SE_CNN_example.py                        # equivalent thin entry point
```

Artifacts are written to `outputs/<experiment>/`: `best_model.pt`,
`loss_curve.png`, `val_metrics.png`, `confusion_matrix.png`.

## Experiment list

| Experiment | Attention mechanism | Backbone |
|---|---|---|
| `SE_CNN` | Squeeze-and-Excitation | CNN |
| `AG_CNN` | Attention Gate | CNN |
| `BAM_CNN` | Bottleneck Attention Module | CNN |
| `CBAM_CNN` | Convolutional Block Attention Module | CNN |
| `FCA_CNN` | Frequency Channel Attention | CNN |
| `SelfAttn_CNN` | self-attention | CNN |
| `DeformAttn_CNN` | deformable attention | CNN |
| `BAM_DCN` | Bottleneck Attention Module | deformable convolution |
| `DeformAttn_DCN` | deformable attention | deformable convolution ※ |

This section's 9 experiments differ **only in the model**; hyperparameters
are identical: `num_classes=2`, `lr=1e-4`, `weight_decay=1e-4`,
`test_size=0.2`, `val_size=0.3`, `patience=10`, `epochs=500`,
`use_abs=False`, `train_test_split(random_state=42)`.

## File layout

```
config.py            # SHARED (shared hyperparameters) + EXPERIMENTS (9 entries)
run_experiment.py    # the single implementation: registry + train/eval pipeline + CLI
data_utils.py        # correlation-based feature reordering, cross-section-to-image conversion, DataLoader
train_utils.py       # training loop, evaluation, result visualization
model/               # the 8 attention modules combined with each backbone
*_example.py         # 9 thin entry points, keeping the book's filenames
```
