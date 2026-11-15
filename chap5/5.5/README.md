# 5.5 CNN classification of cross-sectional stock data

Each stock's 93 contemporaneous features are zero-padded to 99 and
reshaped into a **9x11 single-channel image**, then classified with a
CNN (future returns factorized into two classes by the median). This
section compares four backbones, and the effect of feature ordering
within the image on the results.

## Running

```bash
python run_experiment.py --list                          # list all experiments and their settings
python run_experiment.py --experiment SimpleCNN_Corr_Data
python SimpleCNN_Corr_Data.py                            # equivalent thin entry point
```

Artifacts are written to `outputs/<experiment>/`: `best_model.pt`,
`loss_curve.png`, `val_metrics.png`, `confusion_matrix.png`.

## Experiment list

| Experiment | Backbone | Feature order | lr | val | weight_decay |
|---|---|---|---|---|---|
| `SimpleCNN_Raw_Data` | SimpleCNN | original | 1e-3 | 0.2 | — |
| `SimpleCNN_Corr_Data` | SimpleCNN | reordered by correlation | 1e-3 | 0.2 | — |
| `SimpleCNN_Corr_abs_data` | SimpleCNN | reordered by **absolute** correlation | 1e-3 | 0.2 | — |
| `DCN_Raw_Data` | deformable convolution | original | 1e-3 | 0.2 | — |
| `DCN_Corr_Data` | deformable convolution | reordered by correlation | 1e-3 | 0.2 | — |
| `DCN_Corr_Data_tuned` | deformable convolution | reordered by correlation | **1e-4** | **0.3** | **1e-4** |
| `Resnet_Corr_Data` | ResNet18 | reordered by correlation | 1e-3 | 0.2 | — |
| `ConvNext_Corr_Data` | ConvNeXt | reordered by correlation | 1e-3 | 0.2 | — |

Shared settings: `num_classes=2`, `test_size=0.2`, `patience=10`,
`epochs=500`, `train_test_split(random_state=42)`.

## File layout

```
config.py                 # SHARED (shared hyperparameters) + EXPERIMENTS 
run_experiment.py         # the single implementation: model registry + train/eval pipeline + CLI
data_utils.py             # feature reordering, cross-section-to-9x11-image conversion, DataLoader
train_utils.py            # training loop, evaluation, result visualization
get_data.py               # generates data_ml_2017_11.csv from data_ml.csv 
AugmentedImageDataset.py  # optional augmented DataLoader, not used by the official experiments
model/                    # SimpleCNN / DCN / ResNet18 / ConvNeXt
<experiment name>.py       # thin entry points
```
