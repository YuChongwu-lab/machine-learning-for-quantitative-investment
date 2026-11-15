# Chapter 5: Convolutional and Recurrent Neural Networks (Python)

| Section | Topic | Notes |
|---|---|---|
| [5.5](5.5/) | Reshaping cross-sectional stock features into 9x11 images, classified with a CNN | Compares SimpleCNN, deformable convolution, ResNet18, and ConvNeXt, plus the effect of feature arrangement |
| [5.6](5.6/) | Forecasting stock prices with recurrent networks and temporal convolution | Compares LSTM, GRU, and TCN |

Each section has its own `README.md`, with its experiment list, usage
instructions.

## Running

```bash
python chap5/5.5/run_experiment.py --list          # list every experiment in this section
python chap5/5.5/run_experiment.py --experiment SimpleCNN_Corr_Data
python chap5/5.5/SimpleCNN_Corr_Data.py            # equivalent thin entry point
```

Scripts run from any working directory; outputs are written to each
section's `outputs/<experiment>/`.
