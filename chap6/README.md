# Chapter 6: Attention Mechanisms (Python)

| Section | Topic | Notes |
|---|---|---|
| [6.4](6.4/) | Attention-augmented convolutional neural networks | SE, AG, BAM, CBAM, FCA, self-attention, and deformable attention, each attached to a CNN or deformable-convolution backbone |
| [6.5](6.5/) | Attention-augmented GRU | TPA (temporal pattern attention) and EleAttG (element-wise attention gate, in both pre- and post-GRU variants) |

Each section has its own `README.md`, with its experiment list, usage
instructions, and implementation notes.

## Running

```bash
python chap6/6.4/run_experiment.py --list
python chap6/6.4/run_experiment.py --experiment SE_CNN
python chap6/6.4/SE_CNN_example.py                 # equivalent thin entry point
```

