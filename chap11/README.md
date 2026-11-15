# Chapter 11: Reinforcement Learning for Portfolio Management (Python)

## 11.6 Curriculum-learning + multi-objective PPO graph-RL for portfolio management

Treats portfolio management as a reinforcement-learning problem on the
CSI 100 daily panel: an E-DyGFormer encoder represents the dynamic graph,
a PolicyNet outputs portfolio weights, and PPO trains the policy, layered
with **curriculum learning** (progressively relaxed constraints across
stages) and **multi-objective optimization** (PCGrad gradient projection
with adaptive task weighting).

## Runing

```bash
python chap11/11.6/run_experiment.py --list
python chap11/11.6/run_experiment.py --experiment morl_curriculum-itransformer-30   # the default, matching the original rl_main()'s hardcoded settings
python chap11/11.6/run_grid.py                                                       # the full 16-experiment ablation grid
python chap11/11.6/curriculum_ppo_graph_rl.py                                        # equivalent thin entry point
```


## Inputs and outputs

| File | Role |
|---|---|
| `data/CSI100_stock_data_with_factors.pkl` | input data |
| `outputs/<experiment>/graph_rl_results.json` | **output**: backtest metrics, one directory per experiment |
| `outputs/<experiment>/rl_val_nav_cost.png` / `rl_test_nav_cost.png` | **output**: validation/test NAV curves |
| `outputs/<experiment>/exp_curriculum.yaml` | **output**, not an input (see below) |
| `outputs/ablation_grid/ablation_grid_results.csv` | **output**: the 16-experiment ablation grid summary |
| `curriculum_config.yaml` / `curriculum_config.json` | unused sample configs (see below) |

