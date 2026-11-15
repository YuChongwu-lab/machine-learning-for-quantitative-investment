# Section 11.6: curriculum-learning + multi-objective PPO graph-RL for portfolio management

Treats portfolio management as a reinforcement-learning problem on the
CSI100 daily panel: an E-DyGFormer encoder represents the dynamic
(asset x time) graph, a PolicyNet (Actor-Critic) outputs a Dirichlet
allocation over assets, and PPO trains the policy, optionally layered with:

- **curriculum learning**: training proceeds through 3 stages with
  progressively tightening position/cost/turnover constraints
  (`curriculum.py`'s `CurriculumConfig.create_default()`)
- **multi-objective optimization (MORL)**: an auxiliary next-return
  prediction loss (MSE + IC) is added to the PPO loss, with PCGrad-style
  gradient-conflict projection and adaptive task weighting

## Quick start

```bash
python run_experiment.py --list
python run_experiment.py --experiment morl_curriculum-itransformer-30   # the default setting
python curriculum_ppo_graph_rl.py                                       # equivalent thin entry point
python run_grid.py                                                      # the full 16-experiment ablation grid
```

## The experiment ablation grid

The book compares:

- **4 ablation groups**: `baseline` (encoder + PPO only), `morl` (+
  multi-objective optimization), `curriculum` (+ curriculum learning),
  `morl_curriculum` (+ both)
- **2 encoder types**: `vanilla`, `itransformer`
- **2 asset-pool sizes**: `max_assets` in `{30, 120}`

4 x 2 x 2 = 16 experiments, defined in `config.py`'s `EXPERIMENTS` registry
and reconstructed by `run_grid.py`, which launches each experiment as an
**independent subprocess** and collects the results into `outputs/ablation_grid/ablation_grid_results.csv`.

## File layout

```
.
├── config.py                  # SHARED hyperparameters + the GROUPS / EXPERIMENTS hidden-grid registry
├── curriculum.py               # StageConfig, CurriculumConfig
├── metrics.py                  # cross-sectional signal, Pearson correlation, mse_ic_loss
├── model.py                     # PositionalEncoding, VariableEmbedding, EdgeAwareMHSA, iTransformerMHSA, E_DyGFormer
├── data_utils.py                # load_real_data, _construct_time_edges, create_enhanced_dataset
├── policy.py                    # PolicyNet (Actor-Critic on top of E_DyGFormer)
├── env.py                       # PortfolioEnv (a Gym-like wrapper around the backtest engine)
├── ppo.py                       # RolloutItem, PPOTrainer (GAE + Dirichlet + MORL + gradient surgery), run_curriculum_training
├── adapter.py                   # PolicyModelAdapter (wraps the policy for run_backtest)
├── backtest.py                  # EnhancedBacktestEngine
├── visualize.py                  # plot_enhanced_results
├── run_experiment.py            # run(name) + CLI (--experiment / --list / --quick)
├── run_grid.py                  # the 16-experiment ablation grid (hidden-grid reconstruction)
├── curriculum_ppo_graph_rl.py   # thin entry point, kept under the book-matching filename
├── paths.py
├── curriculum_config.yaml / curriculum_config.json  # dead, unused sample configs (see "Known issues" below)
└── outputs/<experiment>/        # per-experiment checkpoint + plot outputs (gitignored)
```

