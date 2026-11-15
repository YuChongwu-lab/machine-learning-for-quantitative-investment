"""Unified entry point for the Section 11.6 experiments.

Usage:

    python run_experiment.py                                     # default experiment 
    python run_experiment.py --experiment baseline-vanilla-30
    python run_experiment.py --list
    python curriculum_ppo_graph_rl.py                             # equivalent thin entry point
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from adapter import PolicyModelAdapter
from backtest import EnhancedBacktestEngine
from config import DEFAULT_EXPERIMENT, EXPERIMENTS, get_config
from curriculum import CurriculumConfig
from data_utils import create_enhanced_dataset, load_real_data
from env import PortfolioEnv
from model import E_DyGFormer
from paths import data_file
from policy import PolicyNet
from ppo import PPOTrainer, run_curriculum_training
from visualize import plot_enhanced_results

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(s: int = 42) -> None:
    """Fix the random seed."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)

#: Output root. 
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def run(name: str = DEFAULT_EXPERIMENT) -> dict:
    """
    Graph-RL main pipeline:load data -> build the graph -> split the dataset -> build the
    E_DyGFormer encoder + PolicyNet (Actor-Critic) -> PPO training (short
    horizons) -> deterministic evaluation via run_backtest.

    Parameters:
    - name: experiment name in config.EXPERIMENTS, selecting the ablation
      group (baseline / morl / curriculum / morl_curriculum), encoder_type
      (vanilla / itransformer), and max_assets (30 / 120)
    """
    cfg = get_config(name)

    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        # === 1) Data loading ===
        default_file = str(data_file(cfg["data_file"]))
        max_assets = cfg["max_assets"]
        max_days = cfg["max_days"]
        price, bench, panel = load_real_data(default_file, max_assets=max_assets, max_days=max_days)
        seq_len = cfg["seq_len"]
        dataset, feat_names = create_enhanced_dataset(
            panel, seq_len=seq_len,
            corr_window=cfg["dataset"]["corr_window"],
            corr_threshold=cfg["dataset"]["corr_threshold"],
        )

        # split into train/val/test sets
        n_all = len(dataset)
        train_size = int(cfg["train_ratio"] * n_all)
        val_size = int(cfg["val_ratio"] * n_all)
        train_ds = dataset[:train_size]
        val_ds = dataset[train_size:train_size + val_size]
        test_ds = dataset[train_size + val_size:]

        # === 2) Encoder & policy ===
        feat_dim = train_ds[0].x.shape[1]
        num_assets = len(price.columns)
        enc_cfg = dict(
            node_feat_dim=feat_dim,
            num_assets=num_assets,
            seq_len=seq_len,
            encoder_type=cfg["encoder_type"],
            **cfg["encoder"],
        )
        encoder = E_DyGFormer(**enc_cfg)

        policy = PolicyNet(encoder, **cfg["policy"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy = policy.to(device)

        # === 3) Environment & trainer ===
        engine_cfg = dict(cfg["engine"])
        reward_cfg = dict(cfg["reward"])

        use_curriculum = cfg["use_curriculum"]
        use_morl = cfg["use_morl"]
        morl_args = dict(
            morl_enable=use_morl,
            **cfg["morl"],
        )

        if use_curriculum:
            curriculum_config = CurriculumConfig.create_default()

            trainer_kwargs = dict(cfg["trainer"])
            trainer_kwargs.update(morl_args)

            run_curriculum_training(
                policy, price, train_ds,
                base_engine_cfg=engine_cfg,
                base_reward_cfg=reward_cfg,
                trainer_kwargs=trainer_kwargs,
                curriculum_config=curriculum_config,
                horizon=cfg["horizon"],
            )
        else:
            env = PortfolioEnv(policy, train_ds, price, engine_cfg, reward_cfg)
            trainer = PPOTrainer(policy, env, device=device, **cfg["trainer"], **morl_args)
            epochs = cfg["flat_epochs"]
            horizon = cfg["horizon"]
            for ep in range(1, epochs + 1):
                buf = trainer.collect(horizon=horizon, stochastic=True)
                trainer.update(buf)
                if use_morl and (ep % 5 == 0):
                    # roughly eyeball the trend with just the last batch: MSE/IC
                    last_batch_data = buf[-1].data
                    with torch.no_grad():
                        pr = policy.predict_next_return(last_batch_data)
                        y = last_batch_data.y.to(pr.device)
                        mse = F.mse_loss(pr, y).item()
                        rx = pr.argsort().argsort().float();
                        ry = y.argsort().argsort().float()
                        ic = float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])
                    print(f"[MORL] epoch {ep}: aux MSE={mse:.4e}, IC={ic:+.3f}")
                if ep % 5 == 0:
                    last_V = float(buf[-1].V.detach().cpu().item())
                    print(f"[PPO] epoch {ep}/{epochs} | last V~={last_V:,.2f}")

        # === 5) Deterministic backtest evaluation (on the validation/test sets) ===
        adapter_cfg = cfg["adapter"]
        adapter = PolicyModelAdapter(policy, **adapter_cfg)
        adapter.reset_state()
        bengine = EnhancedBacktestEngine(**engine_cfg)

        # validation set
        price_val = price.iloc[-(len(val_ds)+len(test_ds)):-len(test_ds)]
        bench_val = bench.iloc[-(len(val_ds)+len(test_ds)):-len(test_ds)]
        res_val = bengine.run_backtest(adapter, val_ds, price_val, bench_val, temperature=policy.temperature)

        # test set
        bengine2 = EnhancedBacktestEngine(**engine_cfg)
        price_test = price.iloc[-len(test_ds):]
        bench_test = bench.iloc[-len(test_ds):]
        adapter.reset_state()  # reset the EMA smoothing state, so the validation period doesn't "leak" into the test period
        res_test = bengine2.run_backtest(adapter, test_ds, price_test, bench_test, temperature=policy.temperature)

        print("\n=== Graph-RL Evaluation ===")
        print(f"VAL:  ann.ret={res_val['annual_return']:.2%}, sharpe={res_val['sharpe_ratio']:.3f}, "
              f"IR={res_val['information_ratio']:.3f}, MDD={res_val['max_drawdown']:.2%}")
        print(f"TEST: ann.ret={res_test['annual_return']:.2%}, sharpe={res_test['sharpe_ratio']:.3f}, "
              f"IR={res_test['information_ratio']:.3f}, MDD={res_test['max_drawdown']:.2%}")

        # total return, benchmark return, cost ratio, average holdings
        print(f"VAL+: total_ret={res_val['total_return']:.2%}, bench_ret={res_val['benchmark_total_return']:.2%}, "
              f"cost_ratio={res_val['transaction_cost_ratio']:.2%}, avg_holdings={res_val['avg_holdings']:.2f}")
        print(f"TEST+: total_ret={res_test['total_return']:.2%}, bench_ret={res_test['benchmark_total_return']:.2%}, "
              f"cost_ratio={res_test['transaction_cost_ratio']:.2%}, avg_holdings={res_test['avg_holdings']:.2f}")

        # plots: transaction costs + NAV curve (validation and test plotted separately)
        try:
            plot_enhanced_results(res_val, bengine.portfolio_value, bengine.benchmark_levels,
                                  save_path='rl_val_nav_cost.png')
            plot_enhanced_results(res_test, bengine2.portfolio_value, bengine2.benchmark_levels, save_path='rl_test_nav_cost.png')
        except Exception as e:
            print(f"[Plot] Plotting failed: {e}")

        return dict(policy=policy, res_val=res_val, res_test=res_test, engines={'val': bengine, 'test': bengine2})
    finally:
        os.chdir(prev_cwd)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Section 11.6: curriculum-learning + multi-objective PPO graph-RL quantitative investing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available experiments:\n" + "\n".join(
            f"  {k:30} group={v['group']:16} encoder={v['encoder_type']:12} max_assets={v['max_assets']}"
            for k, v in EXPERIMENTS.items()))
    ap.add_argument("--experiment", "-e", default=DEFAULT_EXPERIMENT,
                    choices=list(EXPERIMENTS),
                    help=f"experiment to run (default {DEFAULT_EXPERIMENT}, matching rl_main()'s own hardcoded settings)")
    ap.add_argument("--list", action="store_true", help="list all experiments and exit")
    ap.add_argument("--quick", action="store_true",
                    help="drastically reduce settings (max_days, horizon, PPO minibatch/update epochs) for a fast smoke test")
    return ap.parse_args()


if __name__ == "__main__":
    import json
    import time
    import traceback

    import config as _config

    args = parse_args()

    if args.list:
        print(f"{'Experiment':30}{'Group':16}{'Encoder':12}{'max_assets'}")
        for name, exp_cfg in EXPERIMENTS.items():
            mark = "  <- default" if name == DEFAULT_EXPERIMENT else ""
            print(f"{name:30}{exp_cfg['group']:16}{exp_cfg['encoder_type']:12}{exp_cfg['max_assets']}{mark}")
        raise SystemExit(0)

    if args.quick:
        _config.SHARED["max_days"] = 120
        _config.SHARED["horizon"] = 16
        _config.SHARED["trainer"]["update_epochs"] = 1
        _config.SHARED["trainer"]["minibatch_size"] = 8

    # Note: this line writes a 5-stage "progressive" curriculum config to
    # exp_curriculum.yaml, but this code never actually uses that file -- what
    # takes effect is CurriculumConfig.create_default() (3 stages) inside run().
    # That is why the variable is named _unused_progressive_config.
    # However, readers are encouraged to extend this and try other interesting curriculum settings. 
    _unused_progressive_config = CurriculumConfig.create_progressive(num_stages=5, total_epochs=30)
    out_dir = OUTPUT_ROOT / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    _unused_progressive_config.to_yaml(str(out_dir / 'exp_curriculum.yaml'))

    t0 = time.time()
    try:
        out = run(args.experiment)
        # save the key evaluation metrics to JSON
        res = {
            "val": {
                "annual_return": out.get("res_val", {}).get("annual_return", 0.0),
                "sharpe_ratio": out.get("res_val", {}).get("sharpe_ratio", 0.0),
                "information_ratio": out.get("res_val", {}).get("information_ratio", 0.0),
                "max_drawdown": out.get("res_val", {}).get("max_drawdown", 0.0)
            },
            "test": {
                "annual_return": out.get("res_test", {}).get("annual_return", 0.0),
                "sharpe_ratio": out.get("res_test", {}).get("sharpe_ratio", 0.0),
                "information_ratio": out.get("res_test", {}).get("information_ratio", 0.0),
                "max_drawdown": out.get("res_test", {}).get("max_drawdown", 0.0)
            }
        }
        with open(out_dir / "graph_rl_results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"Training complete. Results saved to {out_dir / 'graph_rl_results.json'}")
    except Exception as e:
        print("Run failed:", e)
        traceback.print_exc()
    finally:
        print(f"Total time: {time.time()-t0:.2f}s")
