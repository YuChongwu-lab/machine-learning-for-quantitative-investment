"""Thin entry point.

See run_experiment.py for the actual implementation; this just runs the
default experiment (the ablation group with both multi-objective
optimization and curriculum learning enabled, iTransformer encoder,
max_assets=30), matching rl_main()'s own hardcoded settings.
"""

import json
import time
import traceback

from config import DEFAULT_EXPERIMENT
from curriculum import CurriculumConfig
from run_experiment import OUTPUT_ROOT, run

if __name__ == "__main__":
    _unused_progressive_config = CurriculumConfig.create_progressive(num_stages=5, total_epochs=30)
    out_dir = OUTPUT_ROOT / DEFAULT_EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    _unused_progressive_config.to_yaml(str(out_dir / 'exp_curriculum.yaml'))

    t0 = time.time()
    try:
        out = run(DEFAULT_EXPERIMENT)
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
