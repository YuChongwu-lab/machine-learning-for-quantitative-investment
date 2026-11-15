"""Curriculum-learning configuration: a sequence of training stages with
progressively tightening position/cost/turnover constraints and loss weights."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single training stage."""
    name: str
    epochs: int
    max_pos: float  # max weight for a single asset
    cost: float  # transaction cost rate
    lambda_turnover: float  # turnover penalty coefficient
    loss_weight: Dict[str, float] = field(default_factory=lambda: {'mse': 1.0, 'ic': 0.0, 'return': 0.0})

    def __post_init__(self):
        """Validate the parameters."""
        assert self.epochs > 0, f"epochs must be > 0, got: {self.epochs}"
        assert 0 < self.max_pos <= 1, f"max_pos must be in (0,1], got: {self.max_pos}"
        assert self.cost >= 0, f"cost must be >= 0, got: {self.cost}"
        assert self.lambda_turnover >= 0, f"lambda_turnover must be >= 0, got: {self.lambda_turnover}"

        # validate loss_weight
        required_keys = {'mse', 'ic', 'return'}
        assert set(self.loss_weight.keys()) == required_keys, \
            f"loss_weight must contain {required_keys}, got: {set(self.loss_weight.keys())}"


@dataclass
class CurriculumConfig:
    """A full curriculum-learning configuration."""
    stages: List[StageConfig]

    @classmethod
    def from_dict(cls, config: Dict) -> 'CurriculumConfig':
        """Build a config from a dict."""
        stages = [StageConfig(**stage) for stage in config['stages']]
        return cls(stages=stages)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CurriculumConfig':
        """Load a config from a YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @classmethod
    def from_json(cls, json_path: str) -> 'CurriculumConfig':
        """Load a config from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls.from_dict(config)

    def to_yaml(self, yaml_path: str):
        """Save the config as YAML."""
        config = {'stages': [asdict(stage) for stage in self.stages]}
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def to_json(self, json_path: str):
        """Save the config as JSON."""
        config = {'stages': [asdict(stage) for stage in self.stages]}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def create_default(cls) -> 'CurriculumConfig':
        """Create the default 3-stage configuration."""
        stages = [
            StageConfig(
                name='Stage-1-Warmup',
                epochs=8,
                max_pos=0.30,
                cost=0.001,
                lambda_turnover=0.02,
                loss_weight={'mse': 1.0, 'ic': 0.0, 'return': 0.0}
            ),
            StageConfig(
                name='Stage-2-Balance',
                epochs=10,
                max_pos=0.25,
                cost=0.002,
                lambda_turnover=0.04,
                loss_weight={'mse': 0.5, 'ic': 0.5, 'return': 0.0}
            ),
            StageConfig(
                name='Stage-3-Finetune',
                epochs=12,
                max_pos=0.20,
                cost=0.003,
                lambda_turnover=0.05,
                loss_weight={'mse': 0.0, 'ic': 0.2, 'return': 1.0}
            ),
        ]
        return cls(stages=stages)

    @classmethod
    def create_progressive(cls,
                           num_stages: int = 5,
                           total_epochs: int = 40,
                           start_max_pos: float = 0.35,
                           end_max_pos: float = 0.15,
                           start_cost: float = 0.0005,
                           end_cost: float = 0.004,
                           start_lambda: float = 0.01,
                           end_lambda: float = 0.08) -> 'CurriculumConfig':
        """
        Auto-generate a progressive curriculum.

        Args:
            num_stages: number of stages
            total_epochs: total number of training epochs
            start_max_pos: initial max single-asset weight (loose)
            end_max_pos: final max single-asset weight (strict)
            start_cost: initial transaction cost (low)
            end_cost: final transaction cost (high)
            start_lambda: initial turnover penalty (low)
            end_lambda: final turnover penalty (high)
        """
        stages = []
        epochs_per_stage = max(1, total_epochs // num_stages)

        for i in range(num_stages):
            # linear interpolation
            progress = i / max(1, num_stages - 1)  # 0 -> 1

            # parameters move from loose to strict
            max_pos = start_max_pos + (end_max_pos - start_max_pos) * progress
            cost = start_cost + (end_cost - start_cost) * progress
            lambda_t = start_lambda + (end_lambda - start_lambda) * progress

            # loss weights: transition from MSE -> IC+Return
            if i < num_stages // 3:
                # first third: focus on MSE (prediction accuracy)
                loss_weight = {'mse': 1.0, 'ic': 0.0, 'return': 0.0}
            elif i < 2 * num_stages // 3:
                # middle third: MSE + IC balance
                loss_weight = {'mse': 0.5, 'ic': 0.5, 'return': 0.0}
            else:
                # final third: IC + Return (rank correlation + return)
                ic_weight = 0.2 + 0.3 * (progress - 2 / 3) * 3  # 0.2 -> 0.5
                loss_weight = {'mse': 0.0, 'ic': ic_weight, 'return': 1.0 - ic_weight}

            # the last stage trains for a few more epochs
            epochs = epochs_per_stage if i < num_stages - 1 else (total_epochs - epochs_per_stage * (num_stages - 1))

            stages.append(StageConfig(
                name=f'Stage-{i + 1}-Progressive',
                epochs=epochs,
                max_pos=round(max_pos, 3),
                cost=round(cost, 5),
                lambda_turnover=round(lambda_t, 3),
                loss_weight=loss_weight
            ))

        return cls(stages=stages)

    def validate_progression(self) -> bool:
        """Check whether the curriculum is a sensible progression (increasing difficulty)."""
        for i in range(len(self.stages) - 1):
            curr = self.stages[i]
            next_stage = self.stages[i + 1]

            # check whether the constraints are actually tightening
            if next_stage.max_pos > curr.max_pos:
                logger.warning(f"{curr.name} -> {next_stage.name}: max_pos got looser (may be unintended)")
            if next_stage.cost < curr.cost:
                logger.warning(f"{curr.name} -> {next_stage.name}: cost decreased (may be unintended)")
            if next_stage.lambda_turnover < curr.lambda_turnover:
                logger.warning(f"{curr.name} -> {next_stage.name}: lambda_turnover decreased (may be unintended)")

        return True

    def print_summary(self):
        """Print a summary of the curriculum."""
        print("\n" + "=" * 60)
        print("Curriculum-Learning Configuration Summary".center(60))
        print("=" * 60)
        print(f"Total stages: {len(self.stages)}")
        print(f"Total training epochs: {sum(s.epochs for s in self.stages)}")
        print("-" * 60)
        for i, stage in enumerate(self.stages, 1):
            print(f"\n[{stage.name}]")
            print(f"Epochs: {stage.epochs}")
            print(f"Constraints: max_pos={stage.max_pos:.2%}, cost={stage.cost:.4%}, lambda_turn={stage.lambda_turnover:.3f}")
            print(
                f"  Loss weights: MSE={stage.loss_weight['mse']:.1f}, IC={stage.loss_weight['ic']:.1f}, Return={stage.loss_weight['return']:.1f}")
        print("=" * 60 + "\n")
