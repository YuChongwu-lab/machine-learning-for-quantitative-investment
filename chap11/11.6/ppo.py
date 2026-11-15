"""A lightweight PPO trainer (GAE + Dirichlet actions), with optional
multi-objective (MORL) auxiliary prediction loss and curriculum-learning
stage scheduling."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch_geometric.data import Data

from curriculum import CurriculumConfig
from env import PortfolioEnv
from metrics import _compute_signal, _pearson_corr
from policy import PolicyNet

logger = logging.getLogger(__name__)


# ===== Curriculum-learning schedule (doesn't change the asset count, just ramps up difficulty) =====
def run_curriculum_training(policy: 'PolicyNet',
                            price_df,
                            train_ds: List[Data],
                            base_engine_cfg: Dict,
                            base_reward_cfg: Dict,
                            trainer_kwargs: Dict,
                            curriculum_config: Optional[CurriculumConfig] = None,
                            horizon: int = 128):
    """
    Train stage by stage: progressively raise the difficulty per the curriculum config.
    Args:
        curriculum_config: a CurriculumConfig; if None, the default config is used.
    """
    # if no config was given, use the default 3-stage config
    if curriculum_config is None:
        logger.info("No curriculum config given; using the default 3-stage config")
        curriculum_config = CurriculumConfig.create_default()

    # validate and print the curriculum summary
    curriculum_config.validate_progression()
    curriculum_config.print_summary()

    device = next(policy.parameters()).device

    for stage_idx, stage in enumerate(curriculum_config.stages, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting training for {stage.name} ({stage_idx}/{len(curriculum_config.stages)})")
        logger.info(f"{'=' * 60}")

        # update the policy's position cap
        policy.max_pos = stage.max_pos

        # update the environment config (transaction cost, etc.)
        eng_cfg = dict(base_engine_cfg)
        eng_cfg['max_position_ratio'] = stage.max_pos
        eng_cfg['transaction_cost_rate'] = stage.cost

        rew_cfg = dict(base_reward_cfg)
        rew_cfg['lambda_turnover'] = stage.lambda_turnover

        # create the environment
        env = PortfolioEnv(policy, train_ds, price_df, eng_cfg, rew_cfg)

        # create the trainer (passing the loss weights)
        trainer = PPOTrainer(
            policy, env,
            **trainer_kwargs,
            device=device,
            loss_weights=stage.loss_weight
        )

        # train this stage
        for ep in range(1, stage.epochs + 1):
            buf = trainer.collect(horizon=horizon, stochastic=True)
            trainer.update(buf)

            if ep % 5 == 0 or ep == stage.epochs:
                last_V = float(buf[-1].V.detach().cpu().item())
                logger.info(f"  [{stage.name}] Epoch {ep}/{stage.epochs} | V~={last_V:,.2f}")

        # stage-completion summary
        logger.info(f" Finished {stage.name}")
        logger.info(f"  Config: max_pos={stage.max_pos:.2%}, cost={stage.cost:.4%}, lambda_turn={stage.lambda_turnover:.3f}")


class RolloutItem:
    __slots__ = ('data', 'a_raw', 'a_proj', 'logp', 'V', 'V_next', 'r', 'done')
    def __init__(self, data, a_raw, a_proj, logp, V, V_next, r, done):
        self.data = data  # the current state (a graph sample Data)
        self.a_raw = a_raw  # the raw action (a Dirichlet sample or deterministic softmax weights), before capped-simplex projection
        self.a_proj = a_proj  # the action after capped-simplex projection (satisfies the cap and normalization)
        self.logp = logp  # the log-probability of a_raw under the current policy (used for the PPO importance-sampling ratio)
        self.V = V  # the value function V(s) estimated for the current state
        self.V_next = V_next  # the next state's value function (for bootstrapping; 0 if terminal)
        self.r = r  # the reward received at this time step
        self.done = done  # whether this is a terminal state


class PPOTrainer:
    def __init__(self, policy: PolicyNet, env: PortfolioEnv,
                 gamma=0.99, lam=0.95, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01,
                 lr=3e-4, minibatch_size=16, update_epochs=4, device=None,
                 loss_weights=None, morl_enable=False, morl_aux_weight=0.5,
                 morl_alpha: float = 1.0, morl_beta: float = 0.5,
                 use_grad_surgery: bool = True,  # whether to use gradient surgery
                 adaptive_weights: bool = True):  # whether to dynamically adjust the task weights
        self.policy = policy  # the policy network
        self.env = env  # the environment

        # PPO hyperparameters
        self.gamma = gamma  # the reward discount factor
        self.lam = lam  # the GAE(lambda) parameter
        self.clip_ratio = clip_ratio  # the PPO clipping ratio
        self.vf_coef = vf_coef  # the value-loss weight
        self.ent_coef = ent_coef  # the entropy-regularization weight, encourages exploration

        # optimizer
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

        # training-related parameters
        self.mb = minibatch_size  # the minibatch size for each update
        self.update_epochs = update_epochs  # how many update passes per rollout
        self.device = device or next(policy.parameters()).device

        # prediction-task loss weights (MSE/IC/return)
        self.loss_weights = loss_weights if loss_weights else {'mse': 1.0, 'ic': 1.0, 'return': 1.0}

        # multi-objective (MORL) settings
        self.morl_enable = morl_enable
        self.morl_aux_weight = morl_aux_weight  # lambda: the overall weight of the auxiliary loss added to the PPO loss
        self.morl_alpha = float(morl_alpha)  # the MSE coefficient alpha in the supervised auxiliary loss
        self.morl_beta = float(morl_beta)  # the -IC coefficient beta in the supervised auxiliary loss (larger emphasizes rank correlation more)

        # whether to use gradient surgery (resolves multi-task gradient conflicts)
        self.use_grad_surgery = use_grad_surgery

        # whether to dynamically adjust the RL/prediction task weights
        self.adaptive_weights = adaptive_weights

        # tracks the RL and prediction loss history, to help with dynamic weighting
        self.loss_history = {
            'rl': [],  # RL loss history (policy+value+entropy)
            'pred': []  # prediction loss history (MSE+IC)
        }

    def _compute_grad_norm(self, loss_component: torch.Tensor) -> float:
        """Compute a loss component's gradient norm, for analysis or weight tuning."""
        self.opt.zero_grad()
        loss_component.backward(retain_graph=True)
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _project_conflicting_gradients(self, grads_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        A simplified PCGrad: detects and projects away conflicts between
        multi-task gradients, to avoid the directions canceling each other out.
        grads_list: [grad_task1, grad_task2, ...], one parameter-gradient dict per task
        each grad_task is {param_name: gradient_tensor}
        """
        num_tasks = len(grads_list)
        if num_tasks <= 1:
            return grads_list[0] if grads_list else {}

        # work with the param names as a list for convenience
        param_names = list(grads_list[0].keys())

        # check for conflicts between every pair of tasks and project them
        merged_grads = {name: torch.zeros_like(grads_list[0][name]) for name in param_names}

        # iterate over each task
        for i in range(num_tasks):
            grad_i = grads_list[i]
            projected_grad = {name: grad_i[name].clone() for name in param_names}

            # check for conflicts against every other task's gradient
            for j in range(num_tasks):
                if i == j:
                    continue
                grad_j = grads_list[j]

                # gradient dot product: a negative value means the directions conflict
                dot_product = 0.0
                for name in param_names:
                    if grad_i[name] is not None and grad_j[name] is not None:
                        dot_product += (grad_i[name] * grad_j[name]).sum().item()

                # if they conflict (negative dot product), project away the conflicting
                # component -- remove grad_i's component along the grad_j direction
                if dot_product < 0:
                    # the squared norm of grad_j
                    norm_j_sq = sum((grad_j[name] ** 2).sum().item() for name in param_names)
                    if norm_j_sq > 1e-8:
                        # projection: g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j
                        for name in param_names:
                            projected_grad[name] = projected_grad[name] - (dot_product / norm_j_sq) * grad_j[name]

            # accumulate the projected gradients
            for name in param_names:
                merged_grads[name] += projected_grad[name]

        # average
        for name in param_names:
            merged_grads[name] /= num_tasks

        return merged_grads

    def _update_adaptive_weights(self, rl_loss: float, pred_loss: float):
        """Dynamically adjust the prediction task's weight based on how fast
        each loss is decreasing, to balance them automatically."""
        self.loss_history['rl'].append(rl_loss)
        self.loss_history['pred'].append(pred_loss)

        # keep the most recent 20 steps of history
        if len(self.loss_history['rl']) > 20:
            self.loss_history['rl'] = self.loss_history['rl'][-20:]
            self.loss_history['pred'] = self.loss_history['pred'][-20:]

        # the rate of change over the last 5 steps
        if len(self.loss_history['rl']) >= 5:
            rl_rate = np.mean(np.diff(self.loss_history['rl'][-5:]))
            pred_rate = np.mean(np.diff(self.loss_history['pred'][-5:]))

            # normalize into the [0.2, 0.8] range
            total_rate = abs(rl_rate) + abs(pred_rate) + 1e-6
            # the faster-decreasing task gets a lower weight
            pred_weight = 0.2 + 0.6 * (abs(rl_rate) / total_rate)
            self.morl_aux_weight = float(np.clip(pred_weight, 0.2, 0.8))

    @torch.no_grad()
    def collect(self, horizon=128, stochastic=True) -> List[RolloutItem]:
        """Interact with the environment and sample a trajectory for training."""
        buf: List[RolloutItem] = []
        s = self.env.reset()
        while len(buf) < horizon:
            # sample an action from the policy
            a_raw, a_proj, logp, V = self.policy.act(s, stochastic=stochastic)
            s_next, r, done, info = self.env.step(a_proj.cpu().numpy())

            # the next state's value
            if (not done) and (s_next is not None):
                _, V_next, _ = self.policy(s_next)
                V_next = V_next.detach()
            else:
                V_next = torch.zeros_like(V)
            buf.append(RolloutItem(
                s, a_raw, a_proj, logp, V, V_next, torch.tensor(r, dtype=torch.float32), done))
            s = s_next if not done else self.env.reset()
        return buf

    def _gae(self, buf: List[RolloutItem]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Sequence[Data]]:
        """Compute the GAE advantage & TD(lambda) returns."""
        rewards = torch.stack([it.r for it in buf]).to(self.device)
        dones = torch.tensor([it.done for it in buf], dtype=torch.float32, device=self.device)
        values = torch.stack([it.V for it in buf]).to(self.device)
        next_values = torch.stack([it.V_next for it in buf]).to(self.device)

        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(buf))):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actions_raw = torch.stack([it.a_raw for it in buf]).to(self.device)
        old_logp = torch.stack([it.logp for it in buf]).to(self.device)
        datas = [it.data for it in buf]
        return actions_raw, old_logp, returns.detach(), adv.detach(), datas

    def update(self, buf: List[RolloutItem]):
        """The main PPO update loop."""
        # compute the advantage, returns, old log-probs, etc.
        actions_raw, old_logp, returns, adv, datas = self._gae(buf)
        T = actions_raw.size(0)
        idx = torch.randperm(T)

        # iterate over several update epochs
        for epoch in range(self.update_epochs):
            for start in range(0, T, self.mb):
                end = min(start + self.mb, T)
                b = idx[start:end]

                # === 1. compute all loss components ===
                alphas, values, _ = self._forward_batch([datas[i] for i in b])
                dist = Dirichlet(alphas)
                logp = dist.log_prob(actions_raw[b])
                ratio = torch.exp(logp - old_logp[b])

                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[b]
                policy_loss = -torch.mean(torch.min(surr1, surr2))  # the PPO objective
                value_loss = F.mse_loss(values, returns[b])  # the critic loss
                entropy = torch.mean(dist.entropy())  # entropy regularization

                # total RL loss
                rl_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # === 2. the MORL auxiliary loss ===
                if self.morl_enable:
                    batch_datas = [datas[i] for i in b]
                    preds_list, ys_list = [], []
                    for d in batch_datas:
                        p = self.policy.predict_next_return(d)
                        y = d.y.to(p.device)
                        preds_list.append(p)
                        ys_list.append(y)

                    preds = torch.stack(preds_list, dim=0)
                    ys = torch.stack(ys_list, dim=0)

                    mse = torch.mean((preds - ys) ** 2)
                    s = _compute_signal(preds, risks=None)
                    ic_per = _pearson_corr(s, ys, dim=1)
                    ic = ic_per.mean()

                    pred_loss = self.morl_alpha * mse + self.morl_beta * (-ic)

                    # dynamically adjust the weights
                    if self.adaptive_weights and epoch == 0:  # adjust once per outer epoch
                        self._update_adaptive_weights(rl_loss.item(), pred_loss.item())

                    # === 3. gradient surgery (resolves conflicts) ===
                    if self.use_grad_surgery:
                        # compute each task's gradient separately
                        self.opt.zero_grad()
                        rl_loss.backward(retain_graph=True)
                        rl_grads = {name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                    for name, p in self.policy.named_parameters()}

                        self.opt.zero_grad()
                        pred_loss.backward(retain_graph=True)
                        pred_grads = {name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                      for name, p in self.policy.named_parameters()}

                        # project away the conflicting gradients
                        merged_grads = self._project_conflicting_gradients([rl_grads, pred_grads])

                        # apply the merged gradient (weighted)
                        self.opt.zero_grad()
                        for name, p in self.policy.named_parameters():
                            if name in merged_grads:
                                # RL gradient + the weighted prediction gradient
                                p.grad = rl_grads[name] + self.morl_aux_weight * merged_grads[name]
                    else:
                        # simple weighted sum
                        loss = rl_loss + self.morl_aux_weight * pred_loss
                        self.opt.zero_grad()
                        loss.backward()
                else:
                    # RL loss only
                    self.opt.zero_grad()
                    rl_loss.backward()

                # gradient clipping, to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()

    def _forward_batch(self, data_batch: List[Data]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched forward pass through the policy network."""
        alphas, vals, logits = [], [], []
        for d in data_batch:
            a, v, l = self.policy(d)
            alphas.append(a)
            vals.append(v)
            logits.append(l)
        return torch.stack(alphas), torch.stack(vals), torch.stack(logits)
