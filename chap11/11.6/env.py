"""A Gym-like environment wrapper around the backtest engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from torch_geometric.data import Data

from backtest import EnhancedBacktestEngine
from policy import PolicyNet


class PortfolioEnv:
    """Wraps EnhancedBacktestEngine as an RL environment.
    - State: the current graph sample Data + the previous weights (extensible into policy features)
    - Action: the target weights (or a Dirichlet sample of them)
    - Reward: log return - lambda * turnover (extensible with more risk terms)
    """
    def __init__(self, policy: PolicyNet, dataset: List[Data],
                 price_df, engine_cfg: Dict, reward_cfg: Dict):
        self.policy = policy  # the policy (encoder + Actor + Critic); used here mainly to generate actions/values
        self.dataset = dataset  # the time-ordered sequence of graph samples (one Data per step)
        self.price_df = price_df  # the corresponding price matrix (DataFrame, rows=dates, columns=assets)
        self.engine = EnhancedBacktestEngine(**engine_cfg)  # instantiate the backtest engine with the cost/constraint config
        self.reward_cfg = reward_cfg  # reward config (e.g. the turnover penalty coefficient lambda_turnover)
        self.reset()  # initialize the environment state (t, cash/positions, the previous portfolio value, etc.)

    def reset(self):
        self.engine.reset()  # clear the backtest engine's positions, cash, NAV series, etc.
        self.t = 0  # reset the current time-step index to the first sample
        self.last_value = self.engine.initial_capital  # the previous portfolio value, set to the initial capital (a baseline for the first step's return)
        return self.dataset[self.t]  # return the first state (the current graph sample Data)

    def step(self, action_weights: np.ndarray) -> Tuple[Optional[Data], float, bool, Dict]:
        date = self.price_df.index[-len(self.dataset):][self.t]  # the t-th trading date, aligned with the dataset
        prices = self.price_df.loc[date].values  # today's price vector for all assets

        # the current portfolio value
        if self.t == 0:
            current_value = self.engine.initial_capital  # no position yet on the first step; portfolio value = initial capital
        else:
            current_shares = np.array([self.engine.positions.get(j, 0.0)  # read the current share count per asset
                                       for j in range(len(prices))])
            holdings_value = float(np.sum(current_shares * prices))  # the market value of the current holdings
            current_value = self.engine.cash + holdings_value  # total assets = cash + holdings value

        _, trade_details = self.engine.execute_trades(  # place orders per the target weights, updating positions/cash
            action_weights, prices, current_value)

        # settle
        pos_shares = np.array([self.engine.positions.get(j, 0.0)  # share count after the trade
                               for j in range(len(prices))])
        V_t = self.engine.cash + float(np.sum(pos_shares * prices))  # today's portfolio value (cash + the latest holdings value)

        # reward: log return - lambda * turnover
        r = float(np.log(max(V_t, 1e-8)) - np.log(max(self.last_value, 1e-8)))  # log return (more numerically stable and additive)
        r -= float(self.reward_cfg.get('lambda_turnover', 0.0) *  # subtract the turnover penalty (the more frequent, the bigger the penalty)
                   trade_details.get('turnover_rate', 0.0))

        self.last_value = V_t  # record this period's portfolio value, as the baseline for next period's return
        self.t += 1            # advance the time step
        done = (self.t >= len(self.dataset))  # terminal condition: the whole sample sequence has been consumed
        next_state = None if done else self.dataset[self.t]  # if not done, return the next state (a graph sample)
        info = {"V": V_t, "turnover": trade_details.get('turnover_rate', 0.0)}  # extra info: current portfolio value, current turnover rate (for monitoring/tuning)
        return next_state, r, done, info  # returns (next state, reward, done, diagnostic info)
