"""The backtest engine (cash + position ledger, mark-to-market from T to T+1)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from model import HierarchicalGNN
from portfolio import EnhancedPortfolioOptimizer


class BacktestEngine:
    """Backtest engine (cash + position ledger, mark-to-market from T to T+1)."""

    def __init__(self, initial_capital: float = 1_000_000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.portfolio_value = []
        self.returns = []
        self.benchmark_returns = []
        self.turnovers = []
        self.trades = []
        self.positions = None      # np.ndarray[ n_assets ]
        self.cash = self.initial_capital

    def run_backtest(self, model: HierarchicalGNN, test_dataset: List[Data],
                     price_data: pd.DataFrame, benchmark_data: pd.Series,
                     optimizer: EnhancedPortfolioOptimizer,
                     optimization_method: str = 'softmax_long_only') -> Dict:

        model.eval()
        dates = price_data.index
        n_assets = price_data.shape[1]

        # Initialize
        self.positions = np.zeros(n_assets, dtype=float)
        self.cash = self.initial_capital
        self.portfolio_value = [self.initial_capital]

        # Align the benchmark
        bench = benchmark_data.reindex(dates).ffill().bfill()
        bench_ret = bench.pct_change().fillna(0).values

        with torch.no_grad():
            # Iterate up to the second-to-last day: rebalance at day i's close, hold to day i+1's close
            for i in range(len(test_dataset) - 1):
                date_t   = dates[i]
                date_tp1 = dates[i + 1]

                # 1) predict using sample i, to decide the target weights at t's close
                out = model([test_dataset[i]])
                exp_ret = np.nan_to_num(out['predictions'][0].cpu().numpy(), nan=0.0)
                pred_rk = np.nan_to_num(out['risks'][0].cpu().numpy(), nan=0.01)
                pred_rk = np.maximum(pred_rk, 1e-3)

                # 2) rebalance at t's close price and deduct fees
                px_t = price_data.loc[date_t].values
                equity_t = self.cash + float(np.dot(self.positions, px_t))

                if optimization_method == 'mean_variance':
                    cov = np.diag(pred_rk ** 2)
                    w = optimizer.mean_variance_optimization(exp_ret, cov, constraints={'long_only': True, 'max_weight': 0.1})
                elif optimization_method == 'risk_parity':
                    cov = np.diag(pred_rk ** 2)
                    w = optimizer.risk_parity_optimization(cov, constraints={'max_weight': 0.2})
                elif optimization_method == 'min_variance':
                    cov = np.diag(pred_rk ** 2)
                    w = optimizer.min_variance_optimization(cov, constraints={'max_weight': 0.25})
                else:
                    w = optimizer.softmax_long_only_optimization(exp_ret, pred_rk, temperature=2.0)

                target_value = w * equity_t
                target_shares = target_value / np.maximum(px_t, 1e-8)

                delta_shares = target_shares - self.positions
                turnover = float(np.sum(np.abs(delta_shares) * px_t))
                fee = turnover * self.transaction_cost

                # Update cash and positions (buy spend, sell proceeds, fee deducted)
                cash_flow = - float(np.dot(delta_shares, px_t)) - fee
                self.cash += cash_flow
                self.positions = target_shares
                self.turnovers.append(turnover)
                self.trades.append(np.count_nonzero(delta_shares))

                # 3) hold to t+1's close, mark-to-market to get portfolio value and the day's return
                px_tp1 = price_data.loc[date_tp1].values
                value_tp1 = self.cash + float(np.dot(self.positions, px_tp1))

                ret = (value_tp1 - self.portfolio_value[-1]) / self.portfolio_value[-1]
                self.portfolio_value.append(value_tp1)
                self.returns.append(ret)
                self.benchmark_returns.append(bench_ret[i + 1])

        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self) -> Dict:
        if len(self.returns) == 0:
            return {
                'total_return': 0.0, 'annual_return': 0.0, 'annual_volatility': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
                'final_portfolio_value': self.portfolio_value[-1] if self.portfolio_value else self.initial_capital,
                'total_trades': int(np.sum(self.trades)) if len(self.trades) else 0,
                'avg_turnover': 0.0
            }

        r = np.asarray(self.returns, dtype=float)
        total_return = self.portfolio_value[-1] / self.portfolio_value[0] - 1.0
        annual_return = (1 + total_return) ** (252 / len(r)) - 1
        annual_vol = np.std(r) * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        cum = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = float(np.min(dd)) if len(dd) else 0.0

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(np.mean(r > 0)),
            'final_portfolio_value': float(self.portfolio_value[-1]),
            'total_trades': int(np.sum(self.trades)),
            'avg_turnover': float(np.mean(self.turnovers)) if self.turnovers else 0.0,
        }
