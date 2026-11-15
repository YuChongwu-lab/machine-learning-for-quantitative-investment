"""Portfolio-construction methods used to turn model outputs into target weights.
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from scipy.optimize import minimize


class EnhancedPortfolioOptimizer:
    """An enhanced portfolio optimizer supporting several optimization methods."""

    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion

    def mean_variance_optimization(self, expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   constraints: Optional[Dict] = None) -> np.ndarray:
        """Mean-variance (Markowitz) optimization."""
        try:
            n_assets = len(expected_returns)

            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.dot(weights, np.dot(cov_matrix, weights))
                return -(portfolio_return - self.risk_aversion * 0.5 * portfolio_risk)

            constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            if constraints and constraints.get('long_only', True):
                bounds = [(0, constraints.get('max_weight', 0.2)) for _ in range(n_assets)]
            else:
                bounds = [(-0.2, 0.2) for _ in range(n_assets)]

            x0 = np.ones(n_assets) / n_assets
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

            return result.x if result.success else x0

        except Exception:
            return np.ones(len(expected_returns)) / len(expected_returns)

    def risk_parity_optimization(self, cov_matrix: np.ndarray,
                                 constraints: Optional[Dict] = None) -> np.ndarray:
        """Risk-parity optimization."""
        try:
            n_assets = cov_matrix.shape[0]

            def risk_budget_objective(weights):
                # Risk contributions
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib

                # Risk parity: every asset should contribute equal risk
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)

            constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.01, constraints.get('max_weight', 0.2) if constraints else 0.5)
                      for _ in range(n_assets)]

            x0 = np.ones(n_assets) / n_assets
            result = minimize(risk_budget_objective, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints_list)

            return result.x if result.success else x0

        except Exception:
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]

    def softmax_long_only_optimization(self, expected_returns: np.ndarray,
                                       predicted_risks: np.ndarray,
                                       temperature: float = 2.0) -> np.ndarray:
        """Softmax-based long-only optimization (fits the GNN's output more naturally)."""
        try:
            # Risk-adjusted returns
            risk_adjusted_returns = expected_returns / (predicted_risks + 1e-8)

            # Softmax weights
            exp_scores = np.exp(risk_adjusted_returns / temperature)
            weights = exp_scores / np.sum(exp_scores)

            # Keep weights within a sane range
            max_weight = 0.2
            weights = np.minimum(weights, max_weight)
            weights = weights / np.sum(weights)  # renormalize

            return weights

        except Exception:
            n_assets = len(expected_returns)
            return np.ones(n_assets) / n_assets

    def min_variance_optimization(self, cov_matrix: np.ndarray,
                                  constraints: Optional[Dict] = None) -> np.ndarray:
        """Minimum-variance optimization."""
        try:
            n_assets = cov_matrix.shape[0]

            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))

            constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, constraints.get('max_weight', 0.2) if constraints else 0.3)
                      for _ in range(n_assets)]

            x0 = np.ones(n_assets) / n_assets
            result = minimize(objective, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints_list)

            return result.x if result.success else x0

        except Exception:
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
