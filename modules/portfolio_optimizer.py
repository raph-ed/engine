"""Robust Portfolio Optimization via Ledoit-Wolf Shrinkage & Dynamic Regime Allocation."""

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized Ledoit-Wolf shrinkage covariance matrix."""
    clean_rets = returns.dropna()
    lw = LedoitWolf()
    lw.fit(clean_rets)
    cov_annual = lw.covariance_ * 252.0
    return pd.DataFrame(cov_annual, index=clean_rets.columns, columns=clean_rets.columns)


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "max_sharpe",
    max_weight: float = 0.35,
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    """SLSQP constrained portfolio optimizer with concentration ceiling."""
    assets = list(returns.columns)
    n = len(assets)
    if n == 0:
        return {}

    cov_matrix = ledoit_wolf_covariance(returns)
    mean_returns = returns.mean() * 252.0

    init_weights = np.ones(n) / n
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def objective(w):
        port_ret = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        if method == "min_vol":
            return port_vol
        # Max Sharpe -> minimize negative Sharpe
        excess_ret = port_ret - risk_free_rate
        return -excess_ret / max(port_vol, 1e-6)

    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200},
    )

    if res.success:
        weights = res.x
    else:
        weights = init_weights

    # Clean small floating values and renormalize
    weights = np.maximum(weights, 0.0)
    weights /= np.sum(weights)

    return dict(zip(assets, np.round(weights, 4)))


def compute_dynamic_regime_allocation(
    regime: int,
    base_weights: Dict[str, float],
    defensive_assets: List[str] = None,
) -> Dict[str, float]:
    """Adapt asset weights conditionally to detected market regime."""
    if defensive_assets is None:
        defensive_assets = ["GLD", "TLT", "SHY"]

    growth_assets = [k for k in base_weights if k not in defensive_assets]
    weights = base_weights.copy()

    # Rule matrix:
    # Regime 0 (Bullish): 100% Growth, 0% Defensive
    # Regime 1 (Neutral): 70% Growth, 30% Defensive
    # Regime 2 (Sell-Off): 15% Growth, 85% Defensive
    if regime == 0:
        target_growth, target_defensive = 1.00, 0.00
    elif regime == 1:
        target_growth, target_defensive = 0.70, 0.30
    else:  # Regime 2
        target_growth, target_defensive = 0.15, 0.85

    growth_sum = sum(weights.get(a, 0.0) for a in growth_assets)
    def_sum = sum(weights.get(a, 0.0) for a in defensive_assets)

    new_weights = {}
    for a in growth_assets:
        prop = weights.get(a, 0.0) / max(growth_sum, 1e-6)
        new_weights[a] = prop * target_growth

    for a in defensive_assets:
        prop = weights.get(a, 0.0) / max(def_sum, 1e-6) if def_sum > 0 else (1.0 / len(defensive_assets))
        new_weights[a] = prop * target_defensive

    # Renormalize to ensure exact sum of 1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    return new_weights
