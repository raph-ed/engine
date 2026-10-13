"""Institutional Backtester with Turnovers, 15 bps Frictions & Performance Analytics."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from .portfolio_optimizer import compute_dynamic_regime_allocation, optimize_portfolio


def compute_financial_metrics(series: pd.Series, risk_free_rate: float = 0.03) -> Dict[str, float]:
    """Compute institutional hedge fund metrics."""
    daily_rets = series.pct_change().dropna()
    if len(daily_rets) < 10:
        return {
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }

    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    n_years = max(len(series) / 252.0, 1e-4)
    cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    vol = daily_rets.std() * np.sqrt(252.0)

    excess_ret = cagr - risk_free_rate
    sharpe = excess_ret / max(vol, 1e-6)

    downside_rets = daily_rets[daily_rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(252.0)
    sortino = excess_ret / max(downside_vol, 1e-6)

    cum_max = series.cummax()
    drawdowns = (series - cum_max) / cum_max
    max_dd = drawdowns.min()

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

    return {
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def run_strategy_backtest(
    df_prices: pd.DataFrame,
    regime_series: pd.Series,
    base_assets: list = None,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_weight: float = 0.35,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], int]:
    """Execute dynamic regime portfolio backtest versus SPY and 60/40 benchmarks."""
    if base_assets is None:
        base_assets = [c for c in df_prices.columns if c != "^VIX"]

    aligned_idx = df_prices.index.intersection(regime_series.index)
    prices = df_prices.loc[aligned_idx, base_assets].ffill()
    regimes = regime_series.loc[aligned_idx]

    returns = prices.pct_change().dropna()
    aligned_idx = returns.index
    regimes = regimes.loc[aligned_idx]

    # 1. Base Markowitz Optimal Weights
    static_weights = optimize_portfolio(returns, method="max_sharpe", max_weight=max_weight)

    # 2. 60/40 Benchmark (SPY 60% / TLT 40% or equal balanced)
    w_6040 = {}
    for a in base_assets:
        if a == "SPY":
            w_6040[a] = 0.60
        elif a in ["TLT", "SHY"]:
            w_6040[a] = 0.40
        else:
            w_6040[a] = 0.0
    if sum(w_6040.values()) == 0:
        w_6040 = {a: 1.0 / len(base_assets) for a in base_assets}

    # Simulate Walk
    strat_wealth = [100.0]
    bench_6040_wealth = [100.0]
    bench_spy_wealth = [100.0]

    current_weights = static_weights.copy()
    total_rebalances = 0
    total_friction_rate = (transaction_cost_bps + slippage_bps) / 10000.0

    prev_regime = regimes.iloc[0]

    for i in range(len(aligned_idx)):
        date = aligned_idx[i]
        curr_regime = regimes.iloc[i]
        daily_r = returns.loc[date]

        # Check regime rotation
        if curr_regime != prev_regime or i == 0:
            target_weights = compute_dynamic_regime_allocation(curr_regime, static_weights)
            # Turnover computation
            turnover = sum(abs(target_weights.get(a, 0.0) - current_weights.get(a, 0.0)) for a in base_assets)
            friction_drag = turnover * total_friction_rate
            total_rebalances += 1
            current_weights = target_weights
            prev_regime = curr_regime
        else:
            friction_drag = 0.0

        port_r = sum(current_weights.get(a, 0.0) * daily_r.get(a, 0.0) for a in base_assets) - friction_drag
        b6040_r = sum(w_6040.get(a, 0.0) * daily_r.get(a, 0.0) for a in base_assets)
        spy_r = daily_r.get("SPY", 0.0)

        strat_wealth.append(strat_wealth[-1] * (1.0 + port_r))
        bench_6040_wealth.append(bench_6040_wealth[-1] * (1.0 + b6040_r))
        bench_spy_wealth.append(bench_spy_wealth[-1] * (1.0 + spy_r))

    # Construct Wealth DataFrame
    wealth_df = pd.DataFrame(
        {
            "Strategy": strat_wealth[1:],
            "Benchmark_6040": bench_6040_wealth[1:],
            "Benchmark_SPY": bench_spy_wealth[1:],
        },
        index=aligned_idx,
    )

    metrics = {
        "Strategy": compute_financial_metrics(wealth_df["Strategy"]),
        "Benchmark_6040": compute_financial_metrics(wealth_df["Benchmark_6040"]),
        "Benchmark_SPY": compute_financial_metrics(wealth_df["Benchmark_SPY"]),
    }

    return wealth_df, metrics, total_rebalances
