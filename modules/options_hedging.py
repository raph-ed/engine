"""Black-Scholes Options Valuation Engine, Dynamic Greeks & Hedging Simulator."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm


class BlackScholesEngine:
    """Analytical closed-form Black-Scholes (1973) options pricer and Greeks calculator."""

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        T = max(T, 1e-4)
        sigma = max(sigma, 1e-4)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @classmethod
    def compute_greeks(
        cls, S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "put"
    ) -> Dict[str, float]:
        """Compute Delta, Gamma, Theta ($/calendar day) and Vega (per 1% vol shift)."""
        T = max(T, 1e-4)
        sigma = max(sigma, 1e-4)
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        pdf_d1 = norm.pdf(d1)

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01  # Per 1% implied vol shift

        if opt_type == "put":
            delta = norm.cdf(d1) - 1.0
            theta_annual = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta_daily = theta_annual / 365.0
        else:
            delta = norm.cdf(d1)
            theta_annual = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            theta_daily = theta_annual / 365.0

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta_daily": float(theta_daily),
            "vega_1pct": float(vega),
        }


def simulate_options_hedged_portfolio(
    spot_prices: pd.Series,
    vix_series: pd.Series,
    regime_series: pd.Series,
    hedge_otm_pct: float = 0.05,
    put_duration_days: int = 30,
    risk_free_rate: float = 0.03,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate systematic 5% OTM Put buying when Regime 2 (Sell-Off) is active."""
    aligned_idx = spot_prices.index.intersection(regime_series.index)
    S = spot_prices.loc[aligned_idx]
    regimes = regime_series.loc[aligned_idx]
    vix = vix_series.loc[aligned_idx] if vix_series is not None else pd.Series(20.0, index=aligned_idx)

    n = len(aligned_idx)
    unhedged_wealth = 100.0 * (S / S.iloc[0])
    hedged_wealth = np.zeros(n)
    hedged_wealth[0] = 100.0

    in_hedge = False
    put_strike = 0.0
    days_held = 0
    put_entry_price = 0.0
    active_hedge_pnl = pd.Series(0.0, index=aligned_idx)

    for i in range(1, n):
        r_asset = (S.iloc[i] - S.iloc[i - 1]) / S.iloc[i - 1]
        sigma = max(vix.iloc[i] / 100.0, 0.10)
        curr_regime = regimes.iloc[i]

        hedge_pnl = 0.0

        # Trigger or maintain hedge in Regime 2
        if curr_regime == 2:
            if not in_hedge or days_held >= put_duration_days:
                # Buy fresh Put OTM 5%
                in_hedge = True
                days_held = 0
                put_strike = S.iloc[i - 1] * (1.0 - hedge_otm_pct)
                T_init = put_duration_days / 365.0
                put_entry_price = BlackScholesEngine.put_price(
                    S.iloc[i - 1], put_strike, T_init, risk_free_rate, sigma
                )
                # Pay premium
                hedge_pnl -= put_entry_price / S.iloc[i - 1]

            days_held += 1
            T_curr = max((put_duration_days - days_held) / 365.0, 1e-4)
            current_put_val = BlackScholesEngine.put_price(
                S.iloc[i], put_strike, T_curr, risk_free_rate, sigma
            )
            # Daily delta in option value
            hedge_pnl += (current_put_val - put_entry_price) / S.iloc[i - 1]
            put_entry_price = current_put_val
        else:
            in_hedge = False
            days_held = 0

        active_hedge_pnl.iloc[i] = hedge_pnl
        hedged_wealth[i] = hedged_wealth[i - 1] * (1.0 + r_asset + hedge_pnl)

    return (
        unhedged_wealth,
        pd.Series(hedged_wealth, index=aligned_idx),
        active_hedge_pnl,
    )
