"""Data Ingestion Module with 400-day warm-up buffer and stochastic fallback."""

import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS: List[str] = [
    "SPY",  # US Equities Large Cap
    "QQQ",  # US Equities Tech
    "EZU",  # Eurozone Equities
    "VGK",  # European Equities
    "GLD",  # Gold Defensive
    "TLT",  # US 20+ Year Treasury Bonds
    "SHY",  # US 1-3 Year Short Treasury Cash Proxy
]
VIX_TICKER = "^VIX"


def generate_synthetic_data(
    tickers: List[str], start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """Generate realistic 3-state Markov switching asset prices."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)
    if n_days == 0:
        dates = pd.date_range(end=datetime.date.today(), periods=300, freq="B")
        n_days = len(dates)

    np.random.seed(42)
    # 0 = Bull, 1 = Neutral, 2 = Bear
    p_trans = np.array([
        [0.96, 0.03, 0.01],
        [0.05, 0.90, 0.05],
        [0.02, 0.08, 0.90],
    ])
    regimes = np.zeros(n_days, dtype=int)
    curr_state = 0
    for t in range(n_days):
        regimes[t] = curr_state
        curr_state = np.random.choice([0, 1, 2], p=p_trans[curr_state])

    regime_mus = {0: 0.14 / 252, 1: 0.03 / 252, 2: -0.28 / 252}
    regime_vols = {0: 0.12 / np.sqrt(252), 1: 0.16 / np.sqrt(252), 2: 0.35 / np.sqrt(252)}

    all_cols = tickers.copy()
    if VIX_TICKER not in all_cols:
        all_cols.append(VIX_TICKER)

    data = {}
    base_mu = np.array([regime_mus[s] for s in regimes])
    base_vol = np.array([regime_vols[s] for s in regimes])

    for ticker in all_cols:
        if ticker == VIX_TICKER:
            vix_base = np.where(regimes == 0, 14.0, np.where(regimes == 1, 18.5, 34.0))
            noise = np.random.normal(0, 1.5, n_days)
            data[ticker] = np.clip(vix_base + noise, 10.0, 75.0)
        else:
            asset_beta = 1.0 if ticker in ["SPY", "QQQ"] else (0.8 if ticker in ["EZU", "VGK"] else (-0.2 if ticker in ["TLT", "GLD"] else 0.02))
            asset_ret = base_mu * asset_beta + np.random.normal(0, 1.0, n_days) * base_vol
            init_price = 100.0 if ticker not in ["SPY", "QQQ"] else 400.0
            price_series = init_price * np.exp(np.cumsum(asset_ret))
            data[ticker] = price_series

    df = pd.DataFrame(data, index=dates)
    return df


def load_multi_asset_data(
    tickers: List[str],
    start_date: datetime.date,
    end_date: datetime.date = None,
    force_demo: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """Load asset universe with a 400 calendar day buffer to avoid indicator cold-starts."""
    if end_date is None:
        end_date = datetime.date.today()

    warmup_start = start_date - datetime.timedelta(days=400)
    fetch_end = end_date + datetime.timedelta(days=1)

    all_symbols = list(set(tickers + [VIX_TICKER]))
    is_fallback = False

    if force_demo:
        df_prices = generate_synthetic_data(all_symbols, warmup_start, fetch_end)
        return df_prices, True

    try:
        data = yf.download(
            tickers=all_symbols,
            start=warmup_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty or "Close" not in data:
            raise ValueError("Empty data returned by yfinance")

        if isinstance(data.columns, pd.MultiIndex):
            df_prices = data["Close"].copy()
        else:
            df_prices = data.copy()

        # Clean missing values
        df_prices = df_prices.ffill().bfill()

        # Check that we have enough records and key ticker SPY exists
        if len(df_prices) < 50 or "SPY" not in df_prices.columns:
            raise ValueError("Insufficient data records or SPY missing")

        # Fill any missing tickers with synthetic proxies if necessary
        for sym in all_symbols:
            if sym not in df_prices.columns:
                df_prices[sym] = df_prices["SPY"] * 0.5

    except Exception:
        is_fallback = True
        df_prices = generate_synthetic_data(all_symbols, warmup_start, fetch_end)

    return df_prices, is_fallback
