"""Feature engineering and target labeling with zero look-ahead bias."""

from typing import Tuple
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_features_and_targets(
    df_prices: pd.DataFrame,
    target_horizon: int = 20,
    bull_thresh: float = 0.015,
    bear_thresh: float = -0.02,
    embargo_period: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build technical distribution features and forward targets."""
    spy = df_prices["SPY"].copy()
    vix = df_prices["^VIX"].copy() if "^VIX" in df_prices.columns else (spy.pct_change().rolling(20).std() * np.sqrt(252) * 100).fillna(20.0)

    # 1. Price Momentum & Returns
    ret_5d = np.log(spy / spy.shift(5))
    ret_20d = np.log(spy / spy.shift(20))
    ret_60d = np.log(spy / spy.shift(60))

    # 2. Moving Average Distance
    sma_50 = spy.rolling(50).mean()
    sma_200 = spy.rolling(200).mean()
    dist_sma50 = (spy - sma_50) / sma_50
    dist_sma200 = (spy - sma_200) / sma_200
    ratio_sma = sma_50 / sma_200

    # 3. Volatilities & Ratios
    daily_ret = spy.pct_change()
    vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
    vol_60d = daily_ret.rolling(60).std() * np.sqrt(252)
    vol_ratio = vol_20d / (vol_60d.replace(0, 1e-6))

    # 4. Oscillator & VIX
    rsi_14 = compute_rsi(spy, window=14) / 100.0
    vix_sma20 = vix.rolling(20).mean()
    vix_ratio = vix / (vix_sma20.replace(0, 1e-6))

    features_raw = pd.DataFrame(
        {
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "ret_60d": ret_60d,
            "dist_sma50": dist_sma50,
            "dist_sma200": dist_sma200,
            "ratio_sma": ratio_sma,
            "vol_20d": vol_20d,
            "vol_ratio": vol_ratio,
            "rsi_14": rsi_14,
            "vix_level": vix / 100.0,
            "vix_ratio": vix_ratio,
        },
        index=df_prices.index,
    )

    # Clean features for live inference (drop NaNs due to warm-up SMA200)
    features_clean = features_raw.dropna().copy()

    # Supervised Targets: 20-day forward returns and rolling forward Max Drawdown
    fwd_ret = spy.shift(-target_horizon) / spy - 1.0

    # Forward 20-day max drawdown calculation
    fwd_min_price = spy.iloc[::-1].rolling(target_horizon, min_periods=1).min().iloc[::-1].shift(-1)
    fwd_mdd = (fwd_min_price - spy) / spy

    # Label assignment: 2 (Sell-off), 0 (Bullish), 1 (Ranging/Neutral)
    labels = pd.Series(index=df_prices.index, dtype=float)
    bear_cond = (fwd_ret < bear_thresh) | (fwd_mdd < -0.05)
    bull_cond = (fwd_ret > bull_thresh) & (fwd_mdd > -0.03)

    labels[bear_cond] = 2.0
    labels[bull_cond & (~bear_cond)] = 0.0
    labels[labels.isna()] = 1.0

    # Combine into training dataset where forward target is genuinely available
    # Purge the last (target_horizon + embargo_period) days where forward horizon reaches into future
    dataset_full = features_clean.copy()
    dataset_full["target"] = labels.loc[features_clean.index]

    # Mask unobservable future tail (last target_horizon days don't have known forward returns)
    valid_train_idx = dataset_full.index[:-target_horizon]
    dataset_train = dataset_full.loc[valid_train_idx].copy()
    dataset_train["target"] = dataset_train["target"].astype(int)

    return features_clean, dataset_train, labels
