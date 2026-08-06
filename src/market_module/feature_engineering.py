"""Engineered temporal market features: rolling, momentum, and volume.

All three functions operate on a single `card_name` + `grade` group and use
strictly past data only — for a transaction at time t, only transactions
with `date_sold < t` are visible. This is what the market-state encoder's
robustness under train/test drift depends on: without this guarantee, the
rolling and momentum features would leak future price information into the
target.
"""

from typing import List

import numpy as np
import pandas as pd


def compute_rolling_features(group: pd.DataFrame, windows: List[int] = [7, 14, 30], min_obs: int = 2) -> pd.DataFrame:
    """Compute rolling price statistics for a single card_name+grade group.

    Uses strictly past data: for the transaction at time t, only uses
    transactions with `date_sold < t` (not `<=`, excluding the current row).
    """
    group = group.sort_values("date_sold").copy()

    for w in windows:
        roll_mean, roll_median, roll_std, roll_count = [], [], [], []

        for _, row in group.iterrows():
            current_date = row["date_sold"]
            window_start = current_date - pd.Timedelta(days=w)

            past_mask = (group["date_sold"] < current_date) & (group["date_sold"] >= window_start)
            past_prices = group.loc[past_mask, "price"]

            if len(past_prices) >= min_obs:
                roll_mean.append(past_prices.mean())
                roll_median.append(past_prices.median())
                roll_std.append(past_prices.std() if len(past_prices) > 1 else 0.0)
                roll_count.append(len(past_prices))
            else:
                roll_mean.append(np.nan)
                roll_median.append(np.nan)
                roll_std.append(np.nan)
                roll_count.append(len(past_prices))

        group[f"roll_mean_{w}d"] = roll_mean
        group[f"roll_median_{w}d"] = roll_median
        group[f"roll_std_{w}d"] = roll_std
        group[f"roll_count_{w}d"] = roll_count

    return group


def compute_momentum_features(group: pd.DataFrame, windows: List[int] = [7, 14]) -> pd.DataFrame:
    """Compute momentum: the change in a card's rolling average price."""
    group = group.sort_values("date_sold").copy()

    for w in windows:
        momentum_abs, momentum_pct = [], []

        for _, row in group.iterrows():
            current_date = row["date_sold"]

            curr_start = current_date - pd.Timedelta(days=w)
            curr_mask = (group["date_sold"] < current_date) & (group["date_sold"] >= curr_start)
            curr_prices = group.loc[curr_mask, "price"]

            prev_start = current_date - pd.Timedelta(days=2 * w)
            prev_mask = (group["date_sold"] < curr_start) & (group["date_sold"] >= prev_start)
            prev_prices = group.loc[prev_mask, "price"]

            if len(curr_prices) >= 2 and len(prev_prices) >= 2:
                curr_mean = curr_prices.mean()
                prev_mean = prev_prices.mean()
                momentum_abs.append(curr_mean - prev_mean)
                momentum_pct.append((curr_mean - prev_mean) / prev_mean * 100 if prev_mean > 0 else 0)
            else:
                momentum_abs.append(np.nan)
                momentum_pct.append(np.nan)

        group[f"momentum_abs_{w}d"] = momentum_abs
        group[f"momentum_pct_{w}d"] = momentum_pct

    return group


def compute_volume_features(group: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Count past transactions in each window for a card_name+grade group."""
    group = group.sort_values("date_sold").copy()

    for w in windows:
        volumes = []
        for _, row in group.iterrows():
            current_date = row["date_sold"]
            window_start = current_date - pd.Timedelta(days=w)
            past_mask = (group["date_sold"] < current_date) & (group["date_sold"] >= window_start)
            volumes.append(past_mask.sum())
        group[f"card_volume_{w}d"] = volumes

    return group


def compute_market_wide_volume(df: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Add market-wide (all-card) transaction volume columns to `df`."""
    df = df.copy()
    for w in windows:
        market_vol = []
        for _, row in df.iterrows():
            current_date = row["date_sold"]
            window_start = current_date - pd.Timedelta(days=w)
            past_mask = (df["date_sold"] < current_date) & (df["date_sold"] >= window_start)
            market_vol.append(past_mask.sum())
        df[f"market_volume_{w}d"] = market_vol
    return df
