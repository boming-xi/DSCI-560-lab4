from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    rsi_period: int = 14,
) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(window=fast_window, min_periods=fast_window).mean()
    out["sma_slow"] = out["close"].rolling(window=slow_window, min_periods=slow_window).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    return out


def generate_signals(
    df: pd.DataFrame,
    rsi_buy: float = 40.0,
    rsi_sell: float = 70.0,
) -> pd.DataFrame:
    out = df.copy()

    up_cross = (out["sma_fast"] > out["sma_slow"]) & (
        out["sma_fast"].shift(1) <= out["sma_slow"].shift(1)
    )
    down_cross = (out["sma_fast"] < out["sma_slow"]) & (
        out["sma_fast"].shift(1) >= out["sma_slow"].shift(1)
    )

    trend_up = out["sma_fast"] > out["sma_slow"]                     # <-- NEW
    rsi_confirm = (out["rsi"] > rsi_buy) & (out["rsi"] < rsi_sell)   # <-- NEW

    out["buy_signal"] = up_cross | (trend_up & rsi_confirm)          # <-- CHANGED
    out["sell_signal"] = down_cross | (out["rsi"] > rsi_sell)

    position = []
    current_position = 0
    for is_buy, is_sell in zip(out["buy_signal"], out["sell_signal"]):
        if bool(is_buy):
            current_position = 1
        elif bool(is_sell):
            current_position = 0
        position.append(current_position)

    out["position"] = position
    return out