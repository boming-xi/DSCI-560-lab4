from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class RegimeParams:
    regime: str
    fast_window: int
    slow_window: int
    stop_loss_pct: float


def calculate_atr(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.Series:

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def detect_regime(
    df: pd.DataFrame,
    atr_window: int = 14,
    high_vol_threshold: float = 0.03,
    low_vol_threshold: float = 0.015,
) -> RegimeParams:

    atr = calculate_atr(df, window=atr_window)
    vol_pct = atr / df["close"]

    current_vol = vol_pct.iloc[-1]

    if current_vol >= high_vol_threshold:
        return RegimeParams(
            regime="HIGH",
            fast_window=5,
            slow_window=10,
            stop_loss_pct=0.05,
        )

    if current_vol <= low_vol_threshold:
        return RegimeParams(
            regime="LOW",
            fast_window=20,
            slow_window=50,
            stop_loss_pct=0.02,
        )

    return RegimeParams(
        regime="MEDIUM",
        fast_window=10,
        slow_window=30,
        stop_loss_pct=0.03,
    )