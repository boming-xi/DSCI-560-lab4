from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    history: pd.DataFrame
    metrics: Dict[str, float]


def backtest(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
) -> BacktestResult:
    out = df.copy()
    out["market_return"] = out["close"].pct_change().fillna(0.0)

    # Shift position by one step so today's signal affects tomorrow's return.
    held_position = out["position"].shift(1).fillna(0.0)
    trades = out["position"].diff().abs().fillna(out["position"].abs())

    out["strategy_return"] = (out["market_return"] * held_position) - (trades * fee_rate)
    out["equity"] = initial_cash * (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = initial_cash * (1 + out["market_return"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0

    n = len(out)
    years = n / 252 if n > 0 else np.nan

    total_return = (out["equity"].iloc[-1] / initial_cash) - 1 if n > 0 else np.nan
    buy_hold_return = (out["buy_hold_equity"].iloc[-1] / initial_cash) - 1 if n > 0 else np.nan
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years and years > 0 else np.nan
    annualized_vol = out["strategy_return"].std(ddof=0) * np.sqrt(252) if n > 1 else np.nan
    sharpe = annualized_return / annualized_vol if annualized_vol and annualized_vol > 0 else np.nan
    max_drawdown = out["drawdown"].min() if n > 0 else np.nan
    trades_count = int(trades.sum())

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "buy_hold_return": float(buy_hold_return),
        "trades_count": float(trades_count),
    }
    return BacktestResult(history=out, metrics=metrics)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if aligned.empty:
        return {"mae": np.nan, "rmse": np.nan}

    errors = aligned["y_true"] - aligned["y_pred"]
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    return {"mae": float(mae), "rmse": float(rmse)}
