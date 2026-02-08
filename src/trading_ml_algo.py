from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1"] = out["close"].pct_change()
    out["return_5"] = out["close"].pct_change(5)
    out["return_10"] = out["close"].pct_change(10)
    out["volatility_10"] = out["return_1"].rolling(10).std()
    volatility_30 = out["return_1"].rolling(30).std()
    out["vol_ratio"] = out["volatility_10"] / volatility_30.replace(0, np.nan)
    out["sma_ratio"] = (out["sma_fast"] / out["sma_slow"]) - 1.0
    out["price_vs_sma_fast"] = (out["close"] / out["sma_fast"]) - 1.0
    out["sma_fast_slope"] = out["sma_fast"].pct_change(3)
    out["rsi_change_3"] = out["rsi"].diff(3)
    out["breakout_20"] = (out["close"] / out["close"].rolling(20).max()) - 1.0
    out["drawdown_10"] = (out["close"] / out["close"].rolling(10).max()) - 1.0

    atr_14 = _atr(out["high"], out["low"], out["close"], window=14)
    out["atr_14"] = atr_14
    out["atr_pct"] = atr_14 / out["close"]

    if "benchmark_close" in out.columns:
        bench_close = out["benchmark_close"].fillna(out["close"])
    else:
        bench_close = out["close"]
    bench_return_1 = bench_close.pct_change()
    bench_return_5 = bench_close.pct_change(5)
    bench_return_10 = bench_close.pct_change(10)

    out["rel_return_5"] = out["return_5"] - bench_return_5
    out["rel_return_10"] = out["return_10"] - bench_return_10

    rolling_cov = out["return_1"].rolling(60).cov(bench_return_1)
    rolling_var = bench_return_1.rolling(60).var()
    out["beta_60"] = rolling_cov / rolling_var.replace(0, np.nan)
    out["corr_60"] = out["return_1"].rolling(60).corr(bench_return_1)

    if "volume" not in out.columns:
        out["volume"] = np.nan
    if out["volume"].isna().all():
        out["volume"] = 1.0

    volume_ma_20 = out["volume"].rolling(20).mean()
    volume_std_20 = out["volume"].rolling(20).std()
    out["volume_ma_ratio"] = out["volume"] / volume_ma_20.replace(0, np.nan)
    out["volume_zscore_20"] = (out["volume"] - volume_ma_20) / volume_std_20.replace(0, np.nan)
    out["volume_zscore_20"] = out["volume_zscore_20"].fillna(0.0)
    return out


def generate_ml_signals(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    prob_threshold: float = 0.48,
    model_type: str = "rf",
    random_state: int = 42,
) -> pd.DataFrame:
    out = add_ml_features(df)
    target = (out["close"].shift(-1) > out["close"]).astype(int)

    feature_cols = [
        "return_1",
        "return_5",
        "return_10",
        "volatility_10",
        "atr_14",
        "atr_pct",
        "vol_ratio",
        "sma_ratio",
        "price_vs_sma_fast",
        "sma_fast_slope",
        "rsi",
        "rsi_change_3",
        "breakout_20",
        "drawdown_10",
        "rel_return_5",
        "rel_return_10",
        "beta_60",
        "corr_60",
        "volume_ma_ratio",
        "volume_zscore_20",
    ]
    mask = out[feature_cols].notna().all(axis=1) & target.notna()
    X = out.loc[mask, feature_cols]
    y = target.loc[mask]

    if len(X) < 60:
        raise ValueError("Not enough data for ML training. Need at least ~60 rows.")

    split_idx = int(len(X) * train_ratio)
    if split_idx < 30 or len(X) - split_idx < 10:
        raise ValueError("Train/test split too small. Adjust --train-ratio or use more data.")

    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
        )
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200)
    else:
        raise ValueError("model_type must be 'rf' or 'logreg'")

    model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    proba = model.predict_proba(X)[:, 1]
    pred_position = (proba >= prob_threshold).astype(int)

    out["ml_proba"] = np.nan
    out.loc[X.index, "ml_proba"] = proba

    position = pd.Series(0, index=out.index, dtype=int)
    test_index = X.index[split_idx:]
    position.loc[test_index] = pred_position[split_idx:]
    out["position"] = position

    position_change = out["position"].diff().fillna(out["position"])
    out["buy_signal"] = position_change > 0
    out["sell_signal"] = position_change < 0
    return out
