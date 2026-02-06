from __future__ import annotations

import numpy as np
import pandas as pd


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1"] = out["close"].pct_change()
    out["return_2"] = out["close"].pct_change(2)
    out["return_5"] = out["close"].pct_change(5)
    out["volatility_10"] = out["return_1"].rolling(10).std()
    out["sma_ratio"] = (out["sma_fast"] / out["sma_slow"]) - 1.0
    out["price_vs_sma_fast"] = (out["close"] / out["sma_fast"]) - 1.0
    return out


def generate_ml_signals(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    prob_threshold: float = 0.55,
    model_type: str = "rf",
    random_state: int = 42,
) -> pd.DataFrame:
    out = add_ml_features(df)
    target = (out["close"].shift(-1) > out["close"]).astype(int)

    feature_cols = [
        "return_1",
        "return_2",
        "return_5",
        "volatility_10",
        "sma_ratio",
        "price_vs_sma_fast",
        "rsi",
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
