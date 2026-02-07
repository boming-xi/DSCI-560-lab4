from __future__ import annotations

import pandas as pd
import numpy as np


def apply_stop_loss(
    df: pd.DataFrame,
    stop_loss_pct: float,
) -> pd.DataFrame:

    out = df.copy()

    entry_price = None
    position = []

    for i, row in out.iterrows():
        if row["position"] == 1 and entry_price is None:
            entry_price = row["close"]

        if row["position"] == 1 and entry_price is not None:
            if row["close"] <= entry_price * (1 - stop_loss_pct):
                # Stop loss triggered
                position.append(0)
                entry_price = None
                continue

        if row["position"] == 0:
            entry_price = None

        position.append(row["position"])

    out["position"] = position
    return out
