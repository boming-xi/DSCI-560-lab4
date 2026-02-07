from __future__ import annotations

from regime_detection import detect_regime
from risk_management import apply_stop_loss

import argparse
from pathlib import Path

import pandas as pd

from backtest_engine import backtest
from trading_ml_algo import generate_ml_signals
from trading_rule_algo import add_indicators, generate_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline stock trading algorithm.")
    parser.add_argument("--csv", help="Path to input CSV file.")
    parser.add_argument("--ticker", default="TSLA", help="Ticker for yfinance (used if --csv not set).")
    parser.add_argument("--start", default=None, help="Start date for yfinance (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date for yfinance (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1d", help="yfinance interval (e.g., 1d, 1h).")
    parser.add_argument("--date-col", default="Date", help="Date column in CSV.")
    parser.add_argument("--price-col", default="Close", help="Price column in CSV.")
    parser.add_argument(
        "--strategy",
        default="rule",
        choices=["rule", "ml"],
        help="Trading strategy: rule (SMA+RSI) or ml (classifier).",
    )
    parser.add_argument("--fast-window", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow-window", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period.")
    parser.add_argument("--rsi-buy", type=float, default=40.0, help="RSI threshold for buy filter.")
    parser.add_argument("--rsi-sell", type=float, default=70.0, help="RSI threshold for sell.")
    parser.add_argument("--model-type", default="rf", choices=["rf", "logreg"], help="ML model type.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split for ML.")
    parser.add_argument(
        "--proba-threshold", type=float, default=0.55, help="ML probability threshold for buy."
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for ML.")
    parser.add_argument("--initial-cash", type=float, default=10000.0, help="Initial capital.")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Trading fee rate per trade.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory.")
    return parser.parse_args()


def load_prices_from_csv(csv_path: str, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {date_col, price_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[[date_col, "High", "Low", price_col]].copy()
    out.columns = ["date", "high", "low", "close"]

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    return out


def load_prices_from_yfinance(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        group_by="column",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No data returned from yfinance. Check ticker or date range.")

    df = raw.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime" if "Datetime" in df.columns else df.columns[0]
    price_col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
    if price_col is None:
        raise ValueError("yfinance data missing Close/Adj Close column.")

    out = df[[date_col, "High", "Low", price_col]].copy()
    out.columns = ["date", "high", "low", "close"]

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    if args.csv:
        prices = load_prices_from_csv(args.csv, args.date_col, args.price_col)
    else:
        prices = load_prices_from_yfinance(args.ticker, args.start, args.end, args.interval)

    regime_params = detect_regime(prices)

    enriched = add_indicators(
        prices,
        fast_window=regime_params.fast_window,
        slow_window=regime_params.slow_window,
        rsi_period=args.rsi_period,
    )

    if args.strategy == "rule":
        signals = generate_signals(enriched, rsi_buy=args.rsi_buy, rsi_sell=args.rsi_sell)

        # risk management
        signals = apply_stop_loss(
            signals,
            stop_loss_pct=regime_params.stop_loss_pct,
        )

    else:
        signals = generate_ml_signals(
            enriched,
            train_ratio=args.train_ratio,
            prob_threshold=args.proba_threshold,
            model_type=args.model_type,
            random_state=args.random_state,
        )
    result = backtest(signals, initial_cash=args.initial_cash, fee_rate=args.fee_rate)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_path = out_dir / "signals.csv"
    metrics_path = out_dir / "metrics.txt"
    result.history.to_csv(signals_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as f:
        for key, value in result.metrics.items():
            f.write(f"{key}: {value}\n")

    print("Run complete.")
    print(f"Signals saved to: {signals_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nKey metrics:")
    for key in [
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "buy_hold_return",
        "trades_count",
    ]:
        print(f"- {key}: {result.metrics[key]:.6f}")


if __name__ == "__main__":
    main()
