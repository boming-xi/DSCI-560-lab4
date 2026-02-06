Stock Trading System

This repo now includes a baseline algorithm development starter for your
assignment:

- Technical indicators: SMA (fast/slow) + RSI
- Buy/sell signal generation
- Simple backtest with portfolio metrics
- Optional MAE/RMSE evaluation helper for forecasting models

## Quick start

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Option A: Use yfinance (default uses TSLA):

```bash
python3 src/run_baseline.py --ticker TSLA --start 2020-01-01 --out-dir outputs
```

Rule-based strategy (SMA + RSI):

```bash
python3 src/run_baseline.py \
  --strategy rule \
  --ticker TSLA \
  --start 2020-01-01 \
  --out-dir outputs
```

ML strategy example (Random Forest classifier):

```bash
python3 src/run_baseline.py \
  --strategy ml \
  --ticker TSLA \
  --start 2020-01-01 \
  --model-type rf \
  --train-ratio 0.7 \
  --proba-threshold 0.55 \
  --out-dir outputs
```

4. Check outputs:
- `outputs/signals.csv`
- `outputs/metrics.txt`
