Stock Trading System

## Quick start

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Option A: Use yfinance (default uses TSLA):

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
  --proba-threshold 0.48 \
  --aggressive \
  --out-dir outputs
```

4. Check outputs:
- Rule strategy: `outputs/signals_rule.csv`, `outputs/metrics_rule.txt`
- ML strategy: `outputs/signals_ml.csv`, `outputs/metrics_ml.txt`
