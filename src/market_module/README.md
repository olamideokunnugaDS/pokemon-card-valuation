# Market Module

Time-aware market state encoder using hybrid temporal regression for modeling extrinsic market dynamics.

## Overview

The market module captures temporal patterns and market dynamics that drive card prices:

- **Seasonality**: Post-holiday corrections, quarterly patterns
- **Trends**: Long-term heating/cooling markets
- **Volatility**: Price stability and fluctuation patterns
- **Regime States**: Bull/bear/neutral market classification
- **Scarcity Signals**: Population reports and demand indicators

## Architecture

```
Input Features
  ├── Static: grade, population, set rarity
  └── Temporal: rolling stats, momentum, volume
         ↓
  Feature Engineering
    ├── Rolling windows (7d, 30d, 60d)
    ├── Momentum indicators
    ├── Volatility measures
    └── Seasonality features
         ↓
  Hybrid Regressor (XGBoost/LightGBM)
         ↓
  Market State Embedding (64-dim)
```

## Key Features

- **Time-Series Features**: Rolling averages, momentum, volatility
- **Regime Detection**: Bull/bear/neutral classification
- **Temporal Validation**: Walk-forward, expanding window
- **Stationarity Handling**: Differencing, detrending
- **Baseline Comparisons**: ARIMA, Prophet, static models

## Usage

### Training

```bash
python scripts/training/train_market.py --config configs/market/temporal_config.yaml
```

### Inference

```python
from src.market_module.model import MarketEncoder
from src.market_module.feature_engineering import engineer_features

# Load model
model = MarketEncoder.from_pretrained('models/market/best_model.pth')

# Prepare features
features = engineer_features(
    card_data=card_df,
    include_temporal=True
)

# Get market state embedding
embedding = model.predict(features)

print(f"Market State: {embedding.shape}")  # (64,)
```

### Feature Engineering

```python
from src.market_module.feature_engineering import (
    compute_rolling_stats,
    compute_momentum,
    detect_regime
)

# Rolling statistics
rolling_features = compute_rolling_stats(
    price_series=prices,
    windows=[7, 30, 60]
)

# Momentum indicators
momentum = compute_momentum(
    price_series=prices,
    periods=[7, 30]
)

# Regime detection
regime = detect_regime(price_series=prices)
```

## Files

- `model.py`: Hybrid temporal regressor
- `feature_engineering.py`: Time-series feature creation
- `train.py`: Temporal training protocol
- `regime_detector.py`: Market regime classification
- `backtesting.py`: Walk-forward validation

## Evaluation Metrics

- **Forecast Accuracy**: MAE, RMSE, MAPE
- **Temporal Stability**: Out-of-sample performance over time
- **Regime Performance**: Accuracy within bull/bear/neutral markets
- **Feature Importance**: Contribution of temporal vs static features

## Why Hybrid Over Pure Time-Series?

Pure LSTM/ARIMA models can produce unstable predictions on sparse data. Our hybrid approach:

1. Uses engineered time-series features (stable, interpretable)
2. Combines with gradient boosting (robust to noise)
3. Maintains temporal awareness without volatility of pure sequence models
4. Allows explicit feature importance analysis

## References

- Makridakis et al. (2018): Statistical and Machine Learning forecasting methods
- Petropoulos et al. (2022): Forecasting: theory and practice
