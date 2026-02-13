# Fusion Module

Multimodal integration network that combines visual condition embeddings and market state embeddings into unified probabilistic valuations.

## Overview

The fusion module answers the core question: **How do intrinsic quality and extrinsic market forces interact to determine price?**

It produces:
- **Point Estimate**: Fair market value V̂
- **Uncertainty Quantification**: Confidence intervals, prediction variance
- **Feature Attribution**: Decomposition of condition vs market contributions

## Architecture

```
Visual Embedding (256-dim) ──┐
                             ├──> Fusion Layer ──> Valuation Head ──> V ~ N(μ, σ²)
Market Embedding (64-dim) ───┘
                                                   
Fusion Strategies:
  - Early: Concatenate then process
  - Late: Process separately, then combine
  - Hybrid: Ensemble of both
```

## Key Features

- **Probabilistic Output**: V ~ N(μ, σ²) with calibrated uncertainty
- **Multi-strategy Fusion**: Early, late, or hybrid fusion
- **Uncertainty Decomposition**: Aleatoric + epistemic uncertainty
- **Interpretability**: SHAP values, attention weights, ablation analysis
- **Calibration**: Isotonic regression, temperature scaling

## Usage

### Training

```bash
python scripts/training/train_fusion.py --config configs/fusion/fusion_config.yaml
```

### Inference

```python
from src.fusion_module.inference import ValuationEngine

# Initialize engine
engine = ValuationEngine.from_pretrained('models/fusion/best_model.pth')

# Generate valuation
result = engine.predict(
    image_path='path/to/card.jpg',
    metadata={
        'grade': 'PSA 10',
        'set': 'Base Set',
        'card_name': 'Charizard',
        'card_number': '4'
    }
)

print(f"Estimated Value: ${result.mean_price:.2f}")
print(f"95% CI: [{result.lower_bound:.2f}, {result.upper_bound:.2f}]")
print(f"Uncertainty: ±${result.std:.2f}")
print(f"\nContributions:")
print(f"  Condition: {result.condition_contribution:.1%}")
print(f"  Market: {result.market_contribution:.1%}")
```

### Uncertainty Analysis

```python
from src.fusion_module.uncertainty import analyze_uncertainty

# Decompose uncertainty
uncertainty_breakdown = analyze_uncertainty(
    model=model,
    input_data=test_data,
    n_samples=100
)

print(f"Aleatoric (data) uncertainty: {uncertainty_breakdown['aleatoric']:.3f}")
print(f"Epistemic (model) uncertainty: {uncertainty_breakdown['epistemic']:.3f}")
```

### Interpretability

```python
from src.fusion_module.interpretability import explain_prediction

# Get SHAP values
explanation = explain_prediction(
    model=model,
    vision_embedding=vision_emb,
    market_embedding=market_emb
)

print("Feature Importance:")
for feature, importance in explanation['shap_values'].items():
    print(f"  {feature}: {importance:.3f}")
```

## Files

- `fusion_network.py`: Multimodal fusion architecture
- `train.py`: End-to-end training pipeline
- `inference.py`: Valuation engine for production use
- `uncertainty.py`: Probabilistic output and calibration
- `interpretability.py`: SHAP, attention analysis

## Evaluation Metrics

### Accuracy Metrics
- MAE, RMSE, MAPE
- R² score
- Median absolute error

### Probabilistic Metrics
- Negative log-likelihood (NLL)
- Calibration error
- Prediction interval coverage
- Sharpness (interval width)

### Ablation Studies
- Vision-only baseline
- Market-only baseline
- Fusion improvement quantification

## Fusion Strategies Compared

| Strategy | Pros | Cons |
|----------|------|------|
| **Early** | Simple, joint optimization | Risk of feature dominance |
| **Late** | Modular, interpretable | May miss interactions |
| **Hybrid** | Best of both | More complex |

Our default: **Late fusion with attention gating** - balances interpretability and performance.

## Uncertainty Calibration

We ensure predicted uncertainty reflects actual error distribution:

1. Split data into train/calibration/test
2. Train fusion network on train set
3. Calibrate uncertainty on calibration set (isotonic regression)
4. Evaluate calibrated uncertainty on test set

**Goal**: 68% of predictions should fall within 1σ interval, 95% within 2σ.

## References

- Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
- Guo et al. (2017): On Calibration of Modern Neural Networks
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions
