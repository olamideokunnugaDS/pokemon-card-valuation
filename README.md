**Multimodal Valuation Engine for Collectible Pokémon Cards**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Project Overview**

A production-grade multimodal machine learning system that integrates visual condition analysis (CNN-trained), time-aware market modelling (hybrid temporal regressor), and probabilistic fusion to generate interpretable, uncertainty-calibrated valuations for graded Pokémon trading cards.

*Core Innovation*: Disentangling intrinsic card quality from extrinsic market dynamics to produce stable, defensible valuations.


**System Architecture**

```bash
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                             │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Card Image      │         │  Market Data     │          │
│  │  (High-Res)      │         │  (Time-Series)   │          │
│  └──────────────────┘         └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────┐
│   VISION MODULE     │      │   MARKET MODULE     │
│   (CNN Encoder)     │      │   (Hybrid Temporal) │
│                     │      │                     │
│  • Centering        │      │  • Rolling Avg      │
│  • Edge Wear        │      │  • Momentum         │
│  • Surface Quality  │      │  • Volatility       │
│  • Corner Analysis  │      │  • Volume Signals   │
│                     │      │  • Regime State     │
└─────────────────────┘      └─────────────────────┘
           │                              │
           │    C_visual (128-256)        │   T_dynamic (16-64)
           └──────────────┬───────────────┘
                          ▼
              ┌─────────────────────┐
              │   FUSION NETWORK    │
              │   (Multimodal)      │
              │                     │
              │  • Embedding Align  │
              │  • Feature Fusion   │
              │  • Uncertainty Est. │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  VALUATION OUTPUT   │
              │                     │
              │  V ~ N(μ, σ²)       │
              │  • Point Estimate   │
              │  • Confidence Int.  │
              │  • Feature Contrib. │
              └─────────────────────┘
```



**Repository Structure**

```bash

pokemon-card-valuation/
│
├── src/                          ## Source code modules
│   ├── vision_module/            ## CNN-based condition encoder
│   │   ├── model.py              ## Architecture definition
│   │   ├── data_loader.py        ## Image preprocessing pipeline
│   │   ├── train.py              ## Training loop
│   │   ├── interpretability.py   ## Grad-CAM, saliency maps
│   │   └── feature_extractor.py  ## Embedding extraction
│   │
│   ├── market_module/            ## Time-aware market state encoder
│   │   ├── feature_engineering.py ## Time-series feature creation
│   │   ├── model.py              ## Hybrid regressor
│   │   ├── train.py              ## Temporal training protocol
│   │   └── regime_detector.py    ## Market regime classification
│   │
│   ├── fusion_module/            ## Multimodal integration
│   │   ├── fusion_network.py     ## Architecture
│   │   ├── train.py              ## End-to-end training
│   │   ├── inference.py          ## Valuation pipeline
│   │   └── uncertainty.py        ## Probabilistic outputs
│   │
│   ├── data_pipeline/            ## Data management
│   │   ├── collectors/           ## Data scrapers (eBay, PSA, etc.)
│   │   ├── preprocessors/        ## Cleaning & transformation
│   │   ├── validators/           ## Data quality checks
│   │   └── augmentation.py       ## Image augmentation
│   │
│   ├── evaluation/               ## Evaluation & metrics
│   │   ├── metrics.py            ## MAE, RMSE, MAPE, calibration
│   │   ├── ablation.py           ## Ablation study runners
│   │   ├── backtesting.py        ## Temporal validation
│   │   └── visualization.py      ## Results plotting
│   │
│   └── utils/                    ## Shared utilities
│       ├── config_loader.py      ## Configuration management
│       ├── logging_setup.py      ## Logging infrastructure
│       ├── seed_manager.py       ## Reproducibility controls
│       └── helpers.py            ## Common functions
│
├── configs/                      ## Configuration files
│   ├── vision/
│   │   └── cnn_config.yaml       ## CNN hyperparameters
│   ├── market/
│   │   └── temporal_config.yaml  ## Market model settings
│   ├── fusion/
│   │   └── fusion_config.yaml    ## Fusion architecture
│   └── data/
│       └── data_config.yaml      ## Data pipeline settings
│
├── data/                         ## Data storage (gitignored)
│   ├── raw/                      ## Original scraped data
│   ├── processed/                ## Cleaned datasets
│   ├── embeddings/               ## Extracted features
│   └── external/                 ## External reference data
│
├── models/                       ## Trained model artifacts
│   ├── vision/                   ## CNN checkpoints
│   ├── market/                   ## Market model weights
│   ├── fusion/                   ## Fusion network weights
│   └── checkpoints/              ## Training checkpoints
│
├── notebooks/                    ## Jupyter notebooks
│   ├── exploratory/              ## Initial data exploration
│   ├── analysis/                 ## Results analysis
│   └── visualization/            ## Figure generation
│
├── scripts/                      ## Executable scripts
│   ├── data_collection/          ## Data scraping scripts
│   ├── preprocessing/            ## Data preparation
│   ├── training/                 ## Model training orchestration
│   └── evaluation/               ## Evaluation runners
│
├── tests/                        ## Unit & integration tests
│   ├── unit/                     ## Component tests
│   └── integration/              ## End-to-end tests
│
├── results/                      ## Experiment outputs
│   ├── figures/                  ## Plots and visualizations
│   ├── tables/                   ## Numerical results
│   ├── metrics/                  ## Performance metrics
│   └── reports/                  ## Generated reports
│
├── docs/                         ## Documentation
│   ├── architecture/             ## System design docs
│   ├── methodology/              ## Research methodology
│   └── api/                      ## API documentation
│
├── logs/                         ## Training & experiment logs
│
├── .gitignore                    ## Git ignore rules
├── requirements.txt              ## Python dependencies
├── environment.yml               ## Conda environment (optional)
├── setup.py                      ## Package installation
├── pyproject.toml                ## Project metadata
├── Makefile                      ## Automation commands
├── LICENSE                       ## MIT License
└── README.md                     ## This file
```



**1. Environment Setup**

```bash
## Clone repository
git clone https://github.com/olamideokunnugaDS/pokemon-card-valuation.git
cd pokemon-card-valuation

## Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Or use conda
conda env create -f environment.yml
conda activate pokemon-valuation
```

**2. Data Preparation**

```bash
## Collect raw data
python scripts/data_collection/scrape_ebay.py --config configs/data/data_config.yaml

## Preprocess and clean
python scripts/preprocessing/prepare_data.py --input data/raw --output data/processed
```

**3. Training Pipeline**

```bash
## Train Vision Module
python scripts/training/train_vision.py --config configs/vision/cnn_config.yaml

## Train Market Module
python scripts/training/train_market.py --config configs/market/temporal_config.yaml

## Train Fusion Network (end-to-end)
python scripts/training/train_fusion.py --config configs/fusion/fusion_config.yaml
```

**4. Evaluation**

```bash
## Run full evaluation suite
python scripts/evaluation/run_evaluation.py --experiment fusion_v1

## Generate ablation studies
python scripts/evaluation/run_ablation.py --output results/ablation/
```

**5. Inference**

```python
from src.fusion_module.inference import ValuationEngine

## Initialize engine
engine = ValuationEngine.from_pretrained('models/fusion/best_model.pth')

## Generate valuation
result = engine.predict(
    image_path='path/to/card_image.jpg',
    metadata={'set': 'base_set', 'number': '4', 'grade': 'PSA 10'}
)

print(f"Estimated Value: ${result.mean_price:.2f}")
print(f"Confidence Interval: [{result.lower_bound:.2f}, {result.upper_bound:.2f}]")
print(f"Uncertainty: ±${result.std:.2f}")
```



**Research Questions**

1. *RQ1*: To what extent can a multimodal framework combining intrinsic visual features and time-aware market features produce accurate and stable valuations?

2. *RQ2*: How effectively can CNNs extract interpretable condition features that correlate with grading outcomes and market prices?

3. *RQ3*: Does incorporating engineered time-series market features improve valuation stability compared to static or purely time-series models?

4. *RQ4*: Does fusing intrinsic and extrinsic embeddings outperform unimodal models in predictive accuracy and robustness?



**Experimental Design**

*Module Evaluation*

| Module | Metrics | Validation Strategy |
|--------|---------|---------------------|
| *Vision* | Grade correlation, interpretability quality, robustness | Cross-validation, lighting/angle variations |
| *Market* | MAE, RMSE, MAPE, regime stability | Walk-forward, expanding window |
| *Fusion* | Multimodal improvement, uncertainty calibration | Out-of-sample backtesting, ablation |

**Ablation Studies**

- Vision-only baseline
- Market-only baseline
- Fusion (early vs late)
- Static vs temporal features
- With/without uncertainty estimation



**Technology Stack**

- *Deep Learning*: PyTorch, torchvision
- *Time-Series*: statsmodels, Prophet, NeuralProphet
- *ML*: scikit-learn, XGBoost, LightGBM
- *Data*: pandas, NumPy, Pillow
- *Visualization*: matplotlib, seaborn, plotly
- *Experiment Tracking*: Weights & Biases (wandb)
- *Testing*: pytest
- *Code Quality*: black, flake8, mypy



**Key Results**

*(To be updated after evaluation)*

- *Vision Module*: Grade correlation (ρ = TBD)
- *Market Module*: Out-of-sample MAPE = TBD%
- *Fusion System*: Multimodal improvement = TBD%
- *Uncertainty Calibration*: Coverage = TBD%



**Citation**

If you use this work, please cite:

```bibtex
@mastersthesis{Israel Okunnuga 2025pokemon,
  title={A Multimodal Framework for Grading-Aware Valuation of Collectible Assets: A Pokémon Case Study},
  author={Israel Okunnuga},
  year={2025},
  school={Aston University}
}
```



**License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



**Contributions**

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.



**Contact**

**ISRAEL OKUNNUGA**  
Email: olamideokunnuga@gmail.com  
GitHub: [@olamideokunnugaDS](https://github.com/olamideokunnugaDS)  
LinkedIn: [Israel Okunnuga](https://www.linkedin.com/in/israelokunnuga/)



**Acknowledgments**

- PSA and Beckett for grading standards reference
- Pokémon Company for intellectual property
- Academic supervisors and reviewers
- Open-source community



**Status**: Active Development (Week X/10)

**Last Updated**: [DATE]
