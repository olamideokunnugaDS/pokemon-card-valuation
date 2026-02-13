# Project Structure Documentation

## Overview

This document provides a comprehensive overview of the repository structure for the Pokémon Card Valuation Engine project.

## Directory Tree

```
pokemon-card-valuation/
│
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md             # Contribution guidelines
├── 📄 QUICKSTART.md               # Quick start guide
├── 📄 Makefile                    # Automation commands
├── 📄 requirements.txt            # Python dependencies
├── 📄 setup.py                    # Package installation
├── 📄 pyproject.toml              # Modern Python packaging config
│
├── 📁 src/                        # Source code (main package)
│   ├── 📁 vision_module/          # CNN-based condition encoder
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── data_loader.py
│   │   ├── train.py
│   │   ├── interpretability.py
│   │   └── feature_extractor.py
│   │
│   ├── 📁 market_module/          # Time-aware market encoder
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── regime_detector.py
│   │
│   ├── 📁 fusion_module/          # Multimodal integration
│   │   ├── README.md
│   │   ├── fusion_network.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── uncertainty.py
│   │
│   ├── 📁 data_pipeline/          # Data management
│   │   ├── collectors/
│   │   ├── preprocessors/
│   │   ├── validators/
│   │   └── augmentation.py
│   │
│   ├── 📁 evaluation/             # Metrics and evaluation
│   │   ├── metrics.py
│   │   ├── ablation.py
│   │   ├── backtesting.py
│   │   └── visualization.py
│   │
│   └── 📁 utils/                  # Shared utilities
│       ├── config_loader.py
│       ├── logging_setup.py
│       ├── seed_manager.py
│       └── helpers.py
│
├── 📁 configs/                    # Configuration files
│   ├── vision/
│   │   └── cnn_config.yaml
│   ├── market/
│   │   └── temporal_config.yaml
│   ├── fusion/
│   │   └── fusion_config.yaml
│   └── data/
│       └── data_config.yaml
│
├── 📁 data/                       # Data storage (gitignored)
│   ├── raw/                       # Original data
│   ├── processed/                 # Cleaned data
│   ├── embeddings/                # Extracted features
│   └── external/                  # External reference data
│
├── 📁 models/                     # Trained models (gitignored)
│   ├── vision/
│   ├── market/
│   ├── fusion/
│   └── checkpoints/
│
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── exploratory/               # EDA
│   ├── analysis/                  # Results analysis
│   └── visualization/             # Figure generation
│
├── 📁 scripts/                    # Executable scripts
│   ├── data_collection/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
│
├── 📁 tests/                      # Unit & integration tests
│   ├── unit/
│   └── integration/
│
├── 📁 results/                    # Experiment outputs
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── reports/
│
├── 📁 docs/                       # Documentation
│   ├── architecture/
│   ├── methodology/
│   └── api/
│
└── 📁 logs/                       # Training logs (gitignored)
```

## Key Files and Their Purpose

### Root Level

| File | Purpose |
|------|---------|
| `README.md` | Main project overview, architecture, usage |
| `LICENSE` | MIT License |
| `CONTRIBUTING.md` | Contribution guidelines |
| `QUICKSTART.md` | Quick start guide for new users |
| `Makefile` | Automation commands (install, test, train, etc.) |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation configuration |
| `pyproject.toml` | Modern Python packaging and tool configuration |
| `.gitignore` | Git ignore rules |

### Source Code (`src/`)

#### Vision Module
- `model.py`: CNN architecture (ResNet/EfficientNet based)
- `data_loader.py`: Image preprocessing pipeline
- `train.py`: Training loop with validation
- `interpretability.py`: Grad-CAM, saliency maps
- `feature_extractor.py`: Extract condition embeddings

#### Market Module
- `model.py`: Hybrid temporal regressor (XGBoost/LightGBM)
- `feature_engineering.py`: Time-series feature creation
- `train.py`: Temporal training protocol
- `regime_detector.py`: Market regime classification

#### Fusion Module
- `fusion_network.py`: Multimodal fusion architecture
- `train.py`: End-to-end training
- `inference.py`: Production valuation engine
- `uncertainty.py`: Probabilistic outputs and calibration

#### Data Pipeline
- `collectors/`: Data scrapers (eBay, PSA, etc.)
- `preprocessors/`: Data cleaning and transformation
- `validators/`: Data quality checks
- `augmentation.py`: Image augmentation

#### Evaluation
- `metrics.py`: Evaluation metrics (MAE, RMSE, MAPE, etc.)
- `ablation.py`: Ablation study runners
- `backtesting.py`: Temporal validation
- `visualization.py`: Results plotting

#### Utils
- `config_loader.py`: Load and validate YAML configs
- `logging_setup.py`: Logging infrastructure
- `seed_manager.py`: Reproducibility controls
- `helpers.py`: Common utility functions

### Configuration Files (`configs/`)

Each module has its own YAML configuration file:

- `vision/cnn_config.yaml`: Vision module hyperparameters
- `market/temporal_config.yaml`: Market module settings
- `fusion/fusion_config.yaml`: Fusion network configuration
- `data/data_config.yaml`: Data pipeline settings

### Scripts (`scripts/`)

Executable scripts for common operations:

- `data_collection/`: Data scraping scripts
- `preprocessing/`: Data preparation scripts
- `training/`: Training orchestration scripts
- `evaluation/`: Evaluation runners

### Tests (`tests/`)

- `unit/`: Unit tests for individual components
- `integration/`: End-to-end integration tests

## Design Principles

### 1. Modularity
Each component (vision, market, fusion) is self-contained and can be developed/tested independently.

### 2. Separation of Concerns
- **Data**: Separate directory for raw, processed, and derived data
- **Models**: Separate storage for trained weights
- **Code**: Clean separation between modules
- **Configuration**: All hyperparameters in YAML files

### 3. Reproducibility
- All random seeds managed centrally
- Configuration files for all experiments
- Deterministic training modes available
- Comprehensive logging

### 4. Production Awareness
- Modular inference pipeline
- Model checkpointing
- Experiment tracking integration
- Clear API boundaries

### 5. Research Quality
- Comprehensive documentation
- Unit and integration tests
- Ablation study infrastructure
- Visualization tools

## Workflow

### Development Workflow
1. Modify code in `src/`
2. Update tests in `tests/`
3. Run tests: `make test`
4. Format code: `make format`
5. Commit changes

### Training Workflow
1. Prepare data: `scripts/preprocessing/`
2. Train vision: `scripts/training/train_vision.py`
3. Train market: `scripts/training/train_market.py`
4. Train fusion: `scripts/training/train_fusion.py`
5. Evaluate: `scripts/evaluation/run_evaluation.py`

### Experiment Workflow
1. Modify configs in `configs/`
2. Run training with new config
3. Results saved to `results/`
4. Compare experiments using notebooks

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Config files**: `module_config.yaml`
- **Notebooks**: `01_descriptive_name.ipynb` (numbered for order)
- **Models**: `model_name_v{version}_{date}.pth`
- **Results**: `experiment_name_{metric}_{date}.csv`

## Version Control Strategy

### What to Track
- All source code
- Configuration files
- Documentation
- Scripts
- Tests
- Selected result figures (for reproducibility)

### What NOT to Track (gitignored)
- Data files (`data/`)
- Model weights (`models/`)
- Logs (`logs/`)
- Temporary files
- API keys and credentials

## Getting Started

1. **Clone repository**
2. **Install dependencies**: `make install`
3. **Verify setup**: `make test`
4. **Read QUICKSTART.md**
5. **Start with notebooks** in `notebooks/exploratory/`

## Dependencies Management

- **Core**: Listed in `requirements.txt`
- **Development**: Listed in `setup.py` under `dev` extras
- **Optional**: Listed in `setup.py` under `viz`, `tracking` extras

Install all dependencies:
```bash
make install-dev
```

## Continuous Integration (Future)

Planned CI/CD pipeline:
- Automated testing on push
- Code quality checks
- Documentation generation
- Model performance monitoring

## Questions?

See `CONTRIBUTING.md` or open an issue on GitHub.
