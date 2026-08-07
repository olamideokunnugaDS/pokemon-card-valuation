# Pokémon Card Valuation

A multimodal machine learning system for valuing PSA-graded Pokémon cards, built around the core idea that card condition and market state are separate signals and should be learned separately before being fused.

Most valuation approaches either grade a card visually and ignore the market, or model price movement and ignore what the card actually looks like. This project builds both representations independently with a CNN encoder for physical condition and a temporal encoder for market state, then tests, empirically, whether fusing them beats either one alone, and whether keeping them separate beats mashing all features into one model.

Trained and evaluated on 3,812 real auction transactions (Jan 2021-Apr 2026) across 7 Pokémon card species, PSA grades 8-10, under a strict temporal split that forces every model to generalize forward in time through genuine market drift.

## Headline results

**Card condition (vision):** A two-stage disentangled CNN separates *what card it is* from *what condition it's in*. Stage 1 (identity) is trained first and frozen, Stage 2 (condition) is trained against an orthogonality penalty that pushes its embeddings away from Stage 1's, so condition signal can't just piggyback on identity.

| Stage | Task | Accuracy | Baseline | Lift |
|---|---|---|---|---|
| 1 - Identity | Which of 7 cards is this? | 91.3% | 21.3% | +69.9pp |
| 2 - Condition | PSA grade (8/9/10)? | 49.0% | 40.2% | +8.8pp |

The identity encoder is strong and clean (macro F1 = 0.91). Condition is the harder problem. PSA 8-10 differences are visually subtle, but the model captures real ordinal structure. 93.8% of predictions land within one adjacent grade, mean absolute error is 0.54 grade-levels, and the orthogonality constraint drives identity/condition cosine similarity to ~0.000004, i.e. near-complete disentanglement.

**Market state:** Four approaches were compared on identical held-out test data (n=1,050, spanning a period of significant price drift):

| Model | MAE | RMSE | MAPE | R²(log) |
|---|---|---|---|---|
| Static-only (3 features) | $1,421 | $3,483 | 67.6% | -0.256 |
| Static + calendar (6 features) | $1,361 | $3,463 | 65.4% | +0.108 |
| Hybrid XGBoost (31 features) | $1,405 | $3,499 | 67.2% | -0.134 |
| **LSTM (transaction-history sequence)** | **$1,284** | **$3,311** | 112.6% | **+0.307** |

The XGBoost model looks strong in training and then falls apart on test data, a classic overfitting to a market state that later drifted. The LSTM, which conditions on each card's own recent transaction history rather than memorizing static feature-price mappings, is the only model that holds up out of sample.

**Fusion:** 16 architectural variants were trained and evaluated (identity-only, condition-only, market-only, every pairwise combination, and full fusion), each averaged over 5 seeds, with statistical significance assessed via paired double-bootstrap (1,000 resamples over both rows and seeds).

- **Best variant:** identity + condition + market-LSTM embeddings fused through a small MLP head → **R²(log) = +0.340 ± 0.062** on test, the best of all 16 variants.
- **Decomposition beats monolithic modeling:** the best fusion variant beats a single XGBoost model trained on all raw features by **+0.453 R²(log)** (95% CI [+0.351, +0.562]). Keeping condition and market representations separate until late fusion is cleaner, and measurably more accurate.
- **Fusion beats the best single modality:** it beats the market-LSTM running alone by **+0.116 R²(log)** (95% CI [+0.032, +0.177]).
- **More modalities isn't automatically better:** adding the XGBoost market embedding into the best 3-input variant hurts it by −0.147 R²(log) (95% CI [−0.236, −0.073]). XGBoost's overfitting propagates straight into the fusion head. Picking the right inputs mattered more than maximizing input count.

Full per-variant numbers are in [`results/fusion_master_comparison.csv`](results/fusion_master_comparison.csv), every pairwise significance test is in [`results/fusion_bootstrap_results.json`](results/fusion_bootstrap_results.json). A fuller write-up, including subgroup breakdowns (temporal segments, PSA grade, cold-start listings) and failure-mode analysis, is in [`docs/technical_report.md`](docs/technical_report.md).

## Repository structure

```
pokemon-card-valuation/
├── notebooks/                          # the experiment record — run top to bottom, self-contained
│   ├── 01_vision_module_intrinsic_condition_encoder.ipynb
│   ├── 02_market_module.ipynb
│   └── 03_fusion_module.ipynb
├── src/                                 # productionized, reusable extraction of the core architecture
│   ├── vision_module/                   # IdentityEncoder, ConditionEncoder, orthogonality loss, Grad-CAM
│   ├── market_module/                   # rolling/momentum/volume features, LSTMRegressor, training loops
│   ├── fusion_module/                   # variant-input builder, fusion MLP, early stopping, training loop
│   ├── evaluation/                      # shared evaluate_predictions(), bootstrap + subgroup ablation tools
│   └── utils/                           # seed control, config loading
├── configs/                             # YAML configs for each module (data/vision/market/fusion)
├── models/                              # trained weights (market + fusion; see note below)
├── results/                             # headline metrics + bootstrap significance tests, as CSV/JSON
├── data/                                # sample of engineered market features (full parquet also included)
├── assets/sample_cards/                 # example PSA slab images referenced in the technical report
├── docs/technical_report.md             # full methodology, results, and limitations write-up
└── requirements.txt
```

### Why both notebooks *and* `src/`

The notebooks are the actual experiment record — they run top to bottom, each one self-contained, and reproduce every number in this README exactly as they were originally produced. `src/` is a separate, deliberate refactor: every model class and core training/evaluation function pulled out of the notebooks, stripped of notebook-global state, and rewritten to take explicit parameters instead. `evaluate_predictions` and `set_seed`, which were copy-pasted identically across all three notebooks, now exist once. Variant-specific orchestration and one-off diagnostic code stayed in the notebooks — that's exploratory glue, not reusable architecture.

If you want to see *how* a result was produced, read the notebook. If you want to reuse the architecture in a new project, import from `src/`.

## Setup

Requires **Python 3.12+** — `03_fusion_module.ipynb` uses nested f-strings (PEP 701) that only parse on 3.12 and later.

```bash
git clone https://github.com/olamideokunnugaDS/pokemon-card-valuation.git
cd pokemon-card-valuation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_vision_module_intrinsic_condition_encoder.ipynb
```

Each notebook detects whether it's running in Google Colab or locally and resolves paths accordingly — no manual path editing needed either way.

Using the extracted library directly:

```python
from src.evaluation.metrics import evaluate_predictions
from src.market_module.model import LSTMRegressor
from src.market_module.train import train_lstm
from src.fusion_module.model import make_fusion_mlp
```

## Data

3,812 auction transactions of PSA-graded Pokémon cards (7 species, grades PSA 8–10), scraped from eBay sold listings and cross-referenced against the PSA public certification API, spanning January 2021 to April 2026.

Splits are strictly temporal to prevent look-ahead leakage — for any transaction, only earlier transactions are ever visible to a model:

| Split | Rows | Ends |
|---|---|---|
| Train | 2,170 | 24 Mar 2025 |
| Val | 592 | 2 Oct 2025 |
| Test | 1,050 | 13 Apr 2026 |

`data/market_features_sample.csv` is a 100-row sample of the full engineered feature set (`data/market_features.parquet`, 3,812 rows × 42 columns) included so the pipeline can be inspected without loading the full file.

## What's not included

- **Scraping/collection scripts.** Data collection was done via ad-hoc eBay and PSA API automation rather than a reusable pipeline, so there's no `data_pipeline/` module here — it would need to be built fresh rather than extracted.
- **Vision model checkpoints.** The Stage 1/Stage 2 CNN weights aren't in this repo; retrain via `01_vision_module_intrinsic_condition_encoder.ipynb`, or reach out (contact below) if you need the trained weights directly.
- **`models/market/price_predictor.pkl`** is intentionally excluded — an auxiliary artifact that isn't part of the core pipeline (the two models that matter, XGBoost and the LSTM, are both included and validated).

## Tech stack

PyTorch + torchvision (ResNet50 backbone, LSTM), XGBoost, scikit-learn, pandas/PyArrow, NumPy/SciPy.

## Contact

**Israel Okunnuga**
[GitHub](https://github.com/olamideokunnugaDS) · [LinkedIn](https://www.linkedin.com/in/israelokunnuga/)
