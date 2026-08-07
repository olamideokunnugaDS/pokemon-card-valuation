# Technical Report: Multimodal Pokémon Card Valuation

This is the full write-up behind the headline numbers in the README including methodology, complete results, and an honest account of what didn't work. It's written for an engineering audience with no institutional framing, no formal research-question apparatus, just what was built, what was tested, and what the data actually showed.

## 1. Problem

Estimate the resale price of a PSA-graded Pokémon card from two inputs: a photo of the graded slab, and the market's recent transaction history for that card. The central design bet is that these two signals (physical condition and market state) should be encoded independently and fused late, rather than thrown into one model as a flat feature vector. This report tests that bet against the alternative directly.

## 2. Data

3,812 auction transactions across 7 Pokémon card species, PSA grades 8-10, scraped from eBay sold listings and cross-referenced against PSA's public certification API to confirm grade and authenticity. Transactions span January 2021 to April 2026, a window that includes substantial, real price drift in the collectibles market, which turns out to matter a great deal for which models generalize.

Splits are strictly temporal and enforced programmatically. For a transaction at time *t*, a model only ever sees information from transactions strictly before *t*. No card appears in both a training and test partition for the vision module. Outliers were floored at $1 and capped at $50,000 (3 observations removed). After restricting to grades 8–10 and excluding entries with expired image URLs, the final dataset has 3,812 rows from an initial 5,000.

| Split | Rows | Ends |
|---|---|---|
| Train | 2,170 | 24 Mar 2025 |
| Val | 592 | 2 Oct 2025 |
| Test | 1,050 | 13 Apr 2026 |

This produces uneven split sizes (transaction volume grew over time), but preserves chronological integrity. The alternative, a random split, would leak future market information backward and overstate every model's real-world performance.

## 3. Vision module: intrinsic condition encoding

**Architecture:** A frozen ImageNet-pretrained ResNet50 backbone feeds two sequential encoder heads

- **Stage 1 (identity):** trained to classify which of the 7 card species a slab image shows. Trained first, then frozen.
- **Stage 2 (condition):** trained to classify PSA grade (8/9/10), with an orthogonality penalty added to the loss that pushes its embeddings' cosine similarity toward the (frozen) Stage 1 embeddings down to zero.

The point of the two-stage split is to stop the condition encoder from taking a shortcut-learning to recognize *which card it is* as a proxy for grade, rather than actually learning what wear and surface quality look like. A single-stage version of this model (no orthogonality constraint) was tested earlier and reached 52.1% grade accuracy, several points higher than the disentangled version below, but that version was gamed by exactly the identity-shortcut it wasn't supposed to be using.

**Results:**

| Stage | Task | Accuracy | Baseline | Lift | Notes |
|---|---|---|---|---|---|
| 1. Identity | 7-way species classification | 91.3% | 21.3% (majority class) | +69.9pp | macro F1 = 0.91 |
| 2. Condition | 3-way grade classification | 49.0% | 40.2% (majority class) | +8.8pp | orthogonality \|cos_sim\| ≈ 0.000004 |

Identity classification is close to solved. Condition is harder (PSA 8 vs. 9 vs. 10 differences are visually subtle even to trained human graders) but the encoder captures real ordinal structure rather than noise:

- 93.8% of predictions fall within one adjacent grade (i.e. off-by-one-grade errors dominate, the model rarely confuses PSA 8 for PSA 10)
- Mean absolute error: 0.542 grade-levels
- Global clustering by grade is weak (the condition embedding space doesn't cleanly separate into three grade clusters), but local, card-relative ordinal structure is present

**Known failure modes**, documented directly rather than smoothed over:

- *PSA 9 bias:* the classifier over-predicts the majority grade (PSA 9) when uncertain, at the cost of PSA 8 and PSA 10 recall.
- *Embedding collapse:* 6 of 3,812 samples (0.16%) produce a zero condition-embedding vector, likely from ReLU killing all activations on edge-case images. Negligible at this scale, but worth flagging for anyone extending the dataset.
- *Grade separability:* global clustering by grade (silhouette score) is negative. The model has not learned a globally separable condition space, only a locally consistent ordinal one. Likely ceiling of what's extractable from frozen ImageNet features at 224×224 without fine-grained, higher-resolution attention.

Grad-CAM attribution confirms the model attends to the card body (not the PSA label sticker, which would be an obvious shortcut). Visualizations are produced by `src/vision_module/interpretability.py`.

## 4. Market module: extrinsic market-state encoding

**Feature engineering:** All temporal features are computed with a strict look-back constraint, for a transaction at time *t*, only transactions with `date_sold < t` are visible. Three families of features:

- **Rolling statistics:** mean, median, std, and count of prior sale prices within 7/14/30-day windows, per card+grade
- **Momentum:** change in rolling average price between adjacent windows
- **Volume:** count of prior transactions within each window, both per-card and market-wide

**Models compared**, all evaluated on the identical test set (n=1,050):

| Model | Features | MAE | RMSE | MAPE | R²(log) | Stability |
|---|---|---|---|---|---|---|
| Static-only | 3 | $1,421 | $3,483 | 67.6% | −0.256 | — |
| Static + calendar | 6 | $1,361 | $3,463 | 65.4% | +0.108 | — |
| Hybrid XGBoost | 31 | $1,405 | $3,499 | 67.2% | **−0.134** | Overfit |
| **LSTM (sequence)** | last 10 transactions | **$1,284** | **$3,311** | 112.6% | **+0.307** | Clean |

The XGBoost model, despite access to the richest feature set, generalizes worst. Strong training R² collapses to negative test R², the signature of overfitting to a market regime that later shifted. A feature-ablation check on the XGBoost model (removing drift-exposed features like `days_since_start`) didn't fix this. No XGBoost variant achieves positive test R²(log). The LSTM, which conditions on each card's own recent transaction sequence at inference time rather than a fixed static mapping, is the only model that holds up under drift, and by a wide margin. This is the strongest single empirical finding in the market module: **the choice of temporal representation matters more than feature richness.**

The LSTM's much higher MAPE (112.6% vs. XGBoost's 67.2%) alongside a much better R²(log) and MAE looks contradictory at first but it isn't. MAPE is dominated by a handful of low-priced cards where even a small dollar error is a huge percentage error. R²(log) and dollar-scale MAE/RMSE are the more trustworthy metrics here given the price distribution's skew.

## 5. Fusion module: combining condition and market state

**Architecture:** A locked MLP head (`input_dim → 256 → 64 → 1`, ReLU, dropout 0.2, Huber loss) is shared across every fusion variant, only the input composition changes. This is deliberate: if the architecture were tuned per-variant, gains couldn't be cleanly attributed to *which inputs* were used versus *how the model was tuned*. Four embedding blocks, concatenated in a fixed order when used together:

- **identity** (256-dim, from Stage 1 vision)
- **condition** (256-dim, from Stage 2 vision)
- **market-LSTM** (64-dim, hidden state from the LSTM sequence model)
- **market-XGBoost** (64-dim, engineered feature block used by the XGBoost model)

16 variants were trained. Every single block alone, every pairwise and higher-order combination, a monolithic-XGBoost baseline on raw features (no fusion at all), and a sanity baseline (predict the training-set mean price). Every non-trivial variant was trained across 5 seeds, significance between variants was assessed with a paired double-bootstrap (1,000 resamples, independently resampling both rows and seeds).

**Full results, ranked by test R²(log):**

| Rank | Variant | Inputs | Dim | Test R²(log) | Test MAE |
|---|---|---|---|---|---|
| 1 | V10 | id + cond + mkt-LSTM | 576 | **+0.340 ± 0.062** | $1,253 |
| 2 | V9 | id + mkt-LSTM | 320 | +0.309 ± 0.037 | $1,274 |
| 3 | V14 | cond + mkt-LSTM | 320 | +0.243 ± 0.048 | $1,285 |
| 4 | V5 | mkt-LSTM only | 64 | +0.236 ± 0.056 | $1,301 |
| 5 | V13 | full (all four) | 640 | +0.190 ± 0.062 | $1,265 |
| 6 | V16 | cond + mkt-LSTM + mkt-XGB | 384 | -0.060 ± 0.091 | $1,313 |
| 7 | V2 | monolithic XGBoost (raw) | 31 | -0.105 ± 0.055 | $1,405 |
| 8 | V12 | id + cond + mkt-XGB | 576 | -0.139 ± 0.064 | $1,383 |
| 9 | V11 | id + mkt-XGB | 320 | -0.246 ± 0.088 | $1,411 |
| 10 | V8 | mkt-LSTM + mkt-XGB | 128 | -0.384 ± 0.209 | $1,352 |
| 11 | V6 | mkt-XGB only | 64 | -0.662 ± 0.285 | $1,460 |
| 12 | V4 | cond only | 256 | -0.674 ± 0.058 | $1,500 |
| 13 | V15 | cond + mkt-XGB | 320 | -0.689 ± 0.106 | $1,472 |
| 14 | V7 | id + cond (vision full) | 512 | -0.735 ± 0.050 | $1,455 |
| 15 | V3 | id only | 256 | -0.777 ± 0.039 | $1,471 |
| 16 | V1 | sanity (predict mean) | — | -0.870 ± 0.000 | $1,537 |

Full precision numbers, plus train/val R²(log) and the train-val-test generalization gap for every variant, are in [`results/fusion_master_comparison.csv`](../results/fusion_master_comparison.csv).

**Statistically supported comparisons** (95% bootstrap CI excludes zero):

| Comparison | Gap (R²(log)) | 95% CI | Reading |
|---|---|---|---|
| V10 - V2 (best fusion vs. monolithic XGBoost) | +0.453 | [+0.351, +0.562] | Decomposing condition/market and fusing late beats one model on raw features |
| V13 - V2 (full 4-way fusion vs. monolithic XGBoost) | +0.306 | [+0.207, +0.403] | Even the noisier full-fusion variant clears the monolithic baseline |
| V10 - V5 (best fusion vs. best unimodal) | +0.116 | [+0.032, +0.177] | Fusion adds real value over the strongest single modality |
| V9 - V11 (LSTM vs. XGBoost as fusion partner) | +0.551 | [+0.462, +0.650] | The choice of which market embedding to fuse matters enormously |
| V13 - V10 (adding XGBoost embedding to the best variant) | **-0.147** | [-0.236, -0.073] | Adding a weaker, overfit modality actively hurts a strong variant |
| V12 - V11 (condition lift, XGBoost-fusion variants) | +0.121 | [+0.043, +0.204] | Condition embedding helps even when paired with a weak market signal |

Two comparisons were directional but not statistically conclusive at the 95% level: V10 - V9 (condition's marginal lift over identity+LSTM alone, p=0.966 but CI marginally includes zero) and V14 - V9 (condition substituting for identity as the vision anchor, p=0.954, CI marginally includes zero). Both point the same direction as the supported results but with a smaller, noisier effect.

**The central finding:** decomposing intrinsic (condition) and extrinsic (market) representations and fusing them late beats collapsing everything into one model trained on raw features, clearly and by a wide margin. But fusion is not monotonically better with more inputs. The single worst decision in the entire lattice is adding the XGBoost market embedding to an already-strong 3-input variant, which drags performance down by 0.147 R²(log). XGBoost's test-set overfitting (Section 4) doesn't stay contained to XGBoost, it propagates into any fusion variant that includes it. The practical takeaway is that fusion architectures need input selection discipline, not just input-maximizing.

## 6. Limitations

- **Cold-start dependence:** The LSTM's strength comes from conditioning on each card's own recent transaction history. For a card with little or no sales history, this signal is unavailable, and performance is expected to degrade toward the static-baseline range. This is a structural limitation of the sequence-model approach, not a training artifact.
- **Small-data regime:** 3,812 transactions across 21 card-name × grade strata is a real constraint on how much can be claimed from any single comparison. Reported effect sizes and confidence intervals should be read as evidence from a controlled case study on this dataset, not as population-level claims about multimodal valuation generally.
- **Single grading authority, single franchise:** Scoped deliberately to PSA-graded Pokémon cards to avoid confounding from cross-market demand dynamics or inter-authority grading variation. Generalizing to other card games, grading services, or collectible categories is untested.
- **Vision condition ceiling:** The condition encoder's inability to globally separate grades in embedding space (Section 3) suggests frozen ImageNet features at 224×224 may be close to their practical ceiling for this task. Higher-resolution input or fine-grained attention are the likely next steps, not tested here.
- **Computational cost:** Running vision + market + fusion end to end is heavier than a single tabular model, which matters in resource-constrained deployment settings.

## 7. Reproducing this

- Notebooks `01`–`03` in `notebooks/` reproduce every number in this report, in order, and are self-contained (Colab or local).
- `src/` contains the same core logic extracted into a reusable library. See the README's "Notebooks and `src/`" section.
- Requires **Python 3.12+**: `03_fusion_module.ipynb` contains nested f-string literals (PEP 701) that don't parse on earlier Python versions.
