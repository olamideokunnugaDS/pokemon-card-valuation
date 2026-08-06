"""Ablation and significance-testing utilities.

These are the tools behind the project's headline comparisons — e.g.
"decomposed identity+condition fusion beats monolithic XGBoost by
+0.45 R²(log), 95% CI [0.35, 0.56]." Every reported gap between two
variants goes through `paired_bootstrap_pair`.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def r2_log_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast R²(log): 1 - SS_res / SS_tot, without sklearn overhead. Used inside
    the bootstrap loop, where `evaluate_predictions` would be too slow to
    call thousands of times."""
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1.0 - ss_res / ss_tot


def paired_bootstrap_pair(
    preds_matrix: np.ndarray,
    actuals_vector: np.ndarray,
    variant_index: Dict[int, int],
    vid_a: int,
    vid_b: int,
    n_iter: int = 1000,
    rng_seed: int = 42,
) -> Dict:
    """Paired double-bootstrap of the test R²(log) gap between two variants.

    Parameters
    ----------
    preds_matrix : (n_rows, n_seeds, n_variants) array of log-price predictions
    actuals_vector : (n_rows,) array of log-price actuals
    variant_index : maps variant_id -> column index into preds_matrix's last axis
    vid_a, vid_b : the two variant ids being compared (gap = a - b)
    n_iter : number of bootstrap resamples
    rng_seed : seed for the resampling RNG (independent of model training seeds)

    Each iteration resamples row indices AND seed indices independently
    (both with replacement), averages predictions across the resampled
    seeds per row, then computes the R²(log) gap on the resampled rows.
    Returns the observed gap plus a 95% percentile confidence interval.
    """
    rng = np.random.default_rng(rng_seed)
    a_idx = variant_index[vid_a]
    b_idx = variant_index[vid_b]
    n_rows, n_seeds, _ = preds_matrix.shape

    pred_a_full = preds_matrix[:, :, a_idx].mean(axis=1)
    pred_b_full = preds_matrix[:, :, b_idx].mean(axis=1)
    obs_gap = r2_log_fast(actuals_vector, pred_a_full) - r2_log_fast(actuals_vector, pred_b_full)

    gaps = np.empty(n_iter, dtype=np.float64)
    for k in range(n_iter):
        row_resample = rng.integers(0, n_rows, size=n_rows)
        seed_resample = rng.integers(0, n_seeds, size=n_seeds)

        y_true_resampled = actuals_vector[row_resample]
        pred_a = preds_matrix[row_resample, :, a_idx][:, seed_resample].mean(axis=1)
        pred_b = preds_matrix[row_resample, :, b_idx][:, seed_resample].mean(axis=1)

        gaps[k] = r2_log_fast(y_true_resampled, pred_a) - r2_log_fast(y_true_resampled, pred_b)

    ci_lo = float(np.percentile(gaps, 2.5))
    ci_hi = float(np.percentile(gaps, 97.5))
    p_pos = float((gaps > 0).mean())

    return {
        "pair": f"V{vid_a} - V{vid_b}",
        "vid_a": vid_a,
        "vid_b": vid_b,
        "observed_gap": float(obs_gap),
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "p_gap_pos": p_pos,
        "n_iterations": n_iter,
    }


def classify(observed_gap: float, ci_lo: float, ci_hi: float, p_pos: float, threshold: float = 0.95) -> str:
    """Classify a bootstrap comparison as supported / directional / inconclusive."""
    if ci_lo > 0:
        return "supported (positive)"
    if ci_hi < 0:
        return "supported (negative)"
    if p_pos > threshold:
        return f"directional positive (p={p_pos:.3f}; CI marginally includes 0)"
    if p_pos < (1 - threshold):
        return f"directional negative (p={1 - p_pos:.3f}; CI marginally includes 0)"
    return f"inconclusive (p={p_pos:.3f})"


def find_pair(results: List[Dict], vid_a: int, vid_b: int) -> Optional[Dict]:
    """Look up a specific (vid_a, vid_b) comparison from a list of bootstrap results."""
    for r in results:
        if r["vid_a"] == vid_a and r["vid_b"] == vid_b:
            return r
    return None


def subgroup_metrics(df: pd.DataFrame, subgroup_col: str, evaluate_fn: Callable) -> pd.DataFrame:
    """Aggregate per-row test predictions to per-(variant, subgroup) metrics.

    For each (variant_id, seed, subgroup_value) group, applies `evaluate_fn`
    (typically `evaluation.metrics.evaluate_predictions`) to that group's
    predictions, then aggregates across seeds to mean +/- std. Used for the
    coverage-stratified, temporal, and per-grade breakdowns.
    """
    rows = []
    for (vid, seed, sg), grp in df.groupby(["variant_id", "seed", subgroup_col], observed=True):
        if len(grp) == 0:
            continue
        m = evaluate_fn(grp["log_price_actual"].values, grp["log_price_predicted"].values)
        rows.append({
            "variant_id": int(vid),
            "seed": int(seed),
            subgroup_col: sg,
            "n": int(m["n"]),
            "mae_usd": m["mae_usd"],
            "rmse_usd": m["rmse_usd"],
            "mape_pct": m["mape_pct"],
            "r2_log": m["r2_log"],
        })
    metrics_long = pd.DataFrame(rows)

    agg = (
        metrics_long.groupby(["variant_id", subgroup_col])
        .agg(
            n_rows=("n", "first"),
            r2_mean=("r2_log", "mean"),
            r2_std=("r2_log", lambda x: float(np.std(x, ddof=0))),
            mae_mean=("mae_usd", "mean"),
            mae_std=("mae_usd", lambda x: float(np.std(x, ddof=0))),
        )
        .reset_index()
    )
    return agg


def assign_temporal_segment(date, thresholds) -> str:
    """Assign a date to 'early' / 'middle' / 'late' given two quantile thresholds."""
    if date < thresholds[0]:
        return "early"
    elif date < thresholds[1]:
        return "middle"
    return "late"
