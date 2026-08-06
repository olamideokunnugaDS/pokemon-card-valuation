"""Canonical evaluation convention, shared by the market and fusion modules.

All models in this project predict `log_price` (log1p of dollar price).
This module defines the single evaluation function every training script
uses, so that market-module and fusion-module results are directly
comparable — no ad-hoc metric code anywhere else in the pipeline.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray, label: Optional[str] = None) -> Dict[str, float]:
    """Evaluate log-space predictions against log-space truth.

    Parameters
    ----------
    y_true_log : array-like, log(price + 1) actuals
    y_pred_log : array-like, log(price + 1) predictions
    label : optional, if given the metrics are also printed

    Returns
    -------
    dict with keys: mae_usd, rmse_usd, mape_pct, r2_log, n
    """
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)

    # Dollar space for error metrics
    y_true_usd = np.expm1(y_true_log)
    y_pred_usd = np.expm1(y_pred_log)

    # Clip negative predictions — no card can be worth < $0
    y_pred_usd = np.clip(y_pred_usd, 0, None)

    mae_usd = mean_absolute_error(y_true_usd, y_pred_usd)
    rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))

    nonzero = y_true_usd > 0
    mape_pct = np.mean(np.abs((y_true_usd[nonzero] - y_pred_usd[nonzero]) / y_true_usd[nonzero])) * 100

    # R² on log scale (standard for log-transformed regression)
    r2_log = r2_score(y_true_log, y_pred_log)

    results = {
        "mae_usd": float(mae_usd),
        "rmse_usd": float(rmse_usd),
        "mape_pct": float(mape_pct),
        "r2_log": float(r2_log),
        "n": int(len(y_true_log)),
    }

    if label:
        print(
            f"{label:<25} n={results['n']:>4d}  "
            f"MAE=${results['mae_usd']:>7,.2f}  "
            f"RMSE=${results['rmse_usd']:>8,.2f}  "
            f"MAPE={results['mape_pct']:>6.1f}%  "
            f"R²(log)={results['r2_log']:>6.3f}"
        )

    return results
