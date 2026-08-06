"""Training entrypoints for the market module: XGBoost baseline and LSTM sequence model.

Both functions are parametrized (no notebook-global closures) so they can be
called directly from a script, a notebook, or a test. They both report
through `evaluation.metrics.evaluate_predictions`, the shared log-space
evaluation convention used across market and fusion modules.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from evaluation.metrics import evaluate_predictions
from market_module.model import LSTMRegressor


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    xgb_params: Optional[Dict] = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
    seed: int = 42,
    label: str = "xgboost",
) -> Dict:
    """Train an XGBoost regressor on log_price with early stopping on the val set.

    Reports train/val/test metrics via `evaluate_predictions`. Used both for
    the primary hybrid-feature model and the static-only / static+calendar
    ablation baselines — pass different `feature_names`/columns to reproduce
    either.
    """
    default_params = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
    }
    params = {**default_params, **(xgb_params or {}), "seed": seed, "verbosity": 0}

    dtr = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dva = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dte = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    model = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )

    iter_range = (0, model.best_iteration + 1)
    pred_tr = model.predict(dtr, iteration_range=iter_range)
    pred_va = model.predict(dva, iteration_range=iter_range)
    pred_te = model.predict(dte, iteration_range=iter_range)

    metrics = {
        "train": evaluate_predictions(y_train, pred_tr, label=f"{label}/train"),
        "val": evaluate_predictions(y_val, pred_va, label=f"{label}/val"),
        "test": evaluate_predictions(y_test, pred_te, label=f"{label}/test"),
    }

    return {
        "model": model,
        "best_iteration": model.best_iteration,
        "features": feature_names,
        "pred_train": pred_tr,
        "pred_val": pred_va,
        "pred_test": pred_te,
        "metrics": metrics,
    }


def train_lstm(
    X_train: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, mask_test: np.ndarray, y_test: np.ndarray,
    n_features: int,
    device,
    hidden: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    max_epochs: int = 80,
    patience: int = 10,
    seed: int = 42,
) -> Dict:
    """Train the LSTM sequence model with early stopping on val MSE.

    Consolidates what was previously two near-identical training loops in
    the source notebook (a single-run version and a seed-robustness wrapper)
    into one parametrized function.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xt_tr = torch.from_numpy(X_train).float()
    Mt_tr = torch.from_numpy(mask_train).float()
    yt_tr = torch.from_numpy(y_train).float()
    Xt_va = torch.from_numpy(X_val).float()
    Mt_va = torch.from_numpy(mask_val).float()
    Xt_te = torch.from_numpy(X_test).float()
    Mt_te = torch.from_numpy(mask_test).float()

    model = LSTMRegressor(n_features, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(Xt_tr, Mt_tr, yt_tr),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val, patience_ctr, best_state = float("inf"), 0, None
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            pred = model(xb, mb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xt_va.to(device), Mt_va.to(device)).cpu().numpy()
        val_loss = float(np.mean((vp - y_val) ** 2))
        history.append({"epoch": epoch, "val_mse": val_loss})

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        pred_tr = model(Xt_tr.to(device), Mt_tr.to(device)).cpu().numpy()
        pred_va = model(Xt_va.to(device), Mt_va.to(device)).cpu().numpy()
        pred_te = model(Xt_te.to(device), Mt_te.to(device)).cpu().numpy()

    metrics = {
        "train": evaluate_predictions(y_train, pred_tr, label="lstm/train"),
        "val": evaluate_predictions(y_val, pred_va, label="lstm/val"),
        "test": evaluate_predictions(y_test, pred_te, label="lstm/test"),
    }

    return {
        "model": model,
        "best_val_mse": best_val,
        "epochs_trained": len(history),
        "history": history,
        "pred_train": pred_tr,
        "pred_val": pred_va,
        "pred_test": pred_te,
        "metrics": metrics,
    }
