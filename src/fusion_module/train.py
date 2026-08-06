"""Training entrypoints for the fusion module: the shared MLP variants and
the monolithic-XGBoost baseline they're compared against.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from evaluation.metrics import evaluate_predictions
from fusion_module.dataset import build_variant_input
from fusion_module.model import EarlyStopper, make_fusion_mlp
from utils.seed_manager import set_seed


def _torch_predict(model: nn.Module, X_tensor: torch.Tensor, device, batch_size: int = 256) -> np.ndarray:
    """Inference helper: predict in batches with dropout disabled, return numpy log-scale predictions."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            pred = model(batch).squeeze(-1)
            out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def train_xgb_monolithic(
    fm: pd.DataFrame,
    mf: pd.DataFrame,
    variant: Dict,
    v0_feature_cols: List[str],
    xgb_params: Dict,
    seed: int,
) -> Dict:
    """Train the monolithic-XGBoost baseline (raw tabular features, no fusion).

    `xgb_params` should contain the boosting hyperparameters plus
    `n_estimators` and `early_stopping_rounds`; both are popped before
    being passed to `xgb.train`. This is the variant every fusion MLP is
    ultimately compared against.
    """
    X_tr, y_tr, ids_tr = build_variant_input(variant, "train", fm, mf, {}, v0_feature_cols)
    X_va, y_va, ids_va = build_variant_input(variant, "val", fm, mf, {}, v0_feature_cols)
    X_te, y_te, ids_te = build_variant_input(variant, "test", fm, mf, {}, v0_feature_cols)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=v0_feature_cols)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=v0_feature_cols)
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=v0_feature_cols)

    params = dict(xgb_params)
    params["seed"] = seed
    params["verbosity"] = 0
    n_rounds = params.pop("n_estimators")
    es_rounds = params.pop("early_stopping_rounds")

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=es_rounds,
        evals_result=evals_result,
        verbose_eval=0,
    )
    best_iter = int(booster.best_iteration)
    iter_range = (0, best_iter + 1)

    pred_tr = booster.predict(dtrain, iteration_range=iter_range)
    pred_va = booster.predict(dval, iteration_range=iter_range)
    pred_te = booster.predict(dtest, iteration_range=iter_range)

    metrics = {
        "train": evaluate_predictions(y_tr, pred_tr),
        "val": evaluate_predictions(y_va, pred_va),
        "test": evaluate_predictions(y_te, pred_te),
    }
    train_rmse = float(evals_result["train"]["rmse"][best_iter])
    val_rmse = float(evals_result["val"]["rmse"][best_iter])

    return {
        "model": booster,
        "best_iteration": best_iter,
        "metrics": metrics,
        "predictions": {
            "train": {"listing_id": ids_tr, "log_price_predicted": pred_tr.astype(np.float32)},
            "val": {"listing_id": ids_va, "log_price_predicted": pred_va.astype(np.float32)},
            "test": {"listing_id": ids_te, "log_price_predicted": pred_te.astype(np.float32)},
        },
        "train_rmse_log": train_rmse,
        "val_rmse_log": val_rmse,
        "train_val_gap": val_rmse - train_rmse,
    }


def train_fusion_mlp(
    fm: pd.DataFrame,
    mf: pd.DataFrame,
    variant: Dict,
    column_config: Dict[str, List[str]],
    device,
    seed: int,
    hidden_layers=(256, 64),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    huber_delta: float = 1.0,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
) -> Dict:
    """Train the shared-architecture MLP for one (variant, seed) pair.

    This is the single training function used across every MLP variant in
    the lattice (identity-only through full 4-way fusion) — only `variant`
    (which embedding blocks get concatenated, and therefore the resulting
    input dimension) changes between calls.

    Pipeline: seed everything -> build inputs for train/val/test -> fit
    StandardScaler on train only -> train with Adam + Huber loss, early
    stopping on val loss -> restore best-val weights -> predict all splits.
    """
    set_seed(seed)

    X_tr, y_tr, ids_tr = build_variant_input(variant, "train", fm, mf, column_config, [])
    X_va, y_va, ids_va = build_variant_input(variant, "val", fm, mf, column_config, [])
    X_te, y_te, ids_te = build_variant_input(variant, "test", fm, mf, column_config, [])

    input_dim = X_tr.shape[1]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    X_tr_t = torch.from_numpy(X_tr_s)
    y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va_s)
    X_te_t = torch.from_numpy(X_te_s)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, generator=g, drop_last=False,
    )

    X_va_t_dev = X_va_t.to(device)
    y_va_t_dev = torch.from_numpy(y_va).to(device)

    set_seed(seed)  # re-seed immediately before init so weight init is deterministic per (variant, seed)
    model = make_fusion_mlp(input_dim, hidden_layers=hidden_layers, dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=huber_delta)
    stopper = EarlyStopper(patience=patience, min_delta=0.0)

    learning_curve = {"epoch": [], "train_loss": [], "val_loss": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            train_losses.append(float(loss.item()))
        train_loss_epoch = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t_dev).squeeze(-1)
            val_loss_epoch = float(loss_fn(val_pred, y_va_t_dev).item())

        learning_curve["epoch"].append(epoch)
        learning_curve["train_loss"].append(train_loss_epoch)
        learning_curve["val_loss"].append(val_loss_epoch)

        if val_loss_epoch < stopper.best - stopper.min_delta:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if stopper.step(val_loss_epoch, epoch):
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    pred_tr = _torch_predict(model, X_tr_t, device)
    pred_va = _torch_predict(model, X_va_t, device)
    pred_te = _torch_predict(model, X_te_t, device)

    metrics = {
        "train": evaluate_predictions(y_tr, pred_tr),
        "val": evaluate_predictions(y_va, pred_va),
        "test": evaluate_predictions(y_te, pred_te),
    }
    train_val_gap_r2_log = metrics["train"]["r2_log"] - metrics["val"]["r2_log"]

    return {
        "model": model,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "n_epochs_trained": len(learning_curve["epoch"]),
        "best_val_loss": stopper.best,
        "metrics": metrics,
        "predictions": {
            "train": {"listing_id": ids_tr, "log_price_predicted": pred_tr.astype(np.float32)},
            "val": {"listing_id": ids_va, "log_price_predicted": pred_va.astype(np.float32)},
            "test": {"listing_id": ids_te, "log_price_predicted": pred_te.astype(np.float32)},
        },
        "train_val_gap_r2_log": train_val_gap_r2_log,
        "learning_curve": learning_curve,
        "scaler_mean_norm": float(np.linalg.norm(scaler.mean_)),
        "scaler_scale_norm": float(np.linalg.norm(scaler.scale_)),
    }
