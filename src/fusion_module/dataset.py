"""Variant input construction and PyTorch Dataset wrapper for the fusion module.

The fusion architecture is a "variant lattice": 13 named input configurations
built by concatenating different combinations of four upstream embedding
blocks (vision-identity, vision-condition, market-LSTM, market-XGBoost) in a
locked column order. `build_variant_input` is the single function that
constructs those inputs; every downstream training function depends on it
producing row-aligned (X, y, listing_id) triples.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Canonical block concatenation order. Any variant with multiple input
# blocks concatenates them in this order — this is a contract, not an
# implementation detail: predictions are only comparable across runs if
# inputs are always built the same way.
BLOCK_ORDER = ["identity", "condition", "market_lstm", "market_xgb"]


def build_variant_input(
    variant: Dict,
    split: str,
    fm: pd.DataFrame,
    mf: pd.DataFrame,
    column_config: Dict[str, List[str]],
    v0_feature_cols: List[str],
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Construct the (X, y, listing_id) triple for one (variant, split) pair.

    Parameters
    ----------
    variant : one entry from the fusion contract's `variants` list, e.g.
        {"id": 10, "type": "mlp", "inputs": ["identity", "condition", "market_lstm"]}
    split : one of "train", "val", "test"
    fm : the fusion_master dataframe (identity/condition/market_lstm embeddings
        + market_xgb prediction + metadata, one row per listing)
    mf : the market_features dataframe (raw V0 tabular features), used only
        by the "monolithic_xgb" variant type
    column_config : maps block name + "_cols" -> list of column names in fm,
        e.g. {"identity_cols": [...], "condition_cols": [...], ...}
    v0_feature_cols : the raw feature columns used by the monolithic XGBoost
        baseline variant

    Returns
    -------
    X   : (n, d) float32 array, or None for the sanity-baseline variant
    y   : (n,) float32 array of log_price targets
    ids : (n,) str array of listing_id, keeping predictions traceable to rows
    """
    fm_slice = fm.loc[fm["split"] == split].reset_index(drop=True)
    y = fm_slice["log_price"].astype(np.float32).values
    ids = fm_slice["listing_id"].astype(str).values

    if variant["type"] == "sanity":
        return None, y, ids

    if variant["type"] == "monolithic_xgb":
        merged = fm_slice[["listing_id"]].merge(
            mf[["listing_id"] + v0_feature_cols], on="listing_id", how="left", validate="one_to_one",
        )
        assert len(merged) == len(fm_slice), f"merge changed row count: {len(merged)} vs {len(fm_slice)}"
        assert (merged["listing_id"].astype(str).values == ids).all(), "merge altered row ordering"
        X = merged[v0_feature_cols].astype(np.float32).values  # NaN preserved; XGBoost handles natively
        return X, y, ids

    block_cols = {
        "identity": column_config.get("identity_cols", []),
        "condition": column_config.get("condition_cols", []),
        "market_lstm": column_config.get("market_lstm_cols", []),
        "market_xgb": column_config.get("market_xgb_cols", []),
    }
    requested = variant["inputs"]
    pieces = [fm_slice[block_cols[b]].astype(np.float32).values for b in BLOCK_ORDER if b in requested]
    assert len(pieces) == len(requested), (
        f"variant {variant.get('id')} requested {requested} but built {len(pieces)} blocks"
    )
    X = np.concatenate(pieces, axis=1)
    return X, y, ids


class FusionDataset(Dataset):
    """Plain numpy-to-tensor wrapper, keeping listing_id alongside X/y for traceability."""

    def __init__(self, X: np.ndarray, y: np.ndarray, ids: np.ndarray):
        assert X.shape[0] == y.shape[0] == ids.shape[0], (
            f"Dataset shape mismatch: X={X.shape}, y={y.shape}, ids={ids.shape}"
        )
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.ids = ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
