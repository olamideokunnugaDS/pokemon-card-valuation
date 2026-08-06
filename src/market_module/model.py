"""LSTM sequence model and sequence-construction utilities for the market module.

The LSTM consumes the K most recent prior transactions for a card+grade
group (zero-padded, masked when fewer than K exist) and predicts the
current transaction's log-price from the hidden state at the last real
timestep. This is the model that generalises best under the 4x train-test
price-level drift in this dataset (see results/fusion_master_comparison.csv).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class LSTMRegressor(nn.Module):
    """Small, regularised single-layer LSTM regressor over a transaction history."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, K, F), mask: (B, K)
        lengths = mask.sum(dim=1).clamp(min=1).long()
        out, _ = self.lstm(x)  # (B, K, H)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(-1))
        last = out.gather(1, idx).squeeze(1)  # (B, H)
        return self.head(self.drop(last)).squeeze(-1)


def build_sequences(df_sorted: pd.DataFrame, feature_cols: List[str], k: int = 10):
    """Build zero-padded, masked transaction-history sequences.

    For each row, takes the `k` most recent prior transactions within its
    card_name+grade group (strictly past — the current row is never
    included in its own history).

    Returns
    -------
    X       : (N, k, F) sequence features, zero-padded at the start if needed
    mask    : (N, k)    1 where real data, 0 where padding
    y       : (N,)      target log_price for the current row
    idx     : (N,)      original dataframe index (for alignment)
    splits  : (N,)      split label for each row
    listing_ids : (N,)  listing id for each row
    """
    X, mask, y, idx, splits, lids = [], [], [], [], [], []

    for (_, _), g in df_sorted.groupby(["card_name", "grade"]):
        g = g.reset_index(drop=False)  # keep original index in a column
        for i in range(len(g)):
            hist = g.iloc[max(0, i - k):i]  # strictly past, at most k rows
            n_hist = len(hist)

            seq = np.zeros((k, len(feature_cols)), dtype=np.float32)
            m = np.zeros(k, dtype=np.float32)

            if n_hist > 0:
                seq[k - n_hist:k, :] = hist[feature_cols].values.astype(np.float32)
                m[k - n_hist:k] = 1.0

            X.append(seq)
            mask.append(m)
            y.append(g.iloc[i]["log_price"])
            idx.append(g.iloc[i]["index"])
            splits.append(g.iloc[i]["split"])
            lids.append(g.iloc[i]["listing_id"])

    return (
        np.stack(X), np.stack(mask), np.array(y),
        np.array(idx), np.array(splits), np.array(lids),
    )


def fit_normalization_stats(X: np.ndarray, mask: np.ndarray, continuous_idx: List[int]) -> Dict[int, Tuple[float, float]]:
    """Compute (mean, std) for each continuous feature index, over real (unmasked) values only."""
    stats = {}
    for i in continuous_idx:
        vals = X[:, :, i][mask > 0]
        stats[i] = (float(vals.mean()), float(vals.std() + 1e-8))
    return stats


def apply_normalization(X: np.ndarray, mask: np.ndarray, stats: Dict[int, Tuple[float, float]]) -> np.ndarray:
    """Apply precomputed (mean, std) normalization to continuous sequence features."""
    X = X.copy()
    for i, (mu, sd) in stats.items():
        X[:, :, i] = (X[:, :, i] - mu) / sd
        X[:, :, i] = X[:, :, i] * mask  # re-zero padding after normalisation
    return X


def extract_lstm_hidden(model: LSTMRegressor, X: np.ndarray, mask: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    """Extract the hidden state at the final real timestep for each sequence.

    Returns an (N, hidden_dim) array suitable for use as a market-state
    embedding downstream (e.g. by the fusion module).
    """
    model.eval()
    n = X.shape[0]
    hidden_dim = model.lstm.hidden_size
    out = np.zeros((n, hidden_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(X[start:end]).float().to(device)
            mb = torch.from_numpy(mask[start:end]).float().to(device)

            lengths = mb.sum(dim=1).clamp(min=1).long()
            lstm_out, _ = model.lstm(xb)  # (B, K, H)

            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, lstm_out.size(-1))
            last = lstm_out.gather(1, idx).squeeze(1)  # (B, H)

            out[start:end] = last.cpu().numpy()

    return out
