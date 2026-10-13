"""Full-catalogue ranking metrics. No negative sampling: sampled metrics flatter every model
and can reorder them, and with ~500 items the full ranking is cheap."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mask_seen(scores: np.ndarray, histories: Sequence[np.ndarray]) -> np.ndarray:
    """Content is rarely consumed twice, so items already in the history are not candidates.
    Column 0 is padding and is never a candidate either."""
    out = scores.astype(np.float64, copy=True)
    out[:, 0] = -np.inf
    for u, h in enumerate(histories):
        out[u, h] = -np.inf
    return out


def target_rank(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target. Ties count against the target, so a constant scorer
    cannot score well by accident."""
    t = scores[np.arange(len(targets)), targets]
    return (scores > t[:, None]).sum(axis=1) + (scores == t[:, None]).sum(axis=1) - 1


def hit_ndcg(rank: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    hit = rank < k
    ndcg = np.where(hit, 1.0 / np.log2(rank + 2.0), 0.0)
    return hit.astype(float), ndcg


def coverage(scores: np.ndarray, n_items: int, k: int = 10) -> float:
    """Share of the catalogue that appears in at least one top-k list."""
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return len(np.unique(top)) / n_items


def top_k(scores: np.ndarray, k: int = 10) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Equal-mass bins, so the sparse high-probability tail does not dominate."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err / len(p))
