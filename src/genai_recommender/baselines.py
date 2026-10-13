"""Three baselines the sequential model has to beat, from weakest assumption to strongest:
what is popular, what co-occurs with your history, and what usually follows your last item."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse


class Popularity:
    def fit(self, sequences: Sequence[np.ndarray], n_items: int) -> Popularity:
        self.counts = np.bincount(np.concatenate(sequences), minlength=n_items + 1).astype(float)
        return self

    def score(self, histories: Sequence[np.ndarray]) -> np.ndarray:
        return np.tile(self.counts, (len(histories), 1))


class ItemKNN:
    """Cosine item-item similarity on the binary user x item matrix; order is ignored."""

    def __init__(self, neighbours: int = 100) -> None:
        self.neighbours = neighbours

    def fit(self, sequences: Sequence[np.ndarray], n_items: int) -> ItemKNN:
        rows = np.repeat(np.arange(len(sequences)), [len(s) for s in sequences])
        cols = np.concatenate(sequences)
        x = sparse.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(sequences), n_items + 1))
        x.data[:] = 1.0
        co = (x.T @ x).toarray()
        norm = np.sqrt(np.diag(co)) + 1e-9
        sim = co / norm[:, None] / norm[None, :]
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < n_items:
            cut = -np.partition(-sim, self.neighbours, axis=1)[:, self.neighbours][:, None]
            sim = np.where(sim >= cut, sim, 0.0)
        self.sim = sim
        self.pop = np.bincount(cols, minlength=n_items + 1) / len(cols)
        return self

    def score(self, histories: Sequence[np.ndarray]) -> np.ndarray:
        out = np.empty((len(histories), self.sim.shape[0]))
        for u, h in enumerate(histories):
            out[u] = self.sim[h].sum(axis=0)
        # Popularity breaks ties among items with no co-occurrence signal.
        return out + 1e-3 * self.pop


class LastItemMarkov:
    """First-order transition counts from the last item. If this matches the transformer,
    the transformer is not using the rest of the sequence."""

    def fit(self, sequences: Sequence[np.ndarray], n_items: int) -> LastItemMarkov:
        a = np.concatenate([s[:-1] for s in sequences if len(s) > 1])
        b = np.concatenate([s[1:] for s in sequences if len(s) > 1])
        t = np.zeros((n_items + 1, n_items + 1))
        np.add.at(t, (a, b), 1.0)
        self.trans = t / np.maximum(t.sum(axis=1, keepdims=True), 1.0)
        pop = np.bincount(np.concatenate(sequences), minlength=n_items + 1).astype(float)
        self.pop = pop / pop.sum()
        return self

    def score(self, histories: Sequence[np.ndarray]) -> np.ndarray:
        last = np.array([h[-1] for h in histories])
        return self.trans[last] + 1e-3 * self.pop
