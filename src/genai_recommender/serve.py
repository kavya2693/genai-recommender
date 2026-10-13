"""The request path: one customer in, one recommendation with channel and draft out.

Stages, each timed separately so a slow one can be found: rank the catalogue with BERT4Rec, pick
the top unseen item, score the three channels, retrieve claims and draft the message.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from torch import nn

from . import bert4rec
from .channel import ChannelModel, eligible_channels
from .datagen import CHANNELS
from .metrics import mask_seen
from .rag import Drafter


@dataclass
class Recommendation:
    customer_id: int
    top10: list[int]
    item_id: int
    channel: str
    channel_p: dict[str, float]
    draft: dict[str, object]
    ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Recommender:
    ranker: nn.Module
    channel_model: ChannelModel
    drafter: Drafter
    customers: pd.DataFrame
    items: pd.DataFrame

    def __post_init__(self) -> None:
        self._cust = self.customers.set_index("customer_id")
        self._items = self.items.set_index("item_id")

    def recommend(
        self, customer_id: int, history: np.ndarray, day: int, recent_touches: int = 0
    ) -> Recommendation:
        ms: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            ms[name] = 1000 * (now - t)
            t = now

        scores = mask_seen(bert4rec.score(self.ranker, [history]), [history])[0]
        top10 = np.argsort(-scores)[:10]
        lap("rank")

        item_id = int(top10[0])
        rows = pd.DataFrame(
            dict(
                customer_id=[customer_id] * 3,
                item_id=[item_id] * 3,
                day=[day] * 3,
                channel=CHANNELS,
                recent_touches=[recent_touches] * 3,
            )
        )
        p = self.channel_model.predict(rows)
        elig = eligible_channels(self.customers, np.array([customer_id]))[0]
        # The per-request path has no rep budget to share, so it takes the eligible argmax;
        # batch planning uses assign_with_rep_budget instead.
        best = int(np.argmax(np.where(elig, p, -np.inf)))
        lap("channel")

        item = self._items.loc[item_id]
        assert isinstance(item, pd.Series)
        draft = self.drafter.draft(str(self._cust.loc[customer_id, "specialty"]), item, CHANNELS[best])
        lap("retrieve+draft")
        ms["total"] = sum(ms.values())
        return Recommendation(
            customer_id=customer_id,
            top10=[int(i) for i in top10],
            item_id=item_id,
            channel=CHANNELS[best],
            channel_p={c: float(v) for c, v in zip(CHANNELS, p, strict=True)},
            draft=draft,
            ms=ms,
        )


def latency_table(
    fn: Callable[[int], Recommendation], n: int, warmup: int = 20
) -> tuple[pd.DataFrame, list[Recommendation]]:
    for i in range(min(warmup, n)):
        fn(i)
    recs = [fn(i) for i in range(n)]
    frame = pd.DataFrame([r.ms for r in recs])
    table = pd.DataFrame(
        {
            "stage": frame.columns,
            "p50 ms": [float(np.percentile(frame[c], 50)) for c in frame.columns],
            "p95 ms": [float(np.percentile(frame[c], 95)) for c in frame.columns],
            "max ms": [float(frame[c].max()) for c in frame.columns],
            "requests": n,
        }
    )
    return table, recs
