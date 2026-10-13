"""Loading, cleaning and the leave-last-out split shared by every ranking model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# A prescriber does not open forty pieces of content in one day; an inbox link scanner does.
MAX_EVENTS_PER_DAY = 40


@dataclass
class Tables:
    customers: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    sends: pd.DataFrame
    snippets: pd.DataFrame


def load(data_dir: Path) -> Tables:
    items = pd.read_csv(data_dir / "items.csv", parse_dates=["release_date"])
    return Tables(
        customers=pd.read_csv(data_dir / "customers.csv"),
        items=items.sort_values("item_id").reset_index(drop=True),
        interactions=pd.read_csv(data_dir / "interactions.csv.gz", parse_dates=["ts"]),
        sends=pd.read_csv(data_dir / "sends.csv.gz", parse_dates=["sent_at"]),
        snippets=pd.read_csv(data_dir / "snippets.csv"),
    )


def drop_scanners(interactions: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Remove customers whose busiest day exceeds MAX_EVENTS_PER_DAY.

    Dropping the whole customer rather than the burst day is deliberate: the remaining
    clicks on a scanned inbox are not trustworthy as preference either.
    """
    per_day = interactions.groupby(["customer_id", interactions["ts"].dt.date]).size()
    worst = per_day.groupby(level=0).max()
    bad = sorted(worst[worst > MAX_EVENTS_PER_DAY].index.tolist())
    return interactions[~interactions["customer_id"].isin(bad)], bad


@dataclass
class Split:
    """Leave-last-out: train on all but the last two events, validate on the second to last,
    test on the last with everything before it as history."""

    customer_ids: np.ndarray
    train: list[np.ndarray]
    valid_target: np.ndarray
    test_history: list[np.ndarray]
    test_target: np.ndarray
    test_ts: np.ndarray
    n_items: int
    meta: dict[str, int] = field(default_factory=dict)


def make_split(interactions: pd.DataFrame, n_items: int, min_len: int = 3) -> Split:
    df = interactions.sort_values(["customer_id", "ts"], kind="stable")
    cids, train, valid, hist, test, ts = [], [], [], [], [], []
    for cid, g in df.groupby("customer_id", sort=True):
        seq = g["item_id"].to_numpy()
        if len(seq) < min_len:
            continue
        cids.append(cid)
        train.append(seq[:-2])
        valid.append(seq[-2])
        hist.append(seq[:-1])
        test.append(seq[-1])
        ts.append(g["ts"].iloc[-1])
    return Split(
        customer_ids=np.array(cids),
        train=train,
        valid_target=np.array(valid),
        test_history=hist,
        test_target=np.array(test),
        test_ts=np.array(ts, dtype="datetime64[ns]"),
        n_items=n_items,
        meta=dict(users=len(cids), train_events=int(sum(len(s) for s in train))),
    )


def valid_history(split: Split) -> list[np.ndarray]:
    return split.train
