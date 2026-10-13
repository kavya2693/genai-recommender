"""Train every ranker on the same split and score the held-out last item, overall and by the
kind of step it was (series continuation, resumed series, format pathway, free choice) and by
whether the target was new content."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn

from . import bert4rec
from .baselines import ItemKNN, LastItemMarkov, Popularity
from .data import Split
from .metrics import coverage, hit_ndcg, mask_seen, target_rank

NEW_CONTENT_DAYS = 30


@dataclass
class RankingResult:
    table: pd.DataFrame
    by_segment: pd.DataFrame
    model: bert4rec.BERT4Rec
    train_info: dict[str, object]
    scorers: dict[str, Callable[[list[np.ndarray]], np.ndarray]]


def test_segments(split: Split, items: pd.DataFrame, truth_dir: Path | None) -> pd.DataFrame:
    """Labels for slicing the test set. The step type comes from the generator's truth file and
    is used only to slice results, never as a feature."""
    rel = items.set_index("item_id")["release_date"]
    age = (pd.to_datetime(split.test_ts) - rel.loc[split.test_target].to_numpy()).days
    seg = pd.DataFrame(
        dict(customer_id=split.customer_ids, item_id=split.test_target, new_content=age < NEW_CONTENT_DAYS)
    )
    seg["history_len"] = [len(h) for h in split.test_history]
    if truth_dir is not None and (truth_dir / "interaction_source.csv.gz").exists():
        src = pd.read_csv(truth_dir / "interaction_source.csv.gz")
        last = src.groupby("customer_id").tail(1).set_index("customer_id")["source"]
        seg["step"] = last.reindex(split.customer_ids).fillna("unknown").to_numpy()
    else:
        seg["step"] = "unknown"
    return seg


def _evaluate(scores: np.ndarray, split: Split, histories: list[np.ndarray], targets: np.ndarray) -> dict:
    s = mask_seen(scores, histories)
    hit, ndcg = hit_ndcg(target_rank(s, targets))
    return dict(hit=hit, ndcg=ndcg, coverage=coverage(s[:, 1:], split.n_items))


def run(
    split: Split,
    items: pd.DataFrame,
    cfg: bert4rec.Config,
    truth_dir: Path | None = None,
    log: Callable[[str], None] = print,
    valid_users: int = 3000,
) -> RankingResult:
    rng = np.random.default_rng(0)
    vu = rng.choice(len(split.train), min(valid_users, len(split.train)), replace=False)
    v_hist = [split.train[i] for i in vu]
    v_tgt = split.valid_target[vu]
    keep = np.array([len(h) > 0 for h in v_hist])
    v_hist = [h for h, k in zip(v_hist, keep, strict=True) if k]
    v_tgt = v_tgt[keep]

    def validate(m: nn.Module) -> float:
        s = mask_seen(bert4rec.score(m, v_hist), v_hist)
        return float(hit_ndcg(target_rank(s, v_tgt))[1].mean())

    pop = Popularity().fit(split.train, split.n_items)
    knn = ItemKNN().fit(split.train, split.n_items)
    markov = LastItemMarkov().fit(split.train, split.n_items)
    log("training BERT4Rec")
    model, info = bert4rec.train(split.train, split.n_items, cfg, validate=validate, log=log)

    scorers: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
        "popularity": pop.score,
        "item-kNN": knn.score,
        "last-item Markov": markov.score,
        "BERT4Rec": lambda h: bert4rec.score(model, h),
    }
    seg = test_segments(split, items, truth_dir)
    rows, seg_rows = [], []
    for name, fn in scorers.items():
        r = _evaluate(fn(split.test_history), split, split.test_history, split.test_target)
        rows.append(
            {
                "model": name,
                "HR@10": r["hit"].mean(),
                "NDCG@10": r["ndcg"].mean(),
                "coverage@10": r["coverage"],
            }
        )
        frame = seg.assign(hit=r["hit"], ndcg=r["ndcg"])
        for col in ("step", "new_content"):
            for key, g in frame.groupby(col):
                seg_rows.append(
                    {
                        "model": name,
                        "slice": f"{col}={key}",
                        "users": len(g),
                        "HR@10": g["hit"].mean(),
                        "NDCG@10": g["ndcg"].mean(),
                    }
                )
    return RankingResult(
        table=pd.DataFrame(rows),
        by_segment=pd.DataFrame(seg_rows),
        model=model,
        train_info=info,
        scorers=scorers,
    )
