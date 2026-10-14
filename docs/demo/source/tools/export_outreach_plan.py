"""Export one territory's outreach plan from the repo's own pipeline, for slide 07 of the demo video.

Re-runs exactly what `make run` does up to the request path (same data/, seed 7 BERT4Rec config,
same channel model and approved-only claim index, stub drafter), then plans one touch per
prescriber in a single region:

- next content: `Recommender.recommend()` (BERT4Rec top unseen item), as in cli.py's timed requests
- channel: `channel.assign_with_rep_budget()` over the three calibrated probabilities, with the rep
  budget set to the historical rep share of logged sends (the batch policy the README scores)
- message: `rag.Drafter` with the deterministic stub, and its grounding check

Usage (from the repo root):
  PYTHONPATH=src .venv/bin/python docs/demo/source/tools/export_outreach_plan.py \
      docs/demo/source/assets/outreach_plan.json --region north
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from genai_recommender import bert4rec, channel, data, datagen, rag, ranking
from genai_recommender.serve import Recommender

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--data", type=Path, default=Path("data"))
ap.add_argument("--region", default="north")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

t0 = time.perf_counter()
torch.set_num_threads(a.threads)
t = data.load(a.data)
truth = a.data / "truth"
inter, scanners = data.drop_scanners(t.interactions)
split = data.make_split(inter, int(t.items["item_id"].max()))
rr = ranking.run(split, t.items, bert4rec.Config(), truth_dir=truth if truth.exists() else None)
cr = channel.run(t.customers, t.items, t.sends, None)
index = rag.SnippetIndex().fit(t.snippets)
rec = Recommender(rr.model, cr.model, rag.Drafter(index, rag.StubLLM()), t.customers, t.items)

# historical rep share of logged sends (the same budget rule as the README's policy table)
sends = channel.prepare_sends(t.sends)
_, _, test = channel.time_split(sends)
rep_share = float((test["channel"] == "rep").mean())

touch_days = {int(str(c)): g["day"].to_numpy() for c, g in sends.groupby("customer_id")}
days = ((pd.to_datetime(split.test_ts) - datagen.START).days).to_numpy()
cust = t.customers.set_index("customer_id")
users = [u for u, cid in enumerate(split.customer_ids) if cust.loc[int(cid), "region"] == a.region]

torch.set_num_threads(1)
recs = []
with threadpool_limits(limits=1):
    for u in users:
        cid, day = int(split.customer_ids[u]), int(days[u])
        past = touch_days.get(cid, np.array([]))
        recent = int(((past < day) & (past >= day - 7)).sum())
        recs.append(rec.recommend(cid, split.test_history[u], day, recent))

ids = np.array([r.customer_id for r in recs])
p = np.array([[r.channel_p[c] for c in datagen.CHANNELS] for r in recs])
elig = channel.eligible_channels(t.customers, ids)
budget = round(rep_share * len(recs))
choice = channel.assign_with_rep_budget(p, elig, budget)
items = t.items.set_index("item_id")

rows = []
for r, ch, pe, el in zip(recs, choice, p, elig, strict=True):
    item = items.loc[r.item_id]
    chan = datagen.CHANNELS[int(ch)]
    d = rec.drafter.draft(str(cust.loc[r.customer_id, "specialty"]), item, chan)
    digital = [pe[k] for k in (0, 1) if el[k]]
    rows.append(
        {
            "customer_id": r.customer_id,
            "specialty": str(cust.loc[r.customer_id, "specialty"]),
            "rep_covered": bool(cust.loc[r.customer_id, "rep_covered"]),
            "item_id": r.item_id,
            "title": str(item["title"]),
            "brand": str(item["brand"]),
            "format": str(item["format"]),
            "channel": chan,
            "request_channel": r.channel,
            "p": {c: round(float(v), 4) for c, v in zip(datagen.CHANNELS, pe, strict=True)},
            "p_chosen": round(float(pe[int(ch)]), 4),
            "rep_gain": round(float(pe[2] - max(digital)), 4) if (el[2] and digital) else None,
            "snippet_ids": [int(s) for s in d["snippet_ids"]],
            "draft": str(d["draft"]),
            "numbers": int(d["numbers"]),  # type: ignore[call-overload]
            "grounded": bool(d["grounded"]),
        }
    )

counts = pd.Series([r["channel"] for r in rows]).value_counts().to_dict()
out = {
    "region": a.region,
    "prescribers": len(rows),
    "rep_share_historical": round(rep_share, 4),
    "rep_budget": int(budget),
    "channels": {c: int(counts.get(c, 0)) for c in datagen.CHANNELS},
    "grounded": int(sum(r["grounded"] for r in rows)),
    "mean_p_chosen": round(float(np.mean([r["p_chosen"] for r in rows])), 4),
    "ranking_check": rr.table.round(4).to_dict(orient="records"),
    "bert4rec_best_epoch": rr.train_info["best_epoch"],
    "seconds": round(time.perf_counter() - t0, 1),
    "rows": rows,
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, separators=(",", ":")))
print(
    f"{len(rows)} prescribers in {a.region}: {out['channels']}, rep budget {budget}, "
    f"grounded {out['grounded']}/{len(rows)}, {out['seconds']} s"
)
