"""End-to-end run: rank, choose channel, retrieve and draft, time the request path.

genai-reco                      full run on data/ (generated if missing), stub drafter
genai-reco --llm ollama         also drafts a few demo messages with llama3.2 via Ollama
genai-reco --quick --refresh --customers 1500    the CI smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import bert4rec, channel, data, datagen, rag, ranking
from .serve import Recommendation, Recommender, latency_table


class Log:
    def __init__(self, path: Path) -> None:
        self.f = path.open("w")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def _recent_touches(sends: pd.DataFrame) -> dict[int, np.ndarray]:
    return {int(str(c)): g["day"].to_numpy() for c, g in sends.groupby("customer_id")}


def _md_drafts(recs: list[Recommendation], items: pd.DataFrame, llm_name: str) -> str:
    title = items.set_index("item_id")["title"]
    out = [f"# Draft samples ({llm_name})", ""]
    for r in recs:
        d = r.draft
        verdict = "pass" if d["grounded"] else f"FAIL, ungrounded numbers {d['ungrounded']}"
        out += [
            f"## Customer {r.customer_id}: {title.loc[r.item_id]} via {r.channel}",
            "",
            f"Channel probabilities: {', '.join(f'{k} {v:.3f}' for k, v in r.channel_p.items())}  ",
            f"Retrieved snippets: {d['snippet_ids']}  ",
            f"Grounding check: {verdict}  ",
            f"Draft latency: {r.ms['retrieve+draft']:.0f} ms",
            "",
            "```text",
            str(d["draft"]),
            "```",
            "",
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Next-content, channel and grounded draft, end to end.")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--refresh", action="store_true", help="regenerate the dataset first")
    ap.add_argument("--customers", type=int, default=datagen.Scale.customers)
    ap.add_argument("--quick", action="store_true", help="a few epochs only, for CI")
    ap.add_argument("--llm", choices=["stub", "ollama"], default="stub")
    ap.add_argument("--demo-n", type=int, default=3, help="drafts written with the real model")
    ap.add_argument("--latency-n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    log = Log(args.out / "run_log.txt")
    torch.set_num_threads(args.threads)
    t_start = time.perf_counter()

    if args.refresh or not (args.data / "interactions.csv.gz").exists():
        log(f"generating data for {args.customers:,} customers")
        counts = datagen.generate(args.data, datagen.Scale(customers=args.customers))
        log("  " + " ".join(f"{k}={v:,}" for k, v in counts.items()))
    t = data.load(args.data)
    truth = args.data / "truth"
    inter, scanners = data.drop_scanners(t.interactions)
    log(f"dropped {len(scanners)} customers with link-scanner bursts")
    split = data.make_split(inter, int(t.items["item_id"].max()))
    log(f"split: {split.meta}")

    # Next-content ranking.
    cfg = bert4rec.Config(epochs=4, eval_every=2) if args.quick else bert4rec.Config()
    rr = ranking.run(split, t.items, cfg, truth_dir=truth if truth.exists() else None, log=log)
    rr.table.to_csv(args.out / "ranking.csv", index=False)
    rr.by_segment.to_csv(args.out / "ranking_by_segment.csv", index=False)
    (args.out / "bert4rec_training.json").write_text(json.dumps(rr.train_info, indent=2))
    log("\nranking, full catalogue, last event held out\n" + rr.table.to_string(index=False))
    seg = rr.by_segment.pivot(index="slice", columns="model", values="NDCG@10")
    log("\nNDCG@10 by slice\n" + seg.round(3).to_string())

    # Channel selection.
    latent = pd.read_csv(truth / "customer_latent.csv") if truth.exists() else None
    cr = channel.run(t.customers, t.items, t.sends, latent)
    cr.calibration.to_csv(args.out / "channel_calibration.csv", index=False)
    cr.policy.to_csv(args.out / "channel_policy.csv", index=False)
    cr.reliability.to_csv(args.out / "channel_reliability.csv", index=False)
    log("\nchannel model, last 20% of sends by time\n" + cr.calibration.to_string(index=False))
    if len(cr.policy):
        log("\nchannel policy, scored by true engagement probability\n" + cr.policy.to_string(index=False))

    # Retrieval.
    ret = rag.retrieval_eval(t.snippets, t.items)
    ret.to_csv(args.out / "retrieval.csv", index=False)
    log("\nclaim retrieval, k=3\n" + ret.to_string(index=False))

    # The request path, timed one request at a time on one thread.
    torch.set_num_threads(1)
    index = rag.SnippetIndex().fit(t.snippets)
    sends = channel.prepare_sends(t.sends)
    touch_days = _recent_touches(sends)
    rng = np.random.default_rng(1)
    users = rng.choice(len(split.customer_ids), min(args.latency_n, len(split.customer_ids)), replace=False)
    days = ((pd.to_datetime(split.test_ts) - datagen.START).days).to_numpy()

    def request(rec: Recommender, i: int) -> Recommendation:
        u = int(users[i % len(users)])
        cid, day = int(split.customer_ids[u]), int(days[u])
        past = touch_days.get(cid, np.array([]))
        recent = int(((past < day) & (past >= day - 7)).sum())
        return rec.recommend(cid, split.test_history[u], day, recent)

    stub = Recommender(rr.model, cr.model, rag.Drafter(index, rag.StubLLM()), t.customers, t.items)
    # One OpenMP thread per request: on a busy host the GBM's thread pool start-up alone cost
    # ~50 ms of a 3-row prediction that takes 4 ms single-threaded.
    with threadpool_limits(limits=1):
        lat, recs = latency_table(lambda i: request(stub, i), len(users))
    lat_all = lat
    lat_all.to_csv(args.out / "latency.csv", index=False)
    log(
        f"\nrequest latency, {len(users)} requests, 1 thread, stub drafter\n"
        + lat_all.round(2).to_string(index=False)
    )
    (args.out / "drafts_stub.md").write_text(_md_drafts(recs[:5], t.items, "stub"))
    grounded = float(np.mean([bool(r.draft["grounded"]) for r in recs]))
    log(f"stub drafts passing the grounding check: {grounded:.1%}")

    llm_summary: dict[str, object] = {}
    if args.llm == "ollama":
        llm = rag.OllamaLLM()
        torch.set_num_threads(args.threads)
        live = Recommender(rr.model, cr.model, rag.Drafter(index, llm), t.customers, t.items)
        demo = [request(live, i) for i in range(args.demo_n)]
        (args.out / f"drafts_{llm.model.replace(':', '_')}.md").write_text(
            _md_drafts(demo, t.items, llm.model)
        )
        ms = [r.ms["retrieve+draft"] for r in demo]
        llm_summary = {
            "model": llm.model,
            "samples": len(demo),
            "grounded": int(sum(bool(r.draft["grounded"]) for r in demo)),
            "draft_ms": [round(m) for m in ms],
        }
        log(f"\n{llm.model} demo drafts: {llm_summary}")

    summary = {
        "ranking": rr.table.round(4).to_dict(orient="records"),
        "bert4rec": {k: v for k, v in rr.train_info.items() if k != "history"},
        "channel_calibration": cr.calibration.round(4).to_dict(orient="records"),
        "channel_policy": cr.policy.round(4).to_dict(orient="records"),
        "retrieval": ret.round(4).to_dict(orient="records"),
        "latency": lat_all.round(2).to_dict(orient="records"),
        "stub_grounded_share": grounded,
        "llm_demo": llm_summary,
        "split": split.meta,
        "scanners_dropped": len(scanners),
        "quick": args.quick,
        "config": asdict(cfg),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"\ndone in {summary['wall_seconds']}s; outputs in {args.out}/")


if __name__ == "__main__":
    main()
