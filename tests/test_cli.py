import json

from genai_recommender import cli


def test_end_to_end_quick_run(tiny_data, tmp_path):
    cli.main(
        ["--quick", "--data", str(tiny_data), "--out", str(tmp_path), "--latency-n", "20", "--threads", "2"]
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    models = {r["model"] for r in summary["ranking"]}
    assert models == {"popularity", "item-kNN", "last-item Markov", "BERT4Rec"}
    stages = {r["stage"] for r in summary["latency"]}
    assert {"rank", "channel", "retrieve+draft", "total"} <= stages
    assert summary["stub_grounded_share"] == 1.0
    for f in ["ranking.csv", "channel_policy.csv", "retrieval.csv", "latency.csv", "drafts_stub.md"]:
        assert (tmp_path / f).exists()
