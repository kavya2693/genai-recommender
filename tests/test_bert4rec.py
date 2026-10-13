import numpy as np

from genai_recommender import bert4rec
from genai_recommender.metrics import hit_ndcg, mask_seen, target_rank


def test_inference_batch_appends_mask_in_last_slot():
    cfg = bert4rec.Config(max_len=5)
    x = bert4rec.inference_batch([np.array([1, 2, 3, 4, 5, 6]), np.array([7])], cfg, mask_id=99)
    assert x[0].tolist() == [3, 4, 5, 6, 99]
    assert x[1].tolist() == [0, 0, 0, 7, 99]


def test_learns_a_sequential_pattern_popularity_cannot():
    # Every user walks the same 12-item cycle from a random start, so every item is equally
    # popular and only order predicts the next one.
    rng = np.random.default_rng(0)
    n = 12
    seqs = [(rng.integers(n) + np.arange(10)) % n + 1 for _ in range(300)]
    train = [s[:-1] for s in seqs]
    cfg = bert4rec.Config(max_len=10, dim=32, epochs=25, eval_every=25, lr=5e-3, batch_size=64, dropout=0.0)
    model, info = bert4rec.train(train, n, cfg, log=lambda _: None)
    s = mask_seen(bert4rec.score(model, train), train)
    hit, _ = hit_ndcg(target_rank(s, np.array([q[-1] for q in seqs])), k=1)
    assert hit.mean() > 0.9
    assert info["parameters"] > 0
