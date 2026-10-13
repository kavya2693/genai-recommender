import numpy as np

from genai_recommender.metrics import expected_calibration_error, hit_ndcg, mask_seen, target_rank


def test_constant_scorer_cannot_score():
    # Ties count against the target: a model that outputs the same score everywhere ranks last.
    scores = np.ones((4, 20))
    rank = target_rank(scores, np.array([1, 5, 9, 19]))
    hit, ndcg = hit_ndcg(rank, k=10)
    assert (rank == 19).all()
    assert hit.sum() == 0 and ndcg.sum() == 0


def test_perfect_scorer_gets_full_marks():
    targets = np.array([3, 7])
    scores = np.zeros((2, 10))
    scores[[0, 1], targets] = 1.0
    hit, ndcg = hit_ndcg(target_rank(scores, targets))
    assert hit.tolist() == [1.0, 1.0]
    assert np.allclose(ndcg, 1.0)


def test_mask_seen_removes_history_and_padding():
    s = mask_seen(np.ones((2, 6)), [np.array([1, 2]), np.array([5])])
    assert np.isinf(s[0, [0, 1, 2]]).all() and np.isfinite(s[0, 3:]).all()
    assert np.isinf(s[1, [0, 5]]).all()


def test_ece_separates_calibrated_from_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.01
    assert expected_calibration_error(np.clip(p * 1.6, 0, 1), y) > 0.1
