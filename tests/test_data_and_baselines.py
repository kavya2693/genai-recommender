import numpy as np
import pandas as pd

from genai_recommender.baselines import ItemKNN, LastItemMarkov, Popularity
from genai_recommender.data import MAX_EVENTS_PER_DAY, drop_scanners, load, make_split


def _events(rows):
    return pd.DataFrame(rows, columns=["customer_id", "item_id", "ts"]).assign(
        ts=lambda d: pd.to_datetime(d.ts)
    )


def test_scanner_burst_drops_the_whole_customer():
    normal = [(1, i, f"2025-01-0{1 + i % 5}") for i in range(1, 8)]
    burst = [(2, i, "2025-03-01") for i in range(1, MAX_EVENTS_PER_DAY + 5)]
    clean, bad = drop_scanners(_events(normal + burst))
    assert bad == [2]
    assert set(clean["customer_id"]) == {1}


def test_leave_last_out_split_never_trains_on_targets():
    order = [4, 2, 9, 7, 5]
    ev = _events([(1, i, f"2025-01-{d:02d}") for d, i in enumerate(order, start=1)] + [(2, 3, "2025-01-01")])
    ev = ev.sample(frac=1.0, random_state=0)  # input order must not matter, only timestamps
    split = make_split(ev, n_items=10)
    assert split.customer_ids.tolist() == [1]  # customer 2 is too short
    assert split.train[0].tolist() == [4, 2, 9]
    assert split.valid_target[0] == 7 and split.test_target[0] == 5
    assert split.test_history[0].tolist() == [4, 2, 9, 7]


def test_baselines_on_a_deterministic_chain():
    seqs = [np.array([1, 2, 3, 4])] * 20 + [np.array([5, 1])]
    markov = LastItemMarkov().fit(seqs, 5)
    assert markov.score([np.array([2])])[0].argmax() == 3
    pop = Popularity().fit(seqs, 5)
    assert pop.score([np.array([1])])[0][1:].argmax() + 1 == 1
    knn = ItemKNN(neighbours=3).fit(seqs, 5)
    assert knn.score([np.array([5])])[0][1:].argmax() + 1 == 1


def test_generator_plants_series_and_scanners(tiny_data):
    t = load(tiny_data)
    assert (t.items["series_id"] >= 0).sum() > 50
    _, bad = drop_scanners(t.interactions)
    src = pd.read_csv(tiny_data / "truth" / "interaction_source.csv.gz")
    bots = set(src.loc[src["source"] == "bot", "customer_id"])
    assert set(bad) == bots
