import numpy as np
import pandas as pd

from genai_recommender import channel
from genai_recommender.data import load


def test_rep_budget_is_respected_and_goes_to_largest_gain():
    p = np.array([[0.1, 0.0, 0.5], [0.1, 0.0, 0.2], [0.3, 0.0, 0.35], [0.1, 0.0, 0.9]])
    elig = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0]], dtype=bool)
    choice = channel.assign_with_rep_budget(p, elig, rep_budget=1)
    assert choice.tolist() == [2, 0, 0, 0]  # row 3 has the best rep score but no rep coverage


def test_history_features_never_see_the_same_day():
    sends = pd.DataFrame(
        dict(customer_id=[1, 1, 1], day=[10, 10, 12], channel=["email"] * 3, engaged=[1, 1, 0])
    )
    hist = channel._history_tables(sends)
    n, e = channel._lookup(sends[["customer_id", "day"]], hist["email"])
    assert n.tolist() == [0, 0, 2] and e.tolist() == [0, 0, 2]


def test_channel_model_beats_chance_and_is_calibrated(tiny_data):
    t = load(tiny_data)
    lat = pd.read_csv(tiny_data / "truth" / "customer_latent.csv")
    r = channel.run(t.customers, t.items, t.sends, lat)
    gbm = r.calibration.set_index("model").loc["GBM + isotonic"]
    assert gbm["AUC"] > 0.6
    assert gbm["ECE"] < 0.05
    pol = r.policy.set_index("policy")["expected engagement"]
    assert pol["oracle, same rep budget"] >= pol["model, same rep budget"]
