"""Channel selection: given a customer and the piece of content picked for them, which of email,
SMS or a rep visit is most likely to be engaged with.

The model is a gradient-boosted classifier over (customer, content, channel) features, fitted on
logged sends and then recalibrated with isotonic regression on a later slice of time. Calibration
matters here because the policy compares probabilities across channels and spends a scarce
resource (rep visits) on the largest gaps; a model that ranks well but is overconfident on one
channel would send reps to the wrong doors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .datagen import CHANNELS, START, engagement_logit
from .metrics import expected_calibration_error

# Pseudo-count for the per-channel engagement history: a customer with two sends and one open
# should not look like a 50% responder.
PRIOR_STRENGTH = 5.0
CATEGORICAL = ["specialty", "region", "format", "therapeutic_area", "channel"]
NUMERIC = [
    "years_in_practice",
    "digital_affinity",
    "rep_covered",
    "email_opt_in",
    "sms_opt_in",
    "area_match",
    "weekday",
    "recent_touches",
    "prior_n_ch",
    "prior_rate_ch",
    "prior_n_all",
    "prior_rate_all",
]
FEATURES = CATEGORICAL + NUMERIC


def _history_tables(sends: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Running per-customer send and engagement counts at the end of each day, per channel and
    overall. Looked up strictly before a send's day, so an outcome is never its own feature."""
    out = {}
    for key in [*CHANNELS, "all"]:
        s = sends if key == "all" else sends[sends["channel"] == key]
        daily = s.groupby(["customer_id", "day"], as_index=False).agg(
            n=("engaged", "size"), e=("engaged", "sum")
        )
        daily["cum_n"] = daily.groupby("customer_id")["n"].cumsum()
        daily["cum_e"] = daily.groupby("customer_id")["e"].cumsum()
        out[key] = daily[["customer_id", "day", "cum_n", "cum_e"]].sort_values("day")
    return out


def _lookup(rows: pd.DataFrame, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = rows[["customer_id", "day"]].reset_index().sort_values("day")
    m = pd.merge_asof(q, table, on="day", by="customer_id", allow_exact_matches=False)
    m = m.set_index("index").reindex(rows.index)
    return m["cum_n"].fillna(0).to_numpy(), m["cum_e"].fillna(0).to_numpy()


@dataclass
class FeatureBuilder:
    customers: pd.DataFrame
    items: pd.DataFrame
    history: dict[str, pd.DataFrame]
    base_rate: dict[str, float]

    @classmethod
    def fit(
        cls, customers: pd.DataFrame, items: pd.DataFrame, sends: pd.DataFrame, train: pd.DataFrame
    ) -> FeatureBuilder:
        """History tables span every send (lookups are strictly earlier than the send being
        scored); the smoothing base rates come from the training window only."""
        base = {str(k): float(v) for k, v in train.groupby("channel")["engaged"].mean().items()}
        base["all"] = float(train["engaged"].mean())
        return cls(
            customers.set_index("customer_id"), items.set_index("item_id"), _history_tables(sends), base
        )

    def build(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: customer_id, item_id, day, channel, recent_touches."""
        c = self.customers.loc[rows["customer_id"]].reset_index(drop=True)
        it = self.items.loc[rows["item_id"]].reset_index(drop=True)
        r = rows.reset_index(drop=True)
        x = pd.DataFrame(
            dict(
                specialty=c["specialty"],
                region=c["region"].fillna("unknown"),
                format=it["format"],
                therapeutic_area=it["therapeutic_area"],
                channel=r["channel"],
                years_in_practice=c["years_in_practice"],
                digital_affinity=c["digital_affinity"],
                rep_covered=c["rep_covered"].astype(int),
                email_opt_in=c["email_opt_in"].astype(int),
                sms_opt_in=c["sms_opt_in"].astype(int),
                area_match=(c["specialty"] == it["therapeutic_area"]).astype(int),
                weekday=(START + pd.to_timedelta(r["day"], unit="D")).dt.dayofweek,
                recent_touches=r["recent_touches"],
            )
        )
        n_all, e_all = _lookup(r, self.history["all"])
        n_ch, e_ch = np.zeros(len(r)), np.zeros(len(r))
        prior_ch = np.zeros(len(r))
        for ch in CHANNELS:
            m = (r["channel"] == ch).to_numpy()
            if m.any():
                n, e = _lookup(r[m], self.history[ch])
                n_ch[m], e_ch[m] = n, e
                prior_ch[m] = self.base_rate.get(ch, self.base_rate["all"])
        x["prior_n_ch"] = n_ch
        x["prior_rate_ch"] = (e_ch + PRIOR_STRENGTH * prior_ch) / (n_ch + PRIOR_STRENGTH)
        x["prior_n_all"] = n_all
        x["prior_rate_all"] = (e_all + PRIOR_STRENGTH * self.base_rate["all"]) / (n_all + PRIOR_STRENGTH)
        for col in CATEGORICAL:
            x[col] = x[col].astype("category")
        return x[FEATURES]


class ChannelModel:
    def __init__(self, builder: FeatureBuilder, seed: int = 0) -> None:
        self.builder = builder
        self.gbm = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            categorical_features="from_dtype",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.categories: dict[str, pd.Index] = {}

    def _align(self, x: pd.DataFrame) -> pd.DataFrame:
        # Single-row requests must see the same category codes as training.
        x = x.copy()
        for col, cats in self.categories.items():
            x[col] = pd.Categorical(x[col].astype(str), categories=cats)
        return x

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame) -> ChannelModel:
        x = self.builder.build(train)
        self.categories = {c: x[c].cat.categories for c in CATEGORICAL}
        self.gbm.fit(x, train["engaged"].to_numpy())
        self.iso.fit(self.raw(calib), calib["engaged"].to_numpy())
        return self

    def raw(self, rows: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict_proba(self._align(self.builder.build(rows)))[:, 1]

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.raw(rows))


def time_split(
    sends: pd.DataFrame, train_end: float = 0.7, calib_end: float = 0.8
) -> tuple[pd.DataFrame, ...]:
    d0, d1 = sends["day"].min(), sends["day"].max()
    a = d0 + train_end * (d1 - d0)
    b = d0 + calib_end * (d1 - d0)
    return sends[sends["day"] < a], sends[(sends["day"] >= a) & (sends["day"] < b)], sends[sends["day"] >= b]


def prepare_sends(sends: pd.DataFrame) -> pd.DataFrame:
    s = sends.copy()
    s["day"] = (s["sent_at"].dt.normalize() - START).dt.days
    return s.sort_values(["day", "send_id"]).reset_index(drop=True)


def eligible_channels(customers: pd.DataFrame, customer_ids: np.ndarray) -> np.ndarray:
    c = customers.set_index("customer_id").loc[customer_ids]
    elig = np.stack(
        [c["email_opt_in"].to_numpy(), c["sms_opt_in"].to_numpy(), c["rep_covered"].to_numpy()], axis=1
    ).astype(bool)
    elig[~elig.any(axis=1), 0] = True  # transactional email is always allowed
    return elig


def assign_with_rep_budget(p: np.ndarray, elig: np.ndarray, rep_budget: int) -> np.ndarray:
    """Pick a channel per touch, with at most rep_budget rep visits.

    Reps are a fixed headcount, so an unconstrained argmax (which sends a rep wherever one is
    allowed, because visits engage best on average) is not a policy anyone could run. Visits go
    to the touches where the rep beats the best digital channel by the widest margin.
    """
    rep = CHANNELS.index("rep")
    digital = np.where(elig[:, :rep], p[:, :rep], -np.inf)
    best_digital = digital.argmax(axis=1)
    best_digital_p = digital.max(axis=1)
    gain = np.where(
        elig[:, rep], p[:, rep] - np.where(np.isfinite(best_digital_p), best_digital_p, 0.0), -np.inf
    )
    choice = best_digital.copy()
    no_digital = ~np.isfinite(best_digital_p)
    choice[no_digital] = rep
    budget = max(rep_budget - int(no_digital.sum()), 0)
    order = np.argsort(-gain)
    take = order[:budget]
    take = take[np.isfinite(gain[take]) & (gain[take] > 0)]
    choice[take] = rep
    return choice


@dataclass
class ChannelResult:
    model: ChannelModel
    calibration: pd.DataFrame
    policy: pd.DataFrame
    reliability: pd.DataFrame


def _all_channel_rows(test: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([test.assign(channel=ch) for ch in CHANNELS], ignore_index=True)


def run(
    customers: pd.DataFrame,
    items: pd.DataFrame,
    sends_raw: pd.DataFrame,
    latent: pd.DataFrame | None,
) -> ChannelResult:
    sends = prepare_sends(sends_raw)
    train, calib, test = time_split(sends)
    builder = FeatureBuilder.fit(customers, items, sends, train)
    model = ChannelModel(builder).fit(train, calib)
    y = test["engaged"].to_numpy()

    # A linear baseline on the same features, one-hot encoded.
    xtr = pd.get_dummies(builder.build(train), columns=CATEGORICAL, dummy_na=True).fillna(0.0)
    xte = pd.get_dummies(builder.build(test), columns=CATEGORICAL, dummy_na=True).fillna(0.0)
    xte = xte.reindex(columns=xtr.columns, fill_value=0.0)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1.0)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit((xtr - mu) / sd, train["engaged"].to_numpy())
    p_lr = lr.predict_proba((xte - mu) / sd)[:, 1]

    p_raw = model.raw(test)
    p_cal = model.iso.predict(p_raw)
    cal_rows = []
    for name, p in [("logistic regression", p_lr), ("GBM, raw", p_raw), ("GBM + isotonic", p_cal)]:
        cal_rows.append(
            {
                "model": name,
                "AUC": roc_auc_score(y, p),
                "log loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                "Brier": brier_score_loss(y, p),
                "ECE": expected_calibration_error(p, y),
            }
        )
    rel = pd.DataFrame(dict(p=p_cal, y=y, channel=test["channel"].to_numpy()))
    rel["bin"] = pd.qcut(rel["p"].rank(method="first"), 10, labels=False)
    reliability = rel.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))

    policy = pd.DataFrame()
    if latent is not None:
        policy = _policy_eval(model, customers, items, test, latent)
    return ChannelResult(model, pd.DataFrame(cal_rows), policy, reliability.reset_index())


def _policy_eval(
    model: ChannelModel,
    customers: pd.DataFrame,
    items: pd.DataFrame,
    test: pd.DataFrame,
    latent: pd.DataFrame,
) -> pd.DataFrame:
    """Score each policy by the generator's true engagement probability for the channel it
    picks. The truth is only read here, never by the model."""
    n = len(test)
    elig = eligible_channels(customers, test["customer_id"].to_numpy())
    allrows = _all_channel_rows(test)
    p_hat = model.predict(allrows).reshape(len(CHANNELS), n).T

    cidx = customers.reset_index(drop=True).reset_index().set_index("customer_id")["index"]
    iidx = items.reset_index(drop=True).reset_index().set_index("item_id")["index"]
    lat = latent.set_index("customer_id").loc[customers["customer_id"]].reset_index()
    true_logit = engagement_logit(
        lat,
        customers.reset_index(drop=True),
        items.reset_index(drop=True),
        cidx.loc[allrows["customer_id"]].to_numpy(),
        iidx.loc[allrows["item_id"]].to_numpy(),
        allrows["channel"].to_numpy(),
        allrows["day"].to_numpy(),
        allrows["recent_touches"].to_numpy(),
    )
    p_true = (1 / (1 + np.exp(-true_logit))).reshape(len(CHANNELS), n).T

    logged = test["channel"].map({c: i for i, c in enumerate(CHANNELS)}).to_numpy()
    rep_budget = int((logged == CHANNELS.index("rep")).sum())
    email_first = np.where(elig[:, 0], 0, elig.argmax(axis=1))
    choices = {
        "logged (historical) policy": logged,
        "email unless opted out": email_first,
        "model, same rep budget": assign_with_rep_budget(p_hat, elig, rep_budget),
        "oracle, same rep budget": assign_with_rep_budget(p_true, elig, rep_budget),
    }
    rows = []
    for name, ch in choices.items():
        val = p_true[np.arange(n), ch]
        rows.append(
            {
                "policy": name,
                "expected engagement": val.mean(),
                "engaged per 1k touches": 1000 * val.mean(),
                "rep share": (ch == 2).mean(),
                "sms share": (ch == 1).mean(),
            }
        )
    out = pd.DataFrame(rows)
    base = float(out["expected engagement"].iloc[0])
    oracle = float(out["expected engagement"].iloc[3])
    out["share of oracle lift"] = (out["expected engagement"] - base) / (oracle - base)
    return out
