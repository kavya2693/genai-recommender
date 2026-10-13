"""Seeded generator for the content-engagement dataset.

Five tables, sized to a commercial team that serves ~20,000 prescribers with ~500 pieces of
approved content over two years:

customers     who the prescriber is and which channels they can be reached on
items         the content catalogue, including multi-part series and congress recaps
interactions  what each customer consumed, in order, through which channel
sends         outbound email / SMS / rep touches and whether each one was engaged with
snippets      approved (and expired / draft) claim text that a message may quote

The behaviour that the models have to find is planted on purpose and kept partly hidden:
series continuation, format pathways within a brand, specialty preference, a launch boost
for new content, weekday and holiday seasonality, contact fatigue and per-customer channel
affinities that are only partly visible through observed features. The hidden parts are
written to ``truth/`` so evaluation can score a policy against known engagement
probabilities; no model reads that folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPECIALTIES = ["oncology", "cardiology", "immunology", "neurology", "hematology", "primary_care"]
SPECIALTY_WEIGHTS = [0.18, 0.2, 0.14, 0.12, 0.08, 0.28]
AREAS = SPECIALTIES[:5]
# Two invented brands per therapeutic area; none of these is a marketed product.
BRANDS = {
    "oncology": ["Plaxumab", "Torvenil"],
    "cardiology": ["Zentrivo", "Ixarond"],
    "immunology": ["Quellimab", "Dermavex"],
    "neurology": ["Neurazine", "Somnavel"],
    "hematology": ["Hemavor", "Coagulix"],
}
TOPICS = [
    "first-line use",
    "renal impairment",
    "elderly patients",
    "switching from prior therapy",
    "long-term extension data",
    "real-world evidence",
    "patient access programme",
    "managing side effects",
]
FORMATS = ["article", "video", "webinar", "case_study", "dosing_guide", "podcast", "congress_recap"]
FORMAT_WEIGHTS = [0.26, 0.16, 0.12, 0.16, 0.12, 0.08, 0.10]
# Which format tends to follow which, within a brand: a reader of an article goes to the
# dosing guide, a webinar attendee to the case study.
FORMAT_NEXT = {
    "article": ["dosing_guide", "case_study", "video"],
    "video": ["webinar", "article"],
    "webinar": ["case_study", "dosing_guide"],
    "case_study": ["dosing_guide", "webinar"],
    "dosing_guide": ["case_study", "article"],
    "podcast": ["article", "video"],
    "congress_recap": ["article", "webinar"],
}
CHANNELS = ["email", "sms", "rep"]
REGIONS = ["north", "south", "east", "west", "central", "coastal"]

START = pd.Timestamp("2024-09-01")
DAYS = 730


@dataclass(frozen=True)
class Scale:
    customers: int = 20_000
    items: int = 480
    series: int = 55
    mean_events: float = 30.0
    sends_per_customer: float = 15.0
    bots_per_10k: int = 20


def _day_weights() -> np.ndarray:
    """Relative activity per calendar day: weekdays, August lull, year-end shutdown, congresses."""
    days = pd.date_range(START, periods=DAYS, freq="D")
    dow = np.array([1.0, 1.05, 1.05, 1.0, 0.85, 0.3, 0.25])[days.dayofweek]
    month = np.ones(DAYS)
    month[days.month == 8] = 0.6
    month[days.month.isin([6, 10])] = 1.25
    yearend = ((days.month == 12) & (days.day >= 22)) | ((days.month == 1) & (days.day <= 2))
    month[yearend] = 0.25
    w = dow * month
    return w / w.sum()


def _items(rng: np.random.Generator, scale: Scale) -> pd.DataFrame:
    rows = []
    brand_list = [(a, b) for a in AREAS for b in BRANDS[a]]
    # Series first: ordered episodes released two to five weeks apart.
    item_id = 1
    for s in range(scale.series):
        area, brand = brand_list[rng.integers(len(brand_list))]
        topic = TOPICS[rng.integers(len(TOPICS))]
        fmt = rng.choice(["webinar", "case_study", "podcast", "video"])
        n_eps = int(rng.integers(3, 8))
        day = int(rng.integers(0, DAYS - 200))
        for ep in range(1, n_eps + 1):
            rows.append(
                dict(
                    item_id=item_id,
                    title=f"{brand} {topic} series, part {ep}",
                    therapeutic_area=area,
                    brand=brand,
                    topic=topic,
                    format=fmt,
                    series_id=s,
                    episode=ep,
                    release_day=day,
                )
            )
            item_id += 1
            day += int(rng.integers(14, 36))
    congress_days = [d for d in range(DAYS) if (START + pd.Timedelta(days=d)).month in (6, 10)]
    while item_id <= scale.items:
        area, brand = brand_list[rng.integers(len(brand_list))]
        topic = TOPICS[rng.integers(len(TOPICS))]
        fmt = str(rng.choice(FORMATS, p=FORMAT_WEIGHTS))
        # A third of the catalogue predates the window; recaps land in congress months.
        if fmt == "congress_recap":
            day = int(rng.choice(congress_days))
        elif rng.random() < 0.33:
            day = 0
        else:
            day = int(rng.integers(0, DAYS - 30))
        rows.append(
            dict(
                item_id=item_id,
                title=f"{brand}: {topic} ({fmt.replace('_', ' ')})",
                therapeutic_area=area,
                brand=brand,
                topic=topic,
                format=fmt,
                series_id=-1,
                episode=0,
                release_day=day,
            )
        )
        item_id += 1
    items = pd.DataFrame(rows[: scale.items])
    items["release_date"] = START + pd.to_timedelta(items["release_day"], unit="D")
    items["quality"] = rng.lognormal(0.0, 0.6, len(items))
    return items


def _customers(rng: np.random.Generator, scale: Scale) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = scale.customers
    specialty = rng.choice(SPECIALTIES, n, p=SPECIALTY_WEIGHTS)
    years = np.clip(rng.gamma(4.0, 4.0, n), 1, 45).round()
    digital_true = np.clip(rng.beta(2.5, 2.5, n) - 0.008 * (years - 15), 0.01, 0.99)
    rep_covered = rng.random(n) < 0.15 + 0.35 * (specialty != "primary_care")
    cust = pd.DataFrame(
        dict(
            customer_id=np.arange(1, n + 1),
            specialty=specialty,
            region=rng.choice(REGIONS, n),
            years_in_practice=years,
            digital_affinity=np.clip(digital_true + rng.normal(0, 0.12, n), 0, 1).round(3),
            rep_covered=rep_covered,
            email_opt_in=rng.random(n) < 0.9,
            sms_opt_in=rng.random(n) < 0.25 + 0.3 * digital_true,
        )
    )
    # Fields that go missing in a real CRM extract.
    cust.loc[rng.random(n) < 0.05, "years_in_practice"] = np.nan
    cust.loc[rng.random(n) < 0.02, "region"] = None
    latent = pd.DataFrame(
        dict(
            customer_id=cust["customer_id"],
            aff_email=0.9 * (digital_true - 0.5) + rng.normal(0, 0.45, n),
            aff_sms=1.2 * (digital_true - 0.5) - 0.025 * (years - 15) + rng.normal(0, 0.5, n),
            aff_rep=0.8 * (0.5 - digital_true) + 0.03 * (years - 15) + rng.normal(0, 0.45, n),
            digital_true=digital_true,
        )
    )
    return cust, latent


def _preferences(rng: np.random.Generator, cust: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Static per-customer item logits: specialty fit, format taste, item quality."""
    n_c, n_i = len(cust), len(items)
    area_idx = {a: i for i, a in enumerate(AREAS)}
    item_area = items["therapeutic_area"].map(area_idx).to_numpy()
    aff = np.zeros((n_c, len(AREAS)))
    for s_i, s in enumerate(SPECIALTIES):
        m = (cust["specialty"] == s).to_numpy()
        base = np.full(len(AREAS), 0.0) if s == "primary_care" else np.eye(len(AREAS))[s_i] * 2.6
        aff[m] = base + rng.normal(0, 0.5, (m.sum(), len(AREAS)))
    fmt_idx = {f: i for i, f in enumerate(FORMATS)}
    item_fmt = items["format"].map(fmt_idx).to_numpy()
    fmt_taste = rng.normal(0, 0.6, (n_c, len(FORMATS)))
    logits = aff[:, item_area] + fmt_taste[:, item_fmt] + np.log(items["quality"].to_numpy())[None, :]
    assert logits.shape == (n_c, n_i)
    return logits


def _sequences(
    rng: np.random.Generator, cust: pd.DataFrame, items: pd.DataFrame, scale: Scale
) -> pd.DataFrame:
    day_w = _day_weights()
    prefs = _preferences(rng, cust, items)
    n_i = len(items)
    release = items["release_day"].to_numpy()
    series = items["series_id"].to_numpy()
    episode = items["episode"].to_numpy()
    brand = items["brand"].to_numpy()
    fmt = items["format"].to_numpy()
    next_ep = np.full(n_i, -1)
    for s in np.unique(series[series >= 0]):
        idx = np.where(series == s)[0]
        idx = idx[np.argsort(episode[idx])]
        next_ep[idx[:-1]] = idx[1:]
    brand_fmt: dict[tuple[str, str], np.ndarray] = {
        (b, f): np.where((brand == b) & (fmt == f))[0] for b in np.unique(brand) for f in FORMATS
    }

    n_events = 3 + rng.negative_binomial(2, 2 / (2 + scale.mean_events - 3), len(cust))
    heavy = rng.random(len(cust)) < 0.01
    n_events[heavy] *= 4
    starts = rng.integers(0, int(DAYS * 0.75), len(cust))
    rep_cov = cust["rep_covered"].to_numpy()
    sms_ok = cust["sms_opt_in"].to_numpy()

    out_c, out_i, out_t, out_s = [], [], [], []
    for c in range(len(cust)):
        days = np.arange(starts[c], DAYS)
        w = day_w[days] / day_w[days].sum()
        n = int(min(n_events[c], 200))
        t = np.sort(rng.choice(days, n, p=w) + rng.uniform(0.33, 0.85, n))
        seen = np.zeros(n_i, bool)
        open_series: list[int] = []  # next episode of a series the customer started
        last = -1
        for k in range(n):
            avail = (release <= t[k]) & ~seen
            choice = -1
            src = "pref"
            r = rng.random()
            if last >= 0 and next_ep[last] >= 0 and avail[next_ep[last]] and r < 0.4:
                choice, src = int(next_ep[last]), "series"
            elif open_series and rng.random() < 0.15:
                cand = [e for e in open_series if avail[e]]
                if cand:
                    choice, src = int(cand[0]), "series_resume"
            if choice < 0 and last >= 0 and rng.random() < 0.3:
                nxt = FORMAT_NEXT[fmt[last]]
                pool = np.concatenate([brand_fmt[(brand[last], f)] for f in nxt])
                pool = pool[avail[pool]]
                if len(pool):
                    choice, src = int(rng.choice(pool)), "pathway"
            if choice < 0:
                age = np.maximum(t[k] - release, 0)
                logit = prefs[c] + 1.4 * np.exp(-age / 21.0)
                logit = np.where(avail, logit, -np.inf)
                p = np.exp(logit - logit.max())
                choice = int(rng.choice(n_i, p=p / p.sum()))
            seen[choice] = True
            if last >= 0 and next_ep[last] >= 0 and choice != next_ep[last] and not seen[next_ep[last]]:
                open_series.append(int(next_ep[last]))
            open_series = [e for e in open_series if not seen[e]]
            if next_ep[choice] >= 0 and not seen[next_ep[choice]]:
                open_series = [int(next_ep[choice]), *open_series][:3]
            last = choice
            out_c.append(c)
            out_i.append(choice)
            out_t.append(t[k])
            out_s.append(src)

    df = pd.DataFrame(dict(ci=out_c, ii=out_i, t=out_t, source=out_s))
    u = rng.random(len(df))
    cc = df["ci"].to_numpy()
    channel = np.where(u < 0.45, "email", "web")
    channel = np.where((u >= 0.45) & (u < 0.55) & sms_ok[cc], "sms", channel)
    channel = np.where((u >= 0.55) & (u < 0.7) & rep_cov[cc], "rep", channel)
    df["channel"] = channel
    return df


def _bots(rng: np.random.Generator, cust: pd.DataFrame, items: pd.DataFrame, scale: Scale) -> pd.DataFrame:
    """Link scanners on a few inboxes: hundreds of clicks on one day, in no meaningful order."""
    rows = []
    n_bots = max(2, len(cust) * scale.bots_per_10k // 10_000)
    for c in rng.choice(len(cust), n_bots, replace=False):
        day = int(rng.integers(200, DAYS))
        released = np.where(items["release_day"].to_numpy() <= day)[0]
        n = min(len(released), int(rng.integers(150, 300)))
        picked = rng.choice(released, n, replace=False)
        for j, i in enumerate(picked):
            rows.append(dict(ci=int(c), ii=int(i), t=day + 0.4 + j * 1e-4, source="bot", channel="email"))
    return pd.DataFrame(rows)


def engagement_logit(
    latent: pd.DataFrame,
    cust: pd.DataFrame,
    items: pd.DataFrame,
    customer_idx: np.ndarray,
    item_idx: np.ndarray,
    channel: np.ndarray,
    day: np.ndarray,
    recent_touches: np.ndarray,
) -> np.ndarray:
    """True engagement log-odds of one touch. Used to draw outcomes and to score policies."""
    base = {"email": -2.1, "sms": -2.4, "rep": -0.9}
    fmt_ch = {
        ("video", "sms"): -0.6,
        ("webinar", "sms"): -0.4,
        ("dosing_guide", "rep"): 0.6,
        ("case_study", "rep"): 0.4,
        ("congress_recap", "email"): 0.4,
        ("podcast", "email"): 0.2,
        ("article", "email"): 0.2,
    }
    spec = cust["specialty"].to_numpy()[customer_idx]
    area = items["therapeutic_area"].to_numpy()[item_idx]
    fmt = items["format"].to_numpy()[item_idx]
    weekday = (START + pd.to_timedelta(day, unit="D")).dayofweek.to_numpy()
    out = np.empty(len(channel))
    for ch in CHANNELS:
        m = channel == ch
        out[m] = base[ch] + latent[f"aff_{ch}"].to_numpy()[customer_idx[m]]
    out += np.array([fmt_ch.get((f, c), 0.0) for f, c in zip(fmt, channel, strict=True)])
    out += np.where(spec == area, 0.7, np.where(spec == "primary_care", 0.2, -0.4))
    out += np.where((weekday >= 5) & (channel == "rep"), -2.0, 0.0)
    out += np.where((weekday >= 5) & (channel == "email"), -0.3, 0.0)
    out -= 0.3 * recent_touches
    return out


def _sends(
    rng: np.random.Generator,
    cust: pd.DataFrame,
    latent: pd.DataFrame,
    items: pd.DataFrame,
    scale: Scale,
) -> pd.DataFrame:
    """Outbound touches under the historical policy: mostly email, reps for covered accounts,
    with a quarter of touches assigned at random among eligible channels."""
    n = len(cust)
    k = rng.poisson(scale.sends_per_customer, n)
    ci = np.repeat(np.arange(n), k)
    day = rng.choice(DAYS, len(ci), p=_day_weights())
    release = items["release_day"].to_numpy()
    area = items["therapeutic_area"].to_numpy()
    spec = cust["specialty"].to_numpy()
    # Vectorised draw: the first released candidate, preferring the customer's own area 70%
    # of the time. Day-0 catalogue items guarantee every row has a released fallback.
    draws = rng.integers(len(items), size=(len(ci), 40))
    released = release[draws] <= day[:, None]
    own = area[draws] == spec[ci][:, None]
    want_own = rng.random(len(ci)) < 0.7
    valid = released & (own | ~want_own[:, None])
    pick = np.where(valid.any(axis=1), valid.argmax(axis=1), released.argmax(axis=1))
    ii = draws[np.arange(len(ci)), pick]
    ii = np.where(release[ii] <= day, ii, int(np.argmin(release)))

    elig = np.stack(
        [
            cust["email_opt_in"].to_numpy()[ci],
            cust["sms_opt_in"].to_numpy()[ci],
            cust["rep_covered"].to_numpy()[ci],
        ],
        axis=1,
    )
    elig[~elig.any(axis=1), 0] = True  # transactional email is always allowed
    hist = np.array([0.65, 0.15, 0.20])
    probs = elig * hist
    explore = rng.random(len(ci)) < 0.25
    probs[explore] = elig[explore].astype(float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cum = probs.cumsum(axis=1)
    ch_idx = (rng.random(len(ci))[:, None] > cum).sum(axis=1)
    channel = np.array(CHANNELS)[ch_idx]

    df = pd.DataFrame(dict(ci=ci, ii=ii, day=day, channel=channel))
    df = df.sort_values(["ci", "day"]).reset_index(drop=True)
    # Touches in the previous seven days, the fatigue term.
    recent = np.zeros(len(df))
    prev_c = -1
    window: list[int] = []
    for j, (c, d) in enumerate(zip(df["ci"].to_numpy(), df["day"].to_numpy(), strict=True)):
        if c != prev_c:
            prev_c, window = c, []
        window = [x for x in window if d - x < 7]
        recent[j] = len(window)
        window.append(d)
    df["recent_touches"] = recent
    logit = engagement_logit(
        latent,
        cust,
        items,
        df["ci"].to_numpy(),
        df["ii"].to_numpy(),
        df["channel"].to_numpy(),
        df["day"].to_numpy(),
        recent,
    )
    df["p_true"] = 1 / (1 + np.exp(-logit))
    df["engaged"] = rng.random(len(df)) < df["p_true"]
    return df


def _snippets(rng: np.random.Generator, items: pd.DataFrame) -> pd.DataFrame:
    """Claim text per (brand, topic): current approved versions plus superseded ones whose
    numbers differ, which is exactly what a draft must never quote."""
    rows = []
    sid = 1
    pairs = items[["brand", "topic", "therapeutic_area"]].drop_duplicates()
    for brand, topic, area in pairs.itertuples(index=False):
        trial = f"{brand[:3].upper()}-{rng.integers(100, 400)}"
        versions = ["approved"] + ["expired"] * int(rng.integers(0, 3))
        for status in versions:
            pct = int(rng.integers(12, 48))
            weeks = int(rng.choice([12, 24, 48, 52, 96]))
            dose = int(rng.choice([5, 10, 20, 40, 100, 200]))
            ae = rng.choice(["headache", "nausea", "fatigue", "injection-site reactions", "dizziness"])
            ae_pct = int(rng.integers(3, 15))
            texts = {
                "efficacy": (
                    f"In the {trial} study in {topic}, {brand} reduced the primary endpoint by "
                    f"{pct}% versus comparator at week {weeks}."
                ),
                "dosing": (
                    f"For {topic}, the recommended {brand} dose is {dose} mg once daily; "
                    f"no adjustment is needed at week {weeks}."
                ),
                "safety": (
                    f"The most common adverse reaction with {brand} in {topic} was {ae} "
                    f"({ae_pct}% of patients)."
                ),
            }
            for kind, text in texts.items():
                st = status
                if status == "approved" and rng.random() < 0.06:
                    st = "draft"
                rows.append(
                    dict(
                        snippet_id=sid,
                        brand=brand,
                        topic=topic,
                        therapeutic_area=area,
                        kind=kind,
                        status=st,
                        text=text,
                    )
                )
                sid += 1
    return pd.DataFrame(rows)


def generate(out_dir: Path, scale: Scale | None = None, seed: int = 7) -> dict[str, int]:
    scale = scale or Scale()
    rng = np.random.default_rng(seed)
    items = _items(rng, scale)
    cust, latent = _customers(rng, scale)
    seq = _sequences(rng, cust, items, scale)
    seq = pd.concat([seq, _bots(rng, cust, items, scale)], ignore_index=True)

    ts = START + pd.to_timedelta(seq["t"], unit="D")
    fmt = items["format"].to_numpy()[seq["ii"].to_numpy()]
    dwell_mu = pd.Series(fmt).map(
        {"video": 5.3, "webinar": 6.8, "podcast": 6.5, "article": 4.6, "case_study": 5.0}
    )
    dwell = rng.lognormal(dwell_mu.fillna(4.2).to_numpy(), 0.7)
    dwell[seq["source"].to_numpy() == "bot"] = rng.uniform(0, 2, (seq["source"] == "bot").sum())
    dwell[rng.random(len(dwell)) < 0.003] *= 50  # tab left open overnight
    dwell[rng.random(len(dwell)) < 0.08] = np.nan  # tracking blocked
    interactions = pd.DataFrame(
        dict(
            customer_id=cust["customer_id"].to_numpy()[seq["ci"].to_numpy()],
            item_id=items["item_id"].to_numpy()[seq["ii"].to_numpy()],
            ts=ts.dt.floor("s"),
            channel=seq["channel"],
            dwell_seconds=np.round(dwell, 1),
        )
    ).sort_values(["customer_id", "ts"], kind="stable")
    sources = pd.DataFrame(
        dict(
            customer_id=interactions["customer_id"],
            item_id=interactions["item_id"],
            source=seq.loc[interactions.index, "source"],
        )
    )

    sends = _sends(rng, cust, latent, items, scale)
    sends_out = pd.DataFrame(
        dict(
            send_id=np.arange(1, len(sends) + 1),
            customer_id=cust["customer_id"].to_numpy()[sends["ci"].to_numpy()],
            item_id=items["item_id"].to_numpy()[sends["ii"].to_numpy()],
            sent_at=(START + pd.to_timedelta(sends["day"], unit="D") + pd.Timedelta(hours=9)),
            channel=sends["channel"],
            recent_touches=sends["recent_touches"].astype(int),
            engaged=sends["engaged"].astype(int),
        )
    )
    snippets = _snippets(rng, items)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(exist_ok=True)
    cust.to_csv(out_dir / "customers.csv", index=False)
    items.drop(columns=["quality", "release_day"]).to_csv(out_dir / "items.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv.gz", index=False)
    sends_out.to_csv(out_dir / "sends.csv.gz", index=False)
    snippets.to_csv(out_dir / "snippets.csv", index=False)
    latent.to_csv(out_dir / "truth" / "customer_latent.csv", index=False)
    sources.to_csv(out_dir / "truth" / "interaction_source.csv.gz", index=False)
    pd.DataFrame(dict(send_id=sends_out["send_id"], p_true=sends["p_true"].round(5))).to_csv(
        out_dir / "truth" / "send_p_true.csv.gz", index=False
    )
    return dict(
        customers=len(cust),
        items=len(items),
        interactions=len(interactions),
        sends=len(sends_out),
        snippets=len(snippets),
    )
