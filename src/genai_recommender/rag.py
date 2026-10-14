"""Grounded message drafts: retrieve approved claim text for the recommended content and have a
language model write the message around it.

The retrieval index is a small dense one (TF-IDF projected to 64 dimensions with truncated SVD,
cosine similarity in numpy). At ~500 snippets an approximate index buys nothing, and an
embedding model download would make the test suite depend on the network.

The governance decision lives here, not in the prompt: superseded and draft claims never enter
the index. Expired claims read almost exactly like current ones (same brand, same topic,
different numbers), so similarity search cannot be trusted to tell them apart.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

NUMBER = re.compile(r"\d+(?:\.\d+)?")
CITATION = re.compile(r"\bS(\d+)\b")


class SnippetIndex:
    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def fit(self, snippets: pd.DataFrame, approved_only: bool = True) -> SnippetIndex:
        docs = snippets[snippets["status"] == "approved"] if approved_only else snippets
        self.docs = docs.reset_index(drop=True)
        text = self.docs[["brand", "topic", "kind", "text"]].astype(str).agg(" ".join, axis=1)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        tfidf = self.vec.fit_transform(text)
        self.svd = TruncatedSVD(min(self.dim, tfidf.shape[1] - 1, len(self.docs) - 1), random_state=self.seed)
        self.emb = self._norm(self.svd.fit_transform(tfidf))
        return self

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def embed(self, queries: list[str]) -> np.ndarray:
        return self._norm(self.svd.transform(self.vec.transform(queries)))

    def search(self, query: str, k: int = 3, brand: str | None = None) -> pd.DataFrame:
        """Top-k snippets by cosine similarity. A brand filter is applied before ranking when
        given, because quoting another product's claim is the one error that cannot ship."""
        sim = self.emb @ self.embed([query])[0]
        if brand is not None:
            sim = np.where(self.docs["brand"].to_numpy() == brand, sim, -np.inf)
        top = np.argsort(-sim)[:k]
        top = top[np.isfinite(sim[top])]
        return self.docs.iloc[top].assign(score=sim[top])


def retrieval_eval(snippets: pd.DataFrame, items: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Ground truth by construction: for an item about (brand, topic), the right snippets are the
    approved ones written for that brand and topic. Three arms: index everything, index only
    approved claims, and approved claims with a brand filter."""
    queries = items.drop_duplicates(["brand", "topic"])
    arms = {
        "all statuses indexed": (SnippetIndex().fit(snippets, approved_only=False), False),
        "approved only": (SnippetIndex().fit(snippets, approved_only=True), False),
        "approved only + brand filter": (SnippetIndex().fit(snippets, approved_only=True), True),
    }
    rows = []
    for name, (index, use_brand) in arms.items():
        prec, unapproved, wrong_brand, full = [], [], [], []
        for q in queries.itertuples(index=False):
            hits = index.search(str(q.title), k=k, brand=str(q.brand) if use_brand else None)
            right = (hits["brand"] == q.brand) & (hits["topic"] == q.topic) & (hits["status"] == "approved")
            n_right = int(
                (
                    (snippets["brand"] == q.brand)
                    & (snippets["topic"] == q.topic)
                    & (snippets["status"] == "approved")
                ).sum()
            )
            prec.append(right.mean() if len(hits) else 0.0)
            full.append(right.sum() >= min(k, n_right))
            unapproved.append((hits["status"] != "approved").any())
            wrong_brand.append((hits["brand"] != q.brand).any())
        rows.append(
            {
                "arm": name,
                "queries": len(queries),
                f"precision@{k}": float(np.mean(prec)),
                "all right snippets found": float(np.mean(full)),
                "any unapproved claim retrieved": float(np.mean(unapproved)),
                "any other-brand claim retrieved": float(np.mean(wrong_brand)),
            }
        )
    return pd.DataFrame(rows)


class LLM(Protocol):
    name: str

    def complete(self, prompt: str) -> str: ...


class StubLLM:
    """Deterministic stand-in used by tests and CI: quotes the first claim verbatim."""

    name = "stub"

    def complete(self, prompt: str) -> str:
        claims = re.findall(r"^\[S\d+\] (.+)$", prompt, flags=re.M)
        title = re.search(r"^Content: (.+)$", prompt, flags=re.M)
        lead = f"New for you: {title.group(1)}." if title else "New content is available."
        return f"{lead} {claims[0]}" if claims else lead


@dataclass
class OllamaLLM:
    model: str = "llama3.2"
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    timeout: float = 300.0
    name: str = "ollama"

    def complete(self, prompt: str) -> str:
        options: dict[str, float | int] = {"temperature": 0.2, "num_predict": 220}
        # Some hosts crash on the GPU path; OLLAMA_NUM_GPU=0 forces CPU inference.
        if "OLLAMA_NUM_GPU" in os.environ:
            options["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "options": options})
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body.encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return str(json.loads(resp.read())["response"]).strip()


CHANNEL_RULES = {
    "email": "Write a short email body (under 120 words) with a one-line subject on the first line.",
    "sms": "Write one SMS under 300 characters. No subject line.",
    "rep": "Write three short talking points a sales rep can say in a visit.",
}


def build_prompt(specialty: str, title: str, channel: str, claims: pd.DataFrame) -> str:
    lines = [
        "You draft outreach to a physician about one piece of approved educational content.",
        f"Recipient specialty: {specialty}",
        f"Content: {title}",
        f"Channel: {channel}. {CHANNEL_RULES[channel]}",
        "Approved claims (quote any claim you use word for word; do not state any other number,"
        " percentage, dose or study name):",
        *[f"[S{sid}] {text}" for sid, text in zip(claims["snippet_id"], claims["text"], strict=True)],
        "Write only the message.",
    ]
    return "\n".join(lines)


def grounding_check(draft: str, claims: pd.DataFrame, title: str) -> dict[str, object]:
    """Every number in the draft must appear in a retrieved approved claim (or the content
    title). A number that does not is either invented or copied from somewhere it should not be."""
    allowed = set(NUMBER.findall(" ".join(claims["text"]) + " " + title))
    # The prompt labels each claim [S<id>], and models copy those labels as citations. A label
    # for a retrieved claim is not a factual number; a label for any other id still fails.
    retrieved = {str(i) for i in claims["snippet_id"]}
    draft = CITATION.sub(lambda m: "" if m.group(1) in retrieved else m.group(0), draft)
    found = NUMBER.findall(draft)
    bad = sorted({n for n in found if n not in allowed})
    return {"numbers": len(found), "ungrounded": bad, "grounded": not bad}


@dataclass
class Drafter:
    index: SnippetIndex
    llm: LLM
    k: int = 3

    def draft(self, specialty: str, item: pd.Series, channel: str) -> dict[str, object]:
        claims = self.index.search(str(item["title"]), k=self.k, brand=str(item["brand"]))
        prompt = build_prompt(specialty, str(item["title"]), channel, claims)
        text = self.llm.complete(prompt)
        check = grounding_check(text, claims, str(item["title"]))
        return {"snippet_ids": claims["snippet_id"].tolist(), "draft": text, **check}
