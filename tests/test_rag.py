import pandas as pd

from genai_recommender import rag
from genai_recommender.data import load


def test_expired_claims_never_reach_the_index(tiny_data):
    t = load(tiny_data)
    idx = rag.SnippetIndex().fit(t.snippets)
    assert set(idx.docs["status"]) == {"approved"}
    item = t.items.iloc[0]
    hits = idx.search(item["title"], k=3, brand=item["brand"])
    assert (hits["brand"] == item["brand"]).all()


def test_retrieval_eval_shows_why_the_filter_matters(tiny_data):
    t = load(tiny_data)
    r = rag.retrieval_eval(t.snippets, t.items).set_index("arm")
    assert r.loc["approved only + brand filter", "any unapproved claim retrieved"] == 0
    assert r.loc["all statuses indexed", "any unapproved claim retrieved"] > 0.2


def test_grounding_check_catches_an_invented_number():
    claims = pd.DataFrame(dict(snippet_id=[1], text=["Brand reduced the endpoint by 31% at week 24."]))
    assert rag.grounding_check("Cut the endpoint by 31% at week 24.", claims, "Title")["grounded"]
    bad = rag.grounding_check("Cut the endpoint by 41% at week 24.", claims, "Title")
    assert not bad["grounded"] and bad["ungrounded"] == ["41"]


def test_citation_labels_pass_only_for_retrieved_claims():
    claims = pd.DataFrame(dict(snippet_id=[21], text=["Brand cut the endpoint by 31% at week 24."]))
    assert rag.grounding_check("Cut by 31% at week 24 (S21).", claims, "Title")["grounded"]
    bad = rag.grounding_check("Cut by 31% at week 24 [S22].", claims, "Title")
    assert bad["ungrounded"] == ["22"]


def test_stub_drafter_quotes_an_approved_claim(tiny_data):
    t = load(tiny_data)
    d = rag.Drafter(rag.SnippetIndex().fit(t.snippets), rag.StubLLM())
    out = d.draft("cardiology", t.items.iloc[3], "sms")
    assert out["grounded"]
    approved = set(t.snippets.loc[t.snippets["status"] == "approved", "text"])
    assert any(text in str(out["draft"]) for text in approved)
