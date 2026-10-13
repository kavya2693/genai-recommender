# ADR-0001: Three separate decisions, and claim governance enforced in the index

## Status
Accepted

## Context
A commercial team has to decide, for each prescriber, what content to send next, through
which channel, and with what wording. The wording may only quote medical-legal approved
claims; a superseded claim with an old number is a compliance incident, not a style issue.
The original brief combined all three in one model with a hosted LLM.

## Decision
- Next content: a small BERT4Rec (2 layers, 64-dim, 30-item window) trained by masked-item
  prediction on CPU, evaluated over the full catalogue (no sampled negatives) against
  popularity, item-kNN and a last-item Markov chain.
- Channel: a gradient-boosted classifier over (customer, content, channel) features with
  isotonic recalibration on a later time slice. The batch policy gives rep visits, a fixed
  headcount, to the touches where the rep beats the best digital channel by the widest
  predicted margin, at the same rep share as the historical policy.
- Draft: only `approved` snippets enter the retrieval index, retrieval is filtered to the
  recommended content's brand before ranking, and every number in a draft is checked
  against the retrieved claims. The model (llama3.2 via Ollama for demos, a deterministic
  stub in tests and CI) only writes prose around them.
- Retrieval uses TF-IDF + truncated SVD embeddings in numpy. At ~230 approved snippets an
  approximate index (FAISS) adds a dependency and no speed.

## Alternatives rejected
- **Status filtering in the prompt** ("do not use expired claims"). Expired and current
  claims differ only in their numbers; the model cannot tell which is which, and the
  retrieval experiment shows similarity search cannot either.
- **One joint model for content and channel.** Channel engagement is driven by the
  customer, content choice by sequence; separating them lets each be measured on its own
  ground truth.
- **Unconstrained channel argmax.** Rep visits engage best on average, so argmax sends a
  rep everywhere one is allowed; no field team could run that plan.
- **Hosted embedding model or LLM.** CI must run with no key and no network model.

## Consequences
- Every number in the README comes from `genai-reco` and lands in `outputs/`.
- The drafting step can swap models without touching the governance guarantees.
