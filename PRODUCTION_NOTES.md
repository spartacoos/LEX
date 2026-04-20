# Notes on Production Effort

This prototype took a few days and covers the full pipeline: Formex XML ingestion from EUR-Lex, hybrid retrieval (dense + BM25), reranking, two-stage RAG with citation extraction, streaming chat UI, and an evaluation suite with 30 curated questions.

Evaluated using Qwen 3.5 9B (quantized, single GPU):

| Metric               | Score | Target |
| -------------------- | ----: | -----: |
| Context recall       |  0.90 |   0.80 |
| Answer relevancy     |  0.90 |   0.85 |
| Context precision    |  0.57 |   0.80 |
| Faithfulness         |  0.80 |   0.90 |
| Citation correctness |  0.51 |   0.90 |

Recall and relevancy are strong. The remaining gaps are mostly model-size limitations. A 31B+ model or API backend would likely close most of them. The smaller model was intentional to speed up iteration and stress retrieval quality.

---

## Production Effort

The prototype validates the core RAG approach for EU legal text. Production, however, must be directive-agnostic, secure, and legally reliable. The final architecture will likely change based on enterprise constraints (cloud, databases, inference stack).

Estimated timeline: **3–9 months**, with **4–6 months most likely** for a dedicated **3–4 engineer team**.

### Track 1 — Architecture & Infrastructure (4–8 weeks)

Finalize production architecture, scalable inference, autoscaling, load balancing, and high-availability storage. The current modular design is useful but not assumed to survive unchanged.

### Track 2 — Data Pipeline & Corpus Management (6–12 weeks)

Handle cross-referenced directives, annexes, amendments, and versioning. Automatic cross-linking and diff tracking introduce the largest uncertainty. Multi-linguality might be another non-trivial feature that could take significant efforts.

### Track 3 — Security & Compliance (4–8 weeks)

Multi-tenant architecture, RBAC, audit logging, authentication, and data residency compliance. Ensuring that no adverserial attacks are possible and that our model can't be used for things outside of the scope of the tool (i.e. as a general LLM interface). Also, that model usage is bounded to avoid runoff costs.

### Track 4 — Evaluation & Legal Alignment (8–14 weeks)

Build a large gold dataset, adversarial testing, and eliminate hallucinated citations. Requires human legal review loops. Model performance is itself a moving target with a new SOTA model every 3-6 months which may force us to revaluate assumptions. Formex XML may itself continue evolving over time. 

---

## Critical Path

Tracks run in parallel, but **Data + Evaluation** form the bottleneck:

* Data pipeline: 6–12 weeks
* Evaluation and legal tuning: 8–14 weeks

Critical path: **14–26 weeks**
Most likely delivery: **4–6 months**
Tail risk: **7–9 months**
Aggressive internal release: **~3 months**

---

## Tooling Note

Claude Opus was used as a coding assistant. I designed the architecture, APIs, diagrams, and system logic. Claude generated the Python code, which I then validated through CLI and UI testing.

AI accelerates implementation but does not reduce the core systems engineering, compliance, and evaluation challenges outlined above.

