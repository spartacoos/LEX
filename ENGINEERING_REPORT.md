# LEX Engineering Report
## Natural-Language Q&A over EU Directives — From Demo to Production-Ready

*Carlo — April 2026*

---

## Overview

This report documents the engineering journey behind LEX, a system that allows
users to ask plain-language questions about EU directives and receive grounded,
cited answers. It is written for both technical and non-technical readers — the
goal is to convey not just *what* was built, but *why* certain problems are
genuinely hard, and what distinguishes a production-ready system from a
convincing prototype.

The target directive is the European Electronic Communications Code (EECC,
CELEX `32018L1972`) — a ~180-page telecoms regulation. The system ingests the
directive, indexes it, and answers questions like *"What obligations apply to
undertakings with significant market power?"* with specific article citations
and paragraph-level grounding.

---

## 1. Why This Problem Is Harder Than It Looks

The naive version of this system is straightforward to build: extract text from
a PDF, split it into chunks, store them in a vector database, and answer
questions using a language model. Such a prototype can be built in an afternoon
and will produce impressive-looking results on simple questions.

The gap between that prototype and something a lawyer or regulator could
actually trust is substantial. Legal text has properties that stress every
layer of the stack simultaneously:

**Dense cross-referencing.** Articles reference other articles, which reference
other directives, which reference technical standards outside any single
document. A question about "national regulatory authority" cannot be fully
answered from the EECC alone — the definition lives in the BEREC Regulation
(EU) 2018/1971. A complete system needs to understand where its knowledge
ends and say so clearly, rather than fabricating a plausible-sounding answer.

**Operative vs. explanatory text.** EU directives contain two distinct layers:
recitals (the preamble sections numbered (1), (2), (3)...) which explain the
legislative intent, and operative articles which contain the actual binding
obligations. The EECC has 326 recitals and ~440 operative paragraphs. For most
legal questions, the operative text is what matters. But recitals are
semantically richer and tend to score higher in standard retrieval — a system
that consistently returns rationale instead of obligations is misleading even
when technically correct.

**Abbreviations and domain vocabulary.** Legal practitioners write "SMP
undertakings" meaning "undertakings designated as having significant market
power." A retrieval system that does not bridge this vocabulary gap will
consistently fail on the most expert queries — precisely the queries where the
system is most needed.

**Temporal and scope boundaries.** Directives are amended. Questions may span
multiple instruments. Answers that were correct in 2018 may be wrong after a
2023 amendment. A trustworthy system needs to be explicit about what it knows,
when it was indexed, and what falls outside its scope.

---

## 2. Architecture — Keeping Complexity Under Control

Before describing the problems, it is worth explaining the structural choice
that made it possible to build and iterate on this system efficiently.

Every entry point in LEX — the command-line interface, the REST API, the chat
UI, the test suite — builds a typed Command object and submits it to a central
Engine. The engine routes it to the appropriate handler. Nothing calls anything
else directly.

```python
# The entire public API in one type
Command = IngestCmd | RetrieveCmd | AnswerCmd
```

This pattern — common in well-structured backend systems — has a specific
advantage when developing with LLM coding assistants: the codebase stays small
and the system contract stays visible. At any point, an assistant shown the
command types and one handler file has all the context needed to add a feature
correctly, without needing to read the entire codebase.

Adding a new capability means: add a variant to the Command union, write a
handler, add a routing case. Existing handlers are never modified. This kept
LEX at ~12 source files throughout development, regardless of how many features
were added.

Each command also carries a UUID, making every operation traceable across logs,
the database, and the Redis job queue without any additional instrumentation.

---

## 3. Parsing EU Directive XML

### The challenge

EU directives are published in a structured XML format called Formex. In
principle this is ideal — every article, paragraph, and recital has a defined
location in a hierarchy. In practice, Formex has evolved over decades and the
documents themselves are large enough that parsing subtleties matter.

### Schema evolution

Early Formex used typed structural tags (`<PART>`, `<TITLE>`, `<CHAPTER>`).
Modern Formex replaces all of these with a generic `<DIVISION>` wrapper whose
kind is encoded in its own heading text — "PART I", "CHAPTER IV". We detect
this with a pattern match on the heading, falling back gracefully when the
pattern does not match rather than crashing.

### The two-hop content fetch

The EUR-Lex CELLAR repository does not serve a single XML file per directive.
It serves an RDF manifest listing all content items. We parse the manifest,
identify the correct Formex4 item (typically the second document — the first
is a cover page), and fetch it. This two-hop approach means a 404 on the
manifest returns a clean error rather than an empty index, and annexes (which
live in separate files) are explicitly out of scope for v1.

### Unicode and whitespace

Formex uses non-breaking space (U+00A0) between labels and numbers —
"Article\u00a01" rather than "Article 1". Without normalisation, every
article number extraction fails silently. Small details like this are invisible
until you notice that the system cannot find Article 1.

### The Article 2 definition problem

Article 2 of the EECC contains 42 numbered definitions in a single 14,072-
character block. Our default chunker splits this by character count, producing
chunks like "definitions 1–8" that dilute every individual definition's
retrieval signal.

The fix required examining the actual boundary pattern in the XML:

```
...term A;(N)term B means...
```

Not `(N) "term"` with quotes as initially assumed, but bare `(N)term` with a
semicolon as the only boundary signal. The splitting regex evolved through
three iterations as we probed the actual content, ultimately requiring negative
lookaheads to suppress inline cross-references like "point (2) of Article 3"
which match the same pattern:

```python
_ART2_DEF_RE = re.compile(
    r'(?<=;)(?=\(\d+\)(?!\s+of\b)(?!\s+subparagraph\b)(?!\s+point\b))'
)
```

Definition (1) has no preceding semicolon (it follows the preamble directly)
and required a separate extraction step. After these fixes, we went from 880
chunks to 921 — 41 additional definition sub-chunks, each retrievable
independently. The VHCN definition now surfaces at rank 1 with rerank score
+0.999 for queries about very high capacity networks.

This kind of micro-problem — a regex that must handle three overlapping
patterns in a 14,000-character block of legal text — is representative of the
gap between a demo and a deployed system. It is invisible in evaluation until
you notice that every "What is X?" question is returning the wrong article.

---

## 4. Chunking Strategy

Chunking is dividing a document into segments small enough to be retrieved
individually but large enough to contain coherent meaning. Legal text makes
this unusually difficult because the natural unit of meaning — an article —
can range from 50 to 3,000 characters, and splitting mid-article loses
essential context.

### Our approach

We exploit the Formex hierarchy directly: each numbered paragraph becomes a
chunk. Articles without numbered paragraphs become single chunks. Paragraphs
exceeding 1,500 characters are split at sentence boundaries with 2-sentence
overlap, so a reference straddling a boundary survives in both halves.

### Sentence splitting in legal prose

Standard sentence splitters fail on legal text because legal abbreviations
("Art.", "No.", "para.", "i.e.") are followed by periods and do not end
sentences. We implemented a two-pass approach: split aggressively on period +
capital letter, then re-join at known abbreviation boundaries using a
whitelist. This is extensible — new abbreviations can be added as encountered.

### Operative vs. explanatory text

The 326 recitals in the EECC outnumber operative paragraphs and tend to score
higher in semantic retrieval because they contain richer explanatory language.
We apply a post-reranking score adjustment — 0.85 multiplier for recitals,
1.0 for articles — to push operative text up the ranking for most query types.
This is a calibrated heuristic, not a learned parameter, and represents a
deliberate design choice that trades some flexibility for predictability.

---

## 5. Retrieval — Hybrid Search

### The two-signal approach

We use two complementary retrieval signals combined via Qdrant's hybrid search:

**Dense embeddings** (BGE-M3) convert text to 1024-dimensional vectors where
semantic similarity corresponds to geometric proximity. "SMP undertaking" and
"undertaking designated as having significant market power" end up close in
this space even though they share no words.

**Sparse BM25 vectors** reward exact token matches weighted by how rare each
token is across the corpus (IDF — Inverse Document Frequency). "Article 63"
gets a high BM25 score only in chunks that actually contain those exact tokens.
This is essential for legal text where specific article references, defined
terms, and numeric identifiers must be matched precisely.

The two signals are fused using Reciprocal Rank Fusion — a parameter-free
method that combines ranked lists without requiring calibrated score scales:

```
RRF_score(document) = Σ  1 / (60 + rank_in_list)
                      lists
```

A document ranked first in either list scores well. Ranked first in both, it
scores highest. No tuning required.

### The BGE-M3 sparse head investigation

BGE-M3 is documented as producing both dense and sparse vectors in a single
pass. Our initial implementation followed this documentation. Systematic
diagnosis revealed the sparse head was silently returning zeros on every query:

```python
# What the model actually returns
type: BaseModelOutputWithPoolingAndCrossAttentions
attributes: last_hidden_state, pooler_output   # no sparse_output
```

The published HuggingFace checkpoint registers `architectures: XLMRobertaModel`
with no custom model class. The sparse head is not present. We had been running
dense-only retrieval while believing we had hybrid search.

The fix was to implement BM25 as the sparse leg. For legal text this is
arguably better than a learned sparse representation — BM25 with proper IDF
weighting rewards exact legal term matches without requiring the sparse head
weights that were not available.

### BM25 at query time

BM25 IDF requires knowing the full corpus token distribution — a quantity that
cannot be computed from a single query. We compute and save the IDF table at
ingest time and load it at query time:

```python
# Ingest: save ~4,000 token IDF weights (~50KB)
idf_save_path.write_text(json.dumps(bm25.idf))

# Query: load IDF, weight query tokens
weight(token) = idf.get(str(token_id), 0.0)
```

The asymmetric formulation — IDF-only for queries, full BM25 TF-IDF for
documents — is standard practice. Document weights already encode term
frequency normalisation; query weights need only the IDF amplification for
rare, important terms.

---

## 6. Hypothetical Document Embedding (HyDE)

### The vocabulary gap problem

There is a fundamental mismatch between how people ask questions and how
directives are written. A user asks: *"What obligations apply to SMP
undertakings?"* A directive says: *"Where an undertaking has been designated
as having significant market power... the national regulatory authority shall
impose obligations of transparency, non-discrimination, access..."*

The words are entirely different. The meaning is the same. Standard embedding
models handle common language paraphrase well but struggle with domain-specific
abbreviations and formal legal phrasing that differs significantly from
conversational text.

### The approach

HyDE (Hypothetical Document Embedding) addresses this by asking the language
model to generate a short passage that *would* answer the question, in the
style of directive text, before embedding. The generated passage uses the same
vocabulary as the actual articles, bringing the query vector closer to the
relevant document vectors in embedding space.

### Model dependency and the validity guard

The approach is sensitive to generation quality. With a 4B parameter model,
the generator produced hallucinated placeholders:

```
"Article 15(1) of Directive [Insert Relevant Directive Number/Title Here]..."
```

Using this as a retrieval query produces worse results than the original
question. With Qwen 3.5 9B, the same prompt generates clean legal prose:

```
"In accordance with the relevant provisions governing undertakings holding
a dominant position in electronic communications markets, designated operators
shall be subject to obligations of transparency, non-discrimination..."
```

The vocabulary match to the actual operative articles (Arts. 68-74) is precise.
Retrieval quality improved substantially and measurably.

We implemented a model-agnostic validity guard that rejects bad expansions
before they contaminate retrieval — checking for bracketed placeholders,
invented article numbers, and near-copies of the original query. When rejected,
the system falls back to the original query silently:

```python
def _hyde_is_usable(original: str, expanded: str) -> bool:
    if len(expanded) <= len(original):            # not an expansion
        return False
    if re.search(r'\[.{2,40}\]', expanded):       # placeholder
        return False
    if re.search(r'\bArticle\s+\d+', expanded):   # invented citation
        return False
    ...
    return True
```

### Tradeoff summary

| Signal | Strengths | Limitations |
|---|---|---|
| Dense embedding | Semantic paraphrase, synonyms | Abbreviations, exact refs |
| BM25 sparse | Exact terms, article numbers | Cannot handle paraphrase |
| HyDE (small model) | Minor vocabulary expansion | Unreliable, hallucination risk |
| HyDE (9B+ model) | Domain abbreviation bridging | Extra LLM call, model-dependent |

The combination of all three — dense + BM25 + HyDE with validity guard — is
what makes the system robust across the full range of query types we observed.

---

## 7. Evaluation

### Approach

We evaluated against 30 hand-curated question-answer pairs covering five
categories designed to stress different failure modes:

- **Definitional** — "What is X?" — tests whether definition sub-chunks
  surface before operative article chunks
- **Procedural** — "How does Y work?" — tests whether operative articles
  rank above explanatory recitals
- **Cross-reference** — "Which articles govern Z?" — tests multi-article
  synthesis
- **Negative** — "Does this directive cover W?" — tests grounded refusal
  without hallucination
- **Multi-hop** — questions requiring synthesis across multiple articles
  and the connections between them

We used DeepEval metrics (context precision, context recall, faithfulness,
answer relevancy) with a larger model as judge, plus a custom citation
correctness metric: the fraction of expected article references appearing
in the answer's citations.

### Key findings

Before the improvements documented in this report, the system scored well on
simple definitional and negative queries but failed consistently on:

- Abbreviation-heavy queries ("SMP obligations" → complete refusal)
- Procedural queries ("market analysis procedure" → recitals, not Arts. 64-67)
- Complex definitions split across chunking boundaries

After all improvements, the previously failing queries now return correct,
well-cited answers. The "SMP obligations" query, previously a complete refusal,
now returns Arts. 83(2) and 77(1) with a structured answer covering retail
price obligations, tariff controls, and functional separation requirements.

### The judge model problem

Small models (4B parameters) produce invalid JSON when acting as evaluation
judges, causing evaluation runs to fail or produce unreliable scores. A minimum
of ~13B parameters is needed for reliable structured output. For evaluation
runs we recommend using a remote API regardless of which model is used for
RAG itself.

---

## 8. Intrinsic Limitations

These are not engineering failures. They are fundamental properties of the
domain.

**Cross-instrument references.** The EECC references at least a dozen other
legal instruments for key definitions and obligations. A question whose answer
lives in a referenced instrument will receive a partial answer or a grounded
refusal. The correct long-term solution is to ingest the full regulatory
corpus. The system is architecturally ready for this — each instrument is a
single `lex ingest <CELEX_ID>` command.

**Temporal validity.** EU directives are amended. We index a specific version
at a specific time. Handling amendments requires either fetching consolidated
versions or maintaining a versioned document store. This is not implemented.

**Annexes.** The EECC's technical annexes live in separate CELLAR files.
The current fetcher retrieves the main act only. Annex content — often the most
technically specific part of a directive — is absent from the index.

**The definition recursion problem.** Legal definitions are recursive. "Electronic
communications service" is defined in terms of "electronic communications
network" which may reference technical standards external to any directive.
The system correctly refuses to answer beyond what is ingested, but cannot
follow the definitional chain across instrument boundaries automatically.

---

## 9. What Remains

In priority order:

**Query classification.** A lightweight classifier routing "What is X?"
queries to article-only search and "Does this directive cover Y?" to
negative-aware prompting. Estimated improvement: +5-10% on definitional
and negative query categories.

**Two-stage retrieval.** Generate a draft answer, extract the article numbers
it references, fetch those articles directly by metadata, merge with the
original top-5. This would fix residual failures where the correct article
appears in a draft but not in the initial retrieval pass.

**Citation graph traversal.** At ingest time, extract all "Article N"
cross-references from chunk text. At retrieval time, when Article 68 is
retrieved, automatically also fetch Articles 67 and 63 which Article 68
explicitly references. This mirrors how a lawyer reads a directive — following
the footnotes, not stopping at the first hit.

**Corpus expansion.** Ingest the BEREC Regulation, ePrivacy Directive, and
Directive 2014/61/EU to resolve the most common cross-instrument reference
failures.

---

## 10. Conclusion

The gap between a working RAG prototype and a system suitable for legal
research is not primarily algorithmic — it is domain-specific. The core
challenges are about correctly representing the structure of EU legal text:
its evolving XML schema, its distinction between operative and explanatory
content, its recursive cross-references, and the vocabulary gap between
how practitioners write and how users ask.

Each layer of the stack required domain-specific decisions: a two-hop content
fetcher that understands CELLAR's manifest structure, a chunker that respects
article boundaries and sub-chunks definition lists at semicolon boundaries,
a retrieval system that combines semantic similarity with BM25 exact matching,
a generation pipeline that bridges abbreviations via hypothetical expansion
while guarding against hallucinated citations.

The result is a system that answers the majority of question types correctly,
fails gracefully on out-of-scope questions, and is architecturally positioned
to improve incrementally as more of the regulatory corpus is ingested. The
modular command-buffer architecture means that none of the planned improvements
require modifying existing components — new capabilities are strictly additive.
