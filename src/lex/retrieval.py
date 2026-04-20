"""
Retrieval: hybrid search + reranking.

## The two-stage retrieval pattern

Why not just do one vector search and call it a day? Because:

1. **Dense embeddings are approximate.** Cosine similarity between a
   1024-dim query vector and a 1024-dim chunk vector captures
   *semantic* relatedness, but it smooths over exact-term matches. For
   a query like "What does Article 3(2)(a) say?", dense search might
   rank a chunk about a conceptually similar topic above the actual
   Article 3(2)(a) chunk.

2. **Cross-encoders are precise but expensive.** A cross-encoder takes
   (query, chunk) as a single input and emits a single relevance
   score. Much more accurate than cosine between independently-encoded
   vectors — but running it on a whole corpus is prohibitive because
   it scales O(corpus_size) per query.

The classic solution is a funnel:

    corpus (thousands)                                        top-k
          │                                                    │
          ▼                                                    ▼
   ┌──────────────┐     top-20     ┌───────────────┐     rerank score
   │ Vector search│ ──────────── ► │ Cross-encoder │ ────────────────►
   │ (hybrid)     │                │ (reranker)    │
   └──────────────┘                └───────────────┘
         fast                          accurate,
         approximate                   slow per candidate
                                       but only sees 20

## Hybrid search: dense + sparse in one query

Qdrant supports querying multiple named vectors in a single request
via its `query_points` API. We send:

  * a *dense* query: BGE-M3's 1024-dim projection of the user query
  * a *sparse* query: BGE-M3's lexical weights for the same query

and fuse the two result sets with **Reciprocal Rank Fusion (RRF)**.

### RRF, briefly

RRF takes two ranked lists and produces a combined ranking without
needing to compare scores across lists (which usually live on
incompatible scales). For each document d and each list L:

    score(d) = Σ over L:  1 / (k + rank_L(d))

with `k` a constant (we use Qdrant's default). A document highly
ranked in *either* list floats up. A document highly ranked in *both*
floats up the most. It's parameter-free, robust, and correct by
default — exactly what you want before you have eval metrics to tune
against.

## What we return

A `RetrieveResult` containing the top-K reranked chunks, each with
its dense_score, sparse_score, and rerank_score. The scores are there
so callers (especially the UI) can display why a chunk was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import structlog
from qdrant_client import AsyncQdrantClient, models as qmodels

from .commands import Chunk, RetrieveCmd, RetrieveResult, RetrievedChunk
from .config import Settings
from .ingestion import BGEEmbedder   # query-time reuse of the same model

from .tracing import observe

log = structlog.get_logger(__name__)


# ===========================================================================
# Section 1: Reranker
#
# BGE-reranker-v2-m3 is a cross-encoder: you feed it (query, passage)
# and it returns a scalar relevance score. It's ~560M params — bigger
# than an embedding model, smaller than an LLM — and runs fast enough
# on laptop MPS for our 20-candidate funnel (~100ms on an M-series).
#
# Unlike BGE-M3 which we drive ourselves via transformers (to get dense
# + sparse), the reranker has a clean sentence-transformers wrapper
# that does exactly what we want. No reason to roll our own.
# ===========================================================================

class BGEReranker:
    """
    Wraps sentence-transformers' `CrossEncoder` around BGE-reranker-v2-m3.

    The cross-encoder takes query-passage pairs and returns a score per
    pair. Higher = more relevant. The absolute scale is uncalibrated —
    don't compare scores across different queries.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str = "auto",
    ) -> None:
        # Lazy import to keep `import lex` fast.
        from sentence_transformers import CrossEncoder
        import torch
        log.info("reranker.loading", model=model_name)

        # Device selection: honor explicit setting, else auto-detect.
        # On 8 GB GPUs hosting an LLM + embedder, the reranker may not
        # fit — setting this to "cpu" trades ~1-2s per query for
        # guaranteed fit. Defaults to "auto" for backward compat with
        # higher-VRAM machines.
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # `trust_remote_code=True` for the same reason as the embedder:
        # BGE ships custom tokenizer plumbing.
        self._model = CrossEncoder(
            model_name, device=device, trust_remote_code=True,
        )
        # Cast to fp16 on accelerator devices. Halves memory with
        # essentially no accuracy impact for reranking, and avoids
        # MPS out-of-memory crashes when other processes are competing
        # for the shared memory pool.
        if device in {"mps", "cuda"}:
            self._model.model.half()
        self._batch_size = batch_size
        log.info("reranker.loaded", model=model_name, device=device)

    def rerank(
        self, query: str, passages: list[str]
    ) -> list[float]:
        """
        Score each passage against the query. Order is preserved —
        result[i] is the score for passages[i]. Caller sorts.
        """
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        # `predict` returns numpy array; convert to plain floats for
        # downstream JSON-serialisability.
        return [float(s) for s in scores]


# ===========================================================================
# Section 2: Hybrid search against Qdrant
#
# Qdrant's `query_points` call with a `prefetch` list + a fusion query
# does dense+sparse retrieval in one round trip. Conceptually:
#
#   prefetch: [dense-search, sparse-search]   # each returns N candidates
#   query:    rrf-fuse                        # merges into one ranking
#
# Our dense and sparse vectors are both keyed by the same names we
# declared at ingestion time ("dense", "sparse"). The collection knows
# how to match them up with its payload.
# ===========================================================================

def _build_filter(filters) -> qmodels.Filter | None:
    """
    Translate our `RetrieveFilter` into a Qdrant Filter.

    All fields are optional and AND-ed together. We always include
    `language` (every RetrieveFilter has one), plus any of `celex_id`
    and `article` that are set.

    Returns None if — somehow — every field ended up empty. Qdrant
    accepts `filter=None` meaning "no filter."
    """
    conditions: list[qmodels.FieldCondition] = []

    if filters.language:
        conditions.append(qmodels.FieldCondition(
            key="language",
            match=qmodels.MatchValue(value=filters.language),
        ))
    if filters.celex_id:
        conditions.append(qmodels.FieldCondition(
            key="celex_id",
            match=qmodels.MatchValue(value=filters.celex_id),
        ))
    if filters.article:
        conditions.append(qmodels.FieldCondition(
            key="article",
            match=qmodels.MatchValue(value=filters.article),
        ))

    return qmodels.Filter(must=conditions) if conditions else None


async def _hybrid_search(
    client: AsyncQdrantClient,
    collection: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    limit: int,
    qfilter: qmodels.Filter | None,
) -> list[qmodels.ScoredPoint]:
    """
    One Qdrant call, two prefetches, RRF fusion.

    `query_points` is the modern Qdrant query API. We structure it as:

      prefetch = [
        { query: dense_vec,  using: "dense",  limit: N },
        { query: sparse_vec, using: "sparse", limit: N },
      ]
      query = Fusion(RRF)

    Qdrant runs both prefetch branches, then fuses their results into a
    single ranked list which it trims to `limit`. Filters apply to both
    branches.
    """
    # Sparse vector requires parallel indices/values lists — same shape
    # we used at write time.
    sparse_query = qmodels.SparseVector(
        indices=list(sparse_vec.keys()),
        values=list(sparse_vec.values()),
    )

    result = await client.query_points(
        collection_name=collection,
        prefetch=[
            qmodels.Prefetch(
                query=dense_vec,
                using="dense",
                limit=limit,
                filter=qfilter,
            ),
            qmodels.Prefetch(
                query=sparse_query,
                using="sparse",
                limit=limit,
                filter=qfilter,
            ),
        ],
        # RRF is parameter-free and robust; no weight tuning needed.
        # Switch to DBSF here if/when eval shows we need calibrated fusion.
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,   # we only need payloads for display
    )
    return result.points


# ===========================================================================
# Section 3: Scoring breakdown (for UI transparency)
#
# After RRF fusion, each returned point has a single fused score but
# we've lost the individual dense/sparse contributions. For UX — "why
# did this chunk rank here?" — we do two lightweight companion queries
# to recover the per-head scores. These are cheap (Qdrant just HNSW-
# traverses the already-indexed vectors) and let us show richer
# citation cards.
# ===========================================================================

async def _score_breakdown(
    client: AsyncQdrantClient,
    collection: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    point_ids: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Get per-head (dense, sparse) scores for a specific set of points.

    Returns two dicts: {point_id: dense_score}, {point_id: sparse_score}.
    Missing entries default to 0.0 at the call site.
    """
    if not point_ids:
        return {}, {}

    # The `query_points` API doesn't have a native "score these specific
    # IDs" mode — it returns the top-N matches. So we over-fetch a
    # top-K big enough to almost certainly contain our IDs, then
    # project. For K=50 this is plenty for our 20-candidate pool.
    over_fetch = max(50, 2 * len(point_ids))

    dense_hits = await client.query_points(
        collection_name=collection,
        query=dense_vec, using="dense",
        limit=over_fetch, with_payload=False, with_vectors=False,
    )
    sparse_hits = await client.query_points(
        collection_name=collection,
        query=qmodels.SparseVector(
            indices=list(sparse_vec.keys()),
            values=list(sparse_vec.values()),
        ),
        using="sparse",
        limit=over_fetch, with_payload=False, with_vectors=False,
    )

    dense_scores = {str(p.id): float(p.score) for p in dense_hits.points}
    sparse_scores = {str(p.id): float(p.score) for p in sparse_hits.points}
    return dense_scores, sparse_scores


# ===========================================================================
# Section 4: Handler
#
# Pulls the query, runs the pipeline, returns a RetrieveResult.
# ===========================================================================

@dataclass
class RetrieveDeps:
    """Dependencies for the retrieve handler, bundled for test injection."""
    settings: Settings
    embedder: BGEEmbedder
    reranker: BGEReranker
    qdrant: AsyncQdrantClient

@observe(name="retrieve")
async def handle_retrieve(cmd: RetrieveCmd, deps: RetrieveDeps) -> RetrieveResult:
    """
    Execute hybrid retrieval + reranking for one query.

    Steps:
      1. Encode the query → (dense_vec, sparse_vec).
      2. Hybrid search Qdrant with RRF fusion → top-K_dense candidates.
      3. Fetch per-head scores for those candidates (for UI display).
      4. Rerank all candidates against the query with the cross-encoder.
      5. Sort by rerank score, keep top_k_rerank.
      6. Return a RetrieveResult with full metadata.
    """
    cmd_id = str(cmd.cmd_id)
    logger = log.bind(cmd_id=cmd_id, query=cmd.query[:80])

    # ---- 1. Embed query ------------------------------------------------
    # Single-item batch — runs in a thread pool because it's CPU/GPU bound
    # and we don't want to block the event loop on short queries.
    import asyncio
    dense_list, _ = await asyncio.to_thread(
        deps.embedder.embed, [cmd.query]
    )
    dense_vec = dense_list[0]
    sparse_vec = deps.embedder.query_sparse(
        cmd.query,
        idf_load_path=deps.settings.bm25_idf_path(cmd.filters.language),
    )

    # ---- 2. Hybrid search ---------------------------------------------
    collection = deps.settings.collection_name(cmd.filters.language)
    qfilter = _build_filter(cmd.filters)

    # Over-fetch from Qdrant so the reranker has enough candidates to
    # work with. Spec default is 20 (top_k_dense).
    fused = await _hybrid_search(
        client=deps.qdrant,
        collection=collection,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        limit=deps.settings.retrieval.top_k_dense,
        qfilter=qfilter,
    )
    logger.info("retrieval.hybrid", candidates=len(fused))

    if not fused:
        # No matches at all — return an empty result rather than fail.
        return RetrieveResult(cmd_id=cmd.cmd_id, query=cmd.query, chunks=[])

    # ---- 3. Per-head score breakdown for UI display -------------------
    dense_scores, sparse_scores = await _score_breakdown(
        client=deps.qdrant,
        collection=collection,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        point_ids=[str(p.id) for p in fused],
    )

    # ---- 4. Rerank ----------------------------------------------------
    passages = [p.payload["text"] for p in fused]
    rerank_scores = await asyncio.to_thread(
        deps.reranker.rerank, cmd.query, passages
    )
    logger.info("retrieval.reranked", candidates=len(rerank_scores))

    # Pair each fused point with its rerank score and sort.
    scored: list[tuple[qmodels.ScoredPoint, float]] = list(zip(fused, rerank_scores))
    CHUNK_TYPE_BOOST: dict[str, float] = {
    "paragraph": 1.00,
    "article":   1.00,
    "annex":     0.90,
    "recital":   0.85,
    }
    scored = [
    (point, score * CHUNK_TYPE_BOOST.get(point.payload.get("chunk_type", "recital"), 0.85))
    for point, score in scored
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # ---- 5. Keep top-K ------------------------------------------------
    top_k = cmd.top_k or deps.settings.retrieval.top_k_rerank
    top = scored[:top_k]

    # ---- 6. Build result ----------------------------------------------
    chunks: list[RetrievedChunk] = []
    for point, rerank_score in top:
        pid = str(point.id)
        # Reconstruct a Chunk from the payload we stored at ingest time.
        # Payload is just the Chunk's model_dump(), so model_validate
        # reverses it cleanly.
        chunk = Chunk.model_validate(point.payload)
        chunks.append(RetrievedChunk(
            chunk=chunk,
            dense_score=dense_scores.get(pid, 0.0),
            sparse_score=sparse_scores.get(pid, 0.0),
            rerank_score=rerank_score,
        ))

    return RetrieveResult(cmd_id=cmd.cmd_id, query=cmd.query, chunks=chunks)


# ===========================================================================
# Section 5: Factory
#
# Builds real dependencies. Tests skip this and construct RetrieveDeps
# directly with fakes.
# ===========================================================================
def build_retrieve_deps(settings: Settings) -> RetrieveDeps:
    """
    Build retrieval dependencies.
    
    Uses ModelClient (model server daemon) if the socket exists,
    otherwise falls back to in-process model loading.
    The interface is identical either way — callers don't need to know.
    """
    from .model_server import ModelClient

    client = ModelClient(settings.model_socket_path())
    if client.available:
        log.info("retrieval.using_model_server",
                 socket=str(settings.model_socket_path()))
        embedder = client      # ModelClient has same .embed() / .query_sparse() interface
        reranker = client      # ModelClient has same .rerank() interface
    else:
        log.info("retrieval.using_inprocess_models")
        embedder = BGEEmbedder(
            model_name=settings.embedding.model,
            batch_size=settings.embedding.batch_size,
            device=settings.embedding.device,
        )
        reranker = BGEReranker(
            model_name=settings.reranker.model,
            batch_size=settings.reranker.batch_size,
            device=settings.reranker.device,
        )

    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    return RetrieveDeps(
        settings=settings,
        embedder=embedder,
        reranker=reranker,
        qdrant=qdrant,
    )
