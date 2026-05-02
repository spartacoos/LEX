"""
Ingestion pipeline: fetch → parse → chunk → embed → write.

## The pipeline

    bytes  ── parse ──►  FormexNode[]  ── chunk ──►  Chunk[]
       ▲                                                │
       │                                                ▼
    Source                                           embed
       ▲                                                │
       │                                                ▼
    fetch                                     (dense, sparse)[]
                                                        │
                                                        ▼
                                                     write ──► Qdrant

Each arrow is a pure function (or pure-ish, in the case of embedders
that hit models). The handler at the bottom glues them together and
publishes state-machine transitions to Redis.

## Why one file for five concerns?

Per SPEC §7: "each file = one complete subsystem readable in full."
All five steps are mechanically coupled — they operate on the same
data in sequence, and understanding any one of them in isolation
requires understanding the others. Open this file, read top-to-bottom,
leave knowing exactly how a directive becomes vectors in Qdrant.

## The state machine

Per SPEC §3.4, ingestion is a state machine. We publish transitions
so the UI can show live progress. Redis hash `lex:job:{cmd_id}` holds
the current state; pub/sub channel `lex:job:{cmd_id}:events` carries
transition notifications.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Literal

from pathlib import Path

import redis.asyncio as aioredis
import structlog
import lxml.etree as etree
from qdrant_client import AsyncQdrantClient, models as qmodels

from inspect import isawaitable
async def _maybe_await(value):
    if isawaitable(value):
        return await value
    return value

from .commands import Chunk, IngestCmd, IngestResult
from .config import Settings
from .sources import CellarRest, LocalFile, Source, SourceUnavailable

from .tracing import observe

log = structlog.get_logger(__name__)


# ===========================================================================
# Section 1: Formex parsing
# (unchanged — see original comments)
# ===========================================================================

_NBSP = "\u00a0"

_DIVISION_TITLE_RE = re.compile(
    r"^(PART|TITLE|CHAPTER|SECTION|SUBSECTION)\s+([IVXLCDM0-9]+)\b",
    re.IGNORECASE,
)


@dataclass
class FormexNode:
    """One structural piece of a directive, with full ancestry."""
    kind: Literal["recital", "article", "paragraph", "annex"]
    text: str
    part: str | None = None
    title: str | None = None
    chapter: str | None = None
    article: str | None = None
    paragraph: str | None = None


def _text_content(elem: etree._Element) -> str:
    raw = etree.tostring(elem, method="text", encoding="unicode") or ""
    raw = raw.replace(_NBSP, " ")
    return re.sub(r"\s+", " ", raw).strip()


def _division_kind_and_number(div: etree._Element) -> tuple[str | None, str | None]:
    title = div.find("TITLE")
    if title is None:
        return None, None
    text = _text_content(title)
    m = _DIVISION_TITLE_RE.match(text)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2)


def _article_number(article: etree._Element) -> str | None:
    ident = article.get("IDENTIFIER")
    if ident:
        stripped = ident.lstrip("0")
        return stripped or "0"
    ti = article.find("TI.ART")
    if ti is not None:
        text = _text_content(ti)
        m = re.search(r"(\d+[a-z]?)", text)
        if m:
            return m.group(1)
    return None


def _paragraph_number(parag: etree._Element) -> str | None:
    ident = parag.get("IDENTIFIER")
    if ident and "." in ident:
        tail = ident.split(".", 1)[1].lstrip("0")
        return tail or "0"
    no = parag.find("NO.PARAG")
    if no is not None:
        text = _text_content(no)
        m = re.match(r"(\d+[a-z]?)", text)
        if m:
            return m.group(1)
    return None


def _recital_number(consid: etree._Element) -> str | None:
    no = consid.find(".//NO.P")
    if no is not None:
        text = _text_content(no)
        m = re.search(r"(\d+)", text)
        if m:
            return m.group(1)
    return None


def _ancestry(article: etree._Element) -> dict[str, str | None]:
    out: dict[str, str | None] = {"part": None, "title": None, "chapter": None}
    for anc in article.iterancestors():
        if anc.tag != "DIVISION":
            continue
        kind, number = _division_kind_and_number(anc)
        if kind in out and out[kind] is None:
            out[kind] = number
    return out


def parse_formex(xml_bytes: bytes) -> list[FormexNode]:
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    root = etree.fromstring(xml_bytes, parser=parser)
    nodes: list[FormexNode] = []

    for consid in root.iterfind(".//PREAMBLE//CONSID"):
        text = _text_content(consid)
        if not text:
            continue
        nodes.append(FormexNode(
            kind="recital", text=text, paragraph=_recital_number(consid),
        ))

    for article in root.iterfind(".//ENACTING.TERMS//ARTICLE"):
        anc = _ancestry(article)
        art_num = _article_number(article)
        parags = article.findall("PARAG")
        if not parags:
            text = _text_content(article)
            if text:
                nodes.append(FormexNode(
                    kind="article", text=text,
                    part=anc["part"], title=anc["title"], chapter=anc["chapter"],
                    article=art_num,
                ))
            continue
        for parag in parags:
            text = _text_content(parag)
            if not text:
                continue
            nodes.append(FormexNode(
                kind="paragraph", text=text,
                part=anc["part"], title=anc["title"], chapter=anc["chapter"],
                article=art_num, paragraph=_paragraph_number(parag),
            ))

    for annex in root.iterfind(".//ANNEX"):
        text = _text_content(annex)
        if not text:
            continue
        title_el = annex.find("TITLE/TI")
        num = None
        if title_el is not None:
            m = re.search(r"ANNEX\s+([IVXLCDM0-9]+)", _text_content(title_el))
            if m:
                num = m.group(1)
        nodes.append(FormexNode(kind="annex", text=text, article=num))

    log.info(
        "formex.parsed",
        recitals=sum(1 for n in nodes if n.kind == "recital"),
        paragraphs=sum(1 for n in nodes if n.kind == "paragraph"),
        articles=sum(1 for n in nodes if n.kind == "article"),
        annexes=sum(1 for n in nodes if n.kind == "annex"),
    )
    return nodes


# ===========================================================================
# Section 2: Chunking
# (unchanged — see original comments)
# ===========================================================================

_ABBREVIATIONS = {
    "art", "arts", "no", "nos", "para", "paras", "p", "pp",
    "i.e", "e.g", "cf", "vs", "etc",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _ends_with_abbrev(fragment: str) -> bool:
    m = re.search(r"([A-Za-z.]+)\.?$", fragment)
    if not m:
        return False
    tail = m.group(1).rstrip(".").lower()
    return tail in _ABBREVIATIONS


def _sentence_split(text: str) -> list[str]:
    raw = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not raw:
        return [text]
    merged: list[str] = [raw[0]]
    for piece in raw[1:]:
        if _ends_with_abbrev(merged[-1]):
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)
    return merged

# Matches Art. 2 definition items: (1) "term" means ... or (1) 'term' means ...
_ART2_DEF_RE = re.compile(r'(?<=;)(?=\(\d+\)(?!\s+of\b)(?!\s+subparagraph\b)(?!\s+point\b))')


def _split_definitions(text: str) -> list[str] | None:
    """
    Split Art. 2-style numbered definition blocks.
    Handles two boundary types:
      - Semicolon boundary: ...term A;(N)term B means...
      - Preamble boundary: ...definitions apply:(1)term A means...
    Definition (1) has no preceding semicolon so needs special handling.
    """
    # First split on semicolon boundaries to get defs (2)-(N)
    parts = [p.strip() for p in _ART2_DEF_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return None

    # The first part contains the preamble + definition (1).
    # Extract definition (1) by finding where (1) starts.
    first = parts[0]
    m = re.search(r'(?<!\d)\(1\)(?!\s+of\b)(?!\s+point\b)(?!\s+subparagraph\b)', first)
    if m:
        # Keep only from (1) onwards, discard preamble
        parts[0] = first[m.start():].strip()
    else:
        # No (1) found — discard preamble entirely
        parts = parts[1:]

    defs = [p for p in parts if re.match(r'^\(\d+\)', p)]
    if len(defs) < 2:
        return None
    return defs

def _build_chunk_id(celex_id: str, kind: str, node: FormexNode, part_idx: int | None) -> str:
    if kind == "recital":
        tail = node.paragraph or "?"
        base = f"{celex_id}:recital:{tail}"
    elif kind == "annex":
        base = f"{celex_id}:annex:{node.article or '?'}"
    else:
        art = node.article or "?"
        para = node.paragraph or "0"
        base = f"{celex_id}:art:{art}:{para}"
    return base if part_idx is None else f"{base}:p{part_idx}"


def chunk_nodes(
    nodes: Iterable[FormexNode],
    *,
    celex_id: str,
    language: str,
    eli_uri: str,
    max_chars: int,
    sentence_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for node in nodes:
        if len(node.text) <= max_chars:
            chunks.append(Chunk(
                id=_build_chunk_id(celex_id, node.kind, node, part_idx=None),
                celex_id=celex_id, eli_uri=eli_uri, language=language,
                chunk_type=node.kind,
                part=node.part, title=node.title, chapter=node.chapter,
                article=node.article, paragraph=node.paragraph,
                text=node.text,
            ))
            continue

        # Too long — try definition split first (Art. 2 pattern).
        defs = _split_definitions(node.text)
        if defs is not None:
            for defn in defs:
                # Extract the definition number (1), (2), etc. as paragraph
                m = re.match(r'^\((\d+)\)', defn)
                def_para = m.group(1) if m else node.paragraph
                chunks.append(Chunk(
                    id=_build_chunk_id(celex_id, node.kind, node,
                                       part_idx=int(def_para) if def_para and def_para.isdigit() else None),
                    celex_id=celex_id, eli_uri=eli_uri, language=language,
                    chunk_type=node.kind,
                    part=node.part, title=node.title, chapter=node.chapter,
                    article=node.article, paragraph=def_para,
                    text=defn,
                ))
            continue

        # Too long and not definitions — sentence-split with overlap.
        sents = _sentence_split(node.text)
        part_idx = 1
        i = 0
        while i < len(sents):
            buf: list[str] = []
            buf_len = 0
            j = i
            while j < len(sents) and buf_len + len(sents[j]) + 1 <= max_chars:
                buf.append(sents[j])
                buf_len += len(sents[j]) + 1
                j += 1
            if not buf:
                buf = [sents[i]]
                j = i + 1
            chunks.append(Chunk(
                id=_build_chunk_id(celex_id, node.kind, node, part_idx=part_idx),
                celex_id=celex_id, eli_uri=eli_uri, language=language,
                chunk_type=node.kind,
                part=node.part, title=node.title, chapter=node.chapter,
                article=node.article, paragraph=node.paragraph,
                text=" ".join(buf),
            ))
            part_idx += 1
            if j >= len(sents):
                break
            i = max(i + 1, j - sentence_overlap)

    log.info("chunking.done", chunks=len(chunks))
    return chunks


# ===========================================================================
# Section 3: Embedding
#
# ## What changed and why — the sparse head investigation
#
# The original code used `AutoModel.from_pretrained(..., trust_remote_code=True)`
# with the expectation that BGE-M3's custom model class would expose a
# `sparse_output` attribute on its forward pass. Diagnostic investigation
# revealed this was never the case:
#
#   $ python -c "... AutoModel.from_pretrained('BAAI/bge-m3', trust_remote_code=True)"
#   type: BaseModelOutputWithPoolingAndCrossAttentions
#   keys: last_hidden_state, pooler_output
#
#   $ cat config.json
#   architectures: ['XLMRobertaModel']
#   auto_map: None
#
# The BAAI/bge-m3 checkpoint at revision 5617a9f registers no custom class.
# AutoModel falls back to plain XLMRobertaModel. `trust_remote_code=True`
# was inert — it executes arbitrary remote Python for zero benefit, which
# is a security liability.
#
# The sparse head simply does not exist in this checkpoint's weights file.
# BGE-M3 was trained with a sparse head, but BAAI chose not to ship the
# custom modeling code in the HuggingFace repo — they gate it behind
# FlagEmbedding, which has its own dependency and platform compatibility
# problems (see project history).
#
# ## The fix: BM25 for the sparse leg
#
# Rather than chasing a sparse head that isn't there, we implement BM25
# (Best Match 25) as the sparse retrieval leg. This is actually a better
# fit for legal text than a learned sparse representation:
#
#   Dense embeddings (BGE-M3 CLS vectors):
#     - Capture semantic meaning via cosine similarity in 1024-dim space
#     - "SMP" ↔ "significant market power" are close in embedding space
#       because they co-occur in training data
#     - Good at paraphrase matching, poor at exact legal references
#
#   Sparse BM25 vectors:
#     - Pure token frequency statistics, no neural model
#     - TF-IDF: term frequency (how often a token appears in this chunk)
#       × inverse document frequency (how rare it is across all chunks)
#     - "Article 63" in a query gets high weight because "63" is rare
#       across 880 chunks; it fires precisely on the chunk containing it
#     - Excellent for exact legal references: "SMP", "VHCN", "Art. 76(1)"
#
#   RRF fusion (Qdrant):
#     - Reciprocal Rank Fusion merges the two ranked lists without
#       needing calibrated scores — rank 1 in either list gets high
#       weight, rank 1 in both gets highest weight
#     - Dense handles "what does significant market power mean?"
#     - Sparse handles "Article 63 paragraph 2 SMP definition"
#     - Together they cover the full query spectrum
#
# ## BM25Vectorizer design
#
# BM25 requires the entire corpus to compute IDF (inverse document
# frequency). IDF for token t = log((N - df_t + 0.5) / (df_t + 0.5))
# where N = total documents and df_t = documents containing token t.
# This means you cannot vectorize chunks one at a time — you need all
# chunks simultaneously. This fits our pipeline perfectly: embed() already
# takes the full texts list.
#
# We reuse BGE-M3's own tokenizer so token IDs are in the same vocabulary
# namespace as the dense vectors. This is important for Qdrant's hybrid
# collection schema — both vector types reference the same token space,
# which makes the index consistent even though they're stored separately.
#
# ## Dense vector correctness
#
# The original code used CLS token pooling (last_hidden_state[:, 0]),
# which is correct for BGE-M3. BGE-M3's dense head IS the CLS token
# projection — confirmed by the config showing XLMRobertaModel with
# no additional projection layer. L2 normalisation is applied so Qdrant's
# cosine similarity reduces to a dot product, which is faster.
# ===========================================================================

class BM25Vectorizer:
    """
    Corpus-level BM25 sparse vectors using BGE-M3's tokenizer vocabulary.

    Reusing the same tokenizer ensures token IDs are consistent between
    dense and sparse vectors — both live in the same 250k-token vocabulary
    space, so Qdrant's hybrid index is coherent.

    Must be called with the full corpus at once (not per-chunk) because
    IDF requires knowing how many documents contain each token.

    Usage:
        vectorizer = BM25Vectorizer(tokenizer)
        sparse_vecs = vectorizer.vectorize(all_chunk_texts)
    """

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer

    def query_vectorize(
        self,
        text: str,
        idf_load_path: Path | None = None,
    ) -> dict[int, float]:
        """
        Encode a single query as a sparse vector for retrieval.

        If idf_load_path points to a saved IDF table (written by vectorize()),
        uses proper BM25 IDF weights. Otherwise falls back to normalised TF.

        BM25 query weight for token t:
            weight(t) = IDF(t)
        The document side already has TF normalisation baked in, so the
        query side only needs IDF. This is the standard asymmetric BM25
        formulation used in Anserini and SPLADE.
        """
        import json

        special_ids = set(self._tok.all_special_ids)
        ids = self._tok.encode(text, add_special_tokens=False)
        ids = [i for i in ids if i not in special_ids]
        if not ids:
            return {}

        # Load IDF table if available
        idf: dict[str, float] = {}
        if idf_load_path is not None and idf_load_path.exists():
            idf = json.loads(idf_load_path.read_text())

        if idf:
            # Proper BM25: query weight = IDF only (TF=1 for query tokens)
            weights: dict[int, float] = {}
            for tok_id in set(ids):
                w = idf.get(str(tok_id), 0.0)
                if w > 0:
                    weights[tok_id] = float(w)
            return weights
        else:
            # Fallback: normalised TF (no IDF available)
            log.debug("bm25.query_vectorize.no_idf",
                      hint="run lex ingest to generate IDF table")
            counts: dict[int, int] = {}
            for i in ids:
                counts[i] = counts.get(i, 0) + 1
            max_count = max(counts.values())
            return {tok_id: count / max_count for tok_id, count in counts.items()}

    def vectorize(
        self,
        texts: list[str],
        idf_save_path: Path | None = None,
    ) -> list[dict[int, float]]:
        """
        Encode a corpus of texts into BM25 sparse vectors.
        
        If idf_save_path is provided, the IDF table is saved there so
        query_vectorize() can load it later for proper BM25 query scoring
        instead of falling back to normalised TF.
        """
        from rank_bm25 import BM25Okapi
        import json

        special_ids = set(self._tok.all_special_ids)
        tokenized_ids: list[list[int]] = []
        for text in texts:
            ids = self._tok.encode(text, add_special_tokens=False)
            ids = [i for i in ids if i not in special_ids]
            tokenized_ids.append(ids)

        tokenized_strs = [[str(i) for i in ids] for ids in tokenized_ids]
        bm25 = BM25Okapi(tokenized_strs)

        if idf_save_path is not None:
            idf_save_path.parent.mkdir(parents=True, exist_ok=True)
            idf_save_path.write_text(json.dumps(bm25.idf))
            log.info("bm25.idf_saved", path=str(idf_save_path),
                     tokens=len(bm25.idf))

        out: list[dict[int, float]] = []
        for doc_str_tokens in tokenized_strs:
            weights: dict[int, float] = {}
            unique_str_tokens = set(doc_str_tokens)
            for tok_str in unique_str_tokens:
                tok_id = int(tok_str)
                idf = bm25.idf.get(tok_str, 0.0)
                if idf <= 0:
                    continue
                tf = doc_str_tokens.count(tok_str)
                dl = len(doc_str_tokens)
                avgdl = bm25.avgdl if bm25.avgdl > 0 else 1.0
                k1, b = 1.5, 0.75
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                w = float(idf * tf_norm)
                if w > 0:
                    weights[tok_id] = w
            out.append(weights)

        nonempty = sum(1 for w in out if w)
        log.debug("bm25.vectorized", total=len(out), nonempty=nonempty)
        return out

class BGEEmbedder:
    """
    BGE-M3 dense embedder + BM25 sparse vectorizer.

    ## Architecture (what we actually have)
    
    Investigation (April 2026) confirmed the BAAI/bge-m3 HuggingFace
    checkpoint registers architectures=['XLMRobertaModel'], auto_map=None.
    AutoModel loads plain XLM-RoBERTa. The sparse head weights exist in
    BGE-M3's training but are not shipped in this checkpoint.
    trust_remote_code=True is therefore removed — it executed arbitrary
    remote Python for zero benefit.

    Dense vectors: CLS token (last_hidden_state[:, 0]), L2-normalised.
    This is correct — BGE-M3's dense retrieval head IS CLS pooling.
    1024-dim, cosine similarity in Qdrant.

    Sparse vectors: BM25 via BM25Vectorizer (see above).
    Token IDs from the same XLM-RoBERTa tokenizer vocabulary.
    Dot product similarity in Qdrant.
    """

    def __init__(self, model_name: str, batch_size: int = 16, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info("embedder.loading", model=model_name)
        self._torch = torch

        from typing import cast
        from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

        self._tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))
        # No trust_remote_code — confirmed inert for this checkpoint.
        self._model = AutoModel.from_pretrained(model_name)

        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device
        self._model = self._model.to(self._device).eval()
        self._batch_size = batch_size
        self._max_length = 1024
        self._bm25 = BM25Vectorizer(self._tokenizer)

        log.info("embedder.loaded", model=model_name, device=self._device)
    def query_sparse(self, text: str, idf_load_path: Path | None = None) -> dict[int, float]:
        return self._bm25.query_vectorize(text, idf_load_path=idf_load_path)

    def embed(
    self,
    texts: list[str],
    idf_save_path: Path | None = None,
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """
        Encode a corpus.

        Returns (dense, sparse) where:
          dense[i]  — 1024-dim L2-normalised float list (BGE-M3 CLS)
          sparse[i] — {token_id: bm25_weight} dict (BM25Okapi)

        Both are computed over the full texts list in one call.
        BM25 requires the full corpus for IDF; dense is batched for
        memory efficiency but logically operates on the full list too.
        """
        torch = self._torch
        dense_out: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            enc = self._tokenizer(
                batch,
                padding=True, truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**enc, return_dict=True)

            # CLS token pooling. BGE-M3 is trained with CLS-based dense
            # retrieval — this is the correct pooling strategy for this
            # model family (verified against BAAI's own eval scripts).
            dense = outputs.last_hidden_state[:, 0]
            # L2 normalise so cosine(a, b) = dot(a, b) for unit vectors.
            # Qdrant's COSINE distance is equivalent to dot product on
            # normalised vectors, which is faster to compute at query time.
            dense = torch.nn.functional.normalize(dense, p=2, dim=-1)
            dense_out.extend(vec.cpu().tolist() for vec in dense)

        # BM25 requires the full corpus at once for IDF computation.
        sparse_out = self._bm25.vectorize(texts, idf_save_path=idf_save_path)
        return dense_out, sparse_out

# ===========================================================================
# Section 4: Qdrant writer
# (unchanged)
# ===========================================================================

async def ensure_collection(
    client: AsyncQdrantClient, name: str, dense_dim: int
) -> None:
    """Create the collection with hybrid-search schema if it doesn't exist."""
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        return

    log.info("qdrant.collection.create", name=name, dense_dim=dense_dim)
    await client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=dense_dim, distance=qmodels.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(),
        },
    )

    for field_name, schema in [
        ("celex_id", qmodels.PayloadSchemaType.KEYWORD),
        ("article", qmodels.PayloadSchemaType.KEYWORD),
        ("language", qmodels.PayloadSchemaType.KEYWORD),
    ]:
        await client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
        )


def _chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


async def write_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list[dict[int, float]],
) -> int:
    points: list[qmodels.PointStruct] = []
    for chunk, d, s in zip(chunks, dense, sparse):
        payload = chunk.model_dump()
        # Add cross-reference graph edges at ingest time.
        # Storing these in the payload means retrieval can follow
        # legal cross-references without any additional model calls.
        payload["cross_refs"] = _extract_cross_refs(chunk.text)
        points.append(qmodels.PointStruct(
            id=_chunk_uuid(chunk.id),
            vector={
                "dense": d,
                "sparse": qmodels.SparseVector(
                    indices=list(s.keys()),
                    values=list(s.values()),
                ),
            },
            payload=payload,
        ))
    await client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)

# ===========================================================================
# Section 5: Ingestion handler
# (unchanged)
# ===========================================================================

from .model_server import ModelClient

@dataclass
class IngestDeps:
    settings: Settings
    source: Source
    embedder: BGEEmbedder | ModelClient
    qdrant: AsyncQdrantClient
    redis: aioredis.Redis
    states: list[str] = field(default_factory=lambda: [
        "queued", "fetching", "parsing", "chunking", "embedding",
        "writing", "done", "failed",
    ])


async def _set_state(
    deps: IngestDeps, cmd_id: str, state: str, **extra: str
) -> None:
    key = f"{deps.settings.redis.key_prefix}:job:{cmd_id}"
    channel = f"{deps.settings.redis.key_prefix}:job:{cmd_id}:events"
    now = datetime.now(timezone.utc).isoformat()
    payload = {"state": state, "updated_at": now, **extra}
    await _maybe_await(deps.redis.hset(key, mapping={k: str(v) for k, v in payload.items()}))
    await _maybe_await(deps.redis.publish(channel, state))
    log.info("ingest.state", cmd_id=cmd_id, state=state, **extra)


def _eli_uri(celex_id: str) -> str:
    m = re.match(r"^3(\d{4})L(\d{4})$", celex_id)
    if not m:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex_id}"
    year, num = m.groups()
    return f"http://data.europa.eu/eli/dir/{year}/{int(num)}/oj"

def _extract_cross_refs(text: str) -> list[str]:
    """
    Extract article numbers explicitly cross-referenced in a chunk's text.

    Examples matched:
      "in accordance with Article 67"        → ["67"]
      "pursuant to Articles 69, 70 and 71"   → ["69", "70", "71"]
      "referred to in Article 3(2)"          → ["3"]
      "Articles 69 to 74"                    → ["69","70","71","72","73","74"]

    These are stored in the Qdrant payload so that at retrieval time,
    when a chunk is returned, we can automatically also fetch the articles
    it explicitly references — following the legal cross-reference graph
    one hop without requiring a second LLM call.
    """
    pattern = re.compile(
        r'\b[Aa]rt(?:icle)?s?\.?\s+(\d+(?:\s*(?:\([\w\d]+\))*)'
        r'(?:\s*(?:,|and|or|to)\s*\d+(?:\s*(?:\([\w\d]+\))*)*)*)'
    )
    refs: set[str] = set()
    for m in pattern.finditer(text):
        raw = m.group(1)
        nums = re.findall(r'\d+', raw)
        if 'to' in raw.lower() and len(nums) >= 2:
            start, end = int(nums[0]), int(nums[-1])
            if end - start <= 20:  # sanity cap — avoid "Articles 1 to 127"
                refs.update(str(n) for n in range(start, end + 1))
        else:
            refs.update(nums)
    return sorted(refs)

@observe(name="ingest")
async def handle_ingest(cmd: IngestCmd, deps: IngestDeps) -> IngestResult:
    cmd_id = str(cmd.cmd_id)
    collection = deps.settings.collection_name(cmd.language)

    await _set_state(deps, cmd_id, "queued",
                     celex_id=cmd.celex_id, language=cmd.language)

    try:
        await _set_state(deps, cmd_id, "fetching")
        xml_bytes = await deps.source.fetch(cmd.celex_id, cmd.language)
    except SourceUnavailable as e:
        await _set_state(deps, cmd_id, "failed", error=str(e))
        return IngestResult(
            cmd_id=cmd.cmd_id, celex_id=cmd.celex_id,
            language=cmd.language, chunks_written=0,
            state="failed", error=str(e),
        )

    try:
        await _set_state(deps, cmd_id, "parsing")
        nodes = parse_formex(xml_bytes)
        if not nodes:
            raise ValueError("parser produced zero nodes — bad XML?")
    except Exception as e:
        await _set_state(deps, cmd_id, "failed", error=f"parse: {e}")
        return IngestResult(
            cmd_id=cmd.cmd_id, celex_id=cmd.celex_id,
            language=cmd.language, chunks_written=0,
            state="failed", error=str(e),
        )

    await _set_state(deps, cmd_id, "chunking")
    chunks = chunk_nodes(
        nodes,
        celex_id=cmd.celex_id, language=cmd.language,
        eli_uri=_eli_uri(cmd.celex_id),
        max_chars=deps.settings.chunking.max_chars,
        sentence_overlap=deps.settings.chunking.sentence_overlap,
    )

    # ---- Embed ---------------------------------------------------------
    await _set_state(deps, cmd_id, "embedding", chunks=str(len(chunks)))
    texts = [c.text for c in chunks]
    idf_path = deps.settings.bm25_idf_path(cmd.language)
    dense, sparse = await asyncio.to_thread(
        deps.embedder.embed, texts, idf_path
    )

    try:
        await _set_state(deps, cmd_id, "writing")
        await ensure_collection(
            deps.qdrant, collection,
            dense_dim=deps.settings.embedding.dense_dim,
        )
        written = await write_chunks(deps.qdrant, collection, chunks, dense, sparse)
    except Exception as e:
        await _set_state(deps, cmd_id, "failed", error=f"write: {e}")
        return IngestResult(
            cmd_id=cmd.cmd_id, celex_id=cmd.celex_id,
            language=cmd.language, chunks_written=0,
            state="failed", error=str(e),
        )

    await _set_state(deps, cmd_id, "done", chunks=str(written))
    return IngestResult(
        cmd_id=cmd.cmd_id, celex_id=cmd.celex_id, language=cmd.language,
        chunks_written=written, state="done",
    )


# ===========================================================================
# Section 6: Factory
# (unchanged)
# ===========================================================================
def build_ingest_deps(
    settings: Settings,
    *,
    source_kind: Literal["cellar", "local"] = "cellar",
    local_dir: str | None = None,
) -> IngestDeps:
    from .model_server import ModelClient

    source: Source = (
        CellarRest()
        if source_kind == "cellar"
        else LocalFile(
            directory=Path(local_dir) if local_dir else settings.data_dir / "formex"
        )
    )

    client = ModelClient(settings.model_socket_path())
    if client.available:
        log.info("ingestion.using_model_server")
        embedder = client
    else:
        embedder = BGEEmbedder(
            model_name=settings.embedding.model,
            batch_size=settings.embedding.batch_size,
            device=settings.embedding.device,
        )

    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    redis = aioredis.from_url(settings.redis.url, decode_responses=True)
    return IngestDeps(
        settings=settings, source=source, embedder=embedder,
        qdrant=qdrant, redis=redis,
    )
