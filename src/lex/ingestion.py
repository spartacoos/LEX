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

import redis.asyncio as aioredis
import structlog
from lxml import etree
from qdrant_client import AsyncQdrantClient, models as qmodels

from .commands import Chunk, IngestCmd, IngestResult
from .config import Settings
from .sources import CellarRest, LocalFile, Source, SourceUnavailable

from .tracing import observe

log = structlog.get_logger(__name__)


# ===========================================================================
# Section 1: Formex parsing
#
# Formex is the EU Publications Office's XML schema for legal acts.
# It preserves the hierarchical structure of a directive in a way that
# HTML and PDF don't. We walk the tree with XPath and emit a flat list
# of "structural nodes" — each node is one recital, article, or
# paragraph along with its ancestry (Part / Title / Chapter numbers).
#
# The chunker in Section 2 takes this list and produces final Chunks.
#
# ## A schema wrinkle: DIVISION vs typed elements
#
# Early Formex used typed structural tags — <PART>, <TITLE>, <CHAPTER>.
# Modern Formex (schema `formex-05.56-20160701.xd` and friends) replaces
# all of these with a generic <DIVISION> wrapper. The *kind* of division
# is encoded as a prefix word in the division's own <TITLE> — e.g. a
# Part's TITLE reads "PART I", a Chapter's reads "CHAPTER I".
#
# We only handle the modern shape here. If we start ingesting very old
# directives that use typed tags, we'll add a second code path — but
# the directive types we target (2004+) all use DIVISION.
# ===========================================================================

# Non-breaking space. Formex uses this between "Article" / "PART" / etc.
# and the number that follows. We normalise to a regular space before
# pattern-matching.
_NBSP = "\u00a0"

# Classifier for <DIVISION> titles. Captures kind (part/title/chapter/...)
# and the division number (Roman numeral or Arabic).
_DIVISION_TITLE_RE = re.compile(
    r"^(PART|TITLE|CHAPTER|SECTION|SUBSECTION)\s+([IVXLCDM0-9]+)\b",
    re.IGNORECASE,
)


@dataclass
class FormexNode:
    """One structural piece of a directive, with full ancestry."""
    kind: Literal["recital", "article", "paragraph", "annex"]
    text: str
    # Ancestry fields — any may be None depending on the directive's
    # structure. Not every directive has Parts; not every Article has
    # numbered Paragraphs.
    part: str | None = None
    title: str | None = None
    chapter: str | None = None
    article: str | None = None
    paragraph: str | None = None


def _text_content(elem: etree._Element) -> str:
    """
    Extract all text inside an element, normalising whitespace.

    Formex elements have mixed content (text + child elements + tails).
    `etree.tostring(method="text")` concatenates all of it. We then
    collapse whitespace runs — including the non-breaking space — so
    chunks are compact and tokenisers see normal spacing.
    """
    raw = etree.tostring(elem, method="text", encoding="unicode") or ""
    # Replace nbsp with a regular space *before* collapsing, so
    # "Article\u00a01" becomes "Article 1" not "Article1".
    raw = raw.replace(_NBSP, " ")
    return re.sub(r"\s+", " ", raw).strip()


def _division_kind_and_number(div: etree._Element) -> tuple[str | None, str | None]:
    """
    Classify a <DIVISION> as a Part/Title/Chapter/... and return its number.

    The division's own <TITLE> child begins with a prefix word telling
    us the structural level (see `_DIVISION_TITLE_RE`). Some DIVISIONs
    are just transparent groupings with no heading — for those we
    return (None, None) and ancestry lookup ignores them.
    """
    title = div.find("TITLE")
    if title is None:
        return None, None
    text = _text_content(title)
    m = _DIVISION_TITLE_RE.match(text)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2)


def _article_number(article: etree._Element) -> str | None:
    """
    Extract the article number.

    Two sources, in preference order:
      1. The IDENTIFIER attribute, e.g. IDENTIFIER="001" → "1".
         Machine-set, reliable, handles leading zeros.
      2. Fallback: parse <TI.ART> text, e.g. "Article 3" → "3".
    """
    ident = article.get("IDENTIFIER")
    if ident:
        stripped = ident.lstrip("0")
        # "000" would strip to "" — guard that.
        return stripped or "0"

    ti = article.find("TI.ART")
    if ti is not None:
        text = _text_content(ti)
        m = re.search(r"(\d+[a-z]?)", text)
        if m:
            return m.group(1)
    return None


def _paragraph_number(parag: etree._Element) -> str | None:
    """
    Extract the paragraph number from a <PARAG>.

    Two sources:
      1. IDENTIFIER attribute, e.g. "001.002" → "2". Formex encodes
         article.paragraph; we strip the article portion.
      2. <NO.PARAG> child text, e.g. "2." → "2".
    """
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
    """
    Extract the recital number from a <CONSID>.

    Structure (verified against 32018L1972):
        <CONSID>
          <NP>
            <NO.P>(15)</NO.P>
            <TXT>... recital text ...</TXT>
          </NP>
        </CONSID>

    We use a descendant search (`.//NO.P`) rather than a direct-child
    `find("NO.P")` so we don't depend on the exact wrapping. The text
    inside `<NO.P>` is the parenthesised number, e.g. "(15)".
    """
    no = consid.find(".//NO.P")
    if no is not None:
        text = _text_content(no)
        m = re.search(r"(\d+)", text)
        if m:
            return m.group(1)
    return None


def _ancestry(article: etree._Element) -> dict[str, str | None]:
    """
    Walk from an <ARTICLE> up to <ENACTING.TERMS>, gathering the
    Part/Title/Chapter numbers from <DIVISION> wrappers on the way.

    Each DIVISION's heading tells us its structural kind (see
    `_division_kind_and_number`). We record the *nearest* of each kind,
    since nested DIVISIONs of the same kind shouldn't exist in practice
    but if they do, closer is more specific.
    """
    out: dict[str, str | None] = {"part": None, "title": None, "chapter": None}
    for anc in article.iterancestors():
        if anc.tag != "DIVISION":
            continue
        kind, number = _division_kind_and_number(anc)
        if kind in out and out[kind] is None:
            out[kind] = number
    return out


def parse_formex(xml_bytes: bytes) -> list[FormexNode]:
    """
    Parse Formex XML into a flat list of FormexNode.

    Verified against Formex schema `formex-05.56-20160701.xd` using
    Directive 2018/1972 as a reference (~330 recitals, ~127 articles,
    ~440 numbered paragraphs).
    """
    # `recover=True` lets lxml limp past trivial DTD/entity issues that
    # EUR-Lex occasionally emits. Legal XML ≠ well-formed XML, alas.
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    root = etree.fromstring(xml_bytes, parser=parser)

    nodes: list[FormexNode] = []

    # ---- Recitals -------------------------------------------------------
    # <PREAMBLE> contains <GR.CONSID> ("group of considerations"), which
    # wraps individual <CONSID> elements — one per recital.
    for consid in root.iterfind(".//PREAMBLE//CONSID"):
        text = _text_content(consid)
        if not text:
            continue
        nodes.append(FormexNode(
            kind="recital",
            text=text,
            paragraph=_recital_number(consid),
        ))

    # ---- Articles + their Paragraphs -----------------------------------
    for article in root.iterfind(".//ENACTING.TERMS//ARTICLE"):
        anc = _ancestry(article)
        art_num = _article_number(article)

        parags = article.findall("PARAG")
        if not parags:
            # Short article with no numbered paragraphs — emit the whole
            # article body (including its TI.ART title) as one chunk.
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
                kind="paragraph",
                text=text,
                part=anc["part"], title=anc["title"], chapter=anc["chapter"],
                article=art_num,
                paragraph=_paragraph_number(parag),
            ))

    # ---- Annexes --------------------------------------------------------
    # Annexes in long directives (like the EECC) usually live in
    # separate CELLAR DOCs from the enacting act. Our v1 fetch only
    # grabs the main DOC, so this loop normally finds nothing — but
    # we keep it for small directives that inline their annexes, and
    # for the future when we concatenate annex DOCs into the feed.
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
#
# Per SPEC §6:
#   1. Each Article becomes a chunk.
#   2. If > max_chars, split at Paragraph boundaries.
#   3. If still > max_chars, split at sentence boundaries with overlap.
#   4. Recitals and Annexes are chunks too.
#
# The parser already did (1) and (2) — it emits one FormexNode per
# paragraph when an article has numbered ones. So this section's main
# job is step (3): sentence-split nodes that are still too long, with
# a configurable overlap so references straddling the boundary survive.
# ===========================================================================

# ---------------------------------------------------------------------------
# Sentence splitting for legal prose.
#
# Python's stdlib `re` requires fixed-width look-behinds, which rules
# out the tidy "split unless preceded by an abbreviation" one-liner.
# Two-pass approach instead:
#
#   1. Split aggressively on any .!? followed by whitespace + capital
#      letter or opening paren.
#   2. Walk the results and *re-join* any split where the previous
#      piece ends in a known abbreviation — i.e. the first pass was
#      wrong.
#
# Same end result, no `regex` dep, trivial to extend: add a token to
# `_ABBREVIATIONS` below.
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "art", "arts", "no", "nos", "para", "paras", "p", "pp",
    "i.e", "e.g", "cf", "vs", "etc",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _ends_with_abbrev(fragment: str) -> bool:
    """True if the fragment's last token looks like a known abbreviation."""
    m = re.search(r"([A-Za-z.]+)\.?$", fragment)
    if not m:
        return False
    tail = m.group(1).rstrip(".").lower()
    return tail in _ABBREVIATIONS


def _sentence_split(text: str) -> list[str]:
    """Two-pass split: aggressive regex, then merge at abbreviation boundaries."""
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


def _build_chunk_id(celex_id: str, kind: str, node: FormexNode, part_idx: int | None) -> str:
    """
    Stable, human-readable chunk ID.

    Examples:
      32018L1972:recital:15
      32018L1972:art:64:1          (article 64, paragraph 1, whole)
      32018L1972:art:64:1:p2       (article 64, paragraph 1, part 2 after split)
      32018L1972:annex:II
    """
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
    """Convert FormexNodes to Chunks, splitting long ones with overlap."""
    chunks: list[Chunk] = []

    for node in nodes:
        # Small enough? Emit as-is.
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

        # Too long — sentence-split with overlap.
        sents = _sentence_split(node.text)
        part_idx = 1
        i = 0
        while i < len(sents):
            # Greedily pack sentences until we'd exceed max_chars.
            buf: list[str] = []
            buf_len = 0
            j = i
            while j < len(sents) and buf_len + len(sents[j]) + 1 <= max_chars:
                buf.append(sents[j])
                buf_len += len(sents[j]) + 1
                j += 1
            # Guard: a single sentence longer than max_chars. Ship it
            # whole — truncating legal text silently would be worse
            # than an oversize chunk.
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
            # Advance, but keep `sentence_overlap` sentences of overlap
            # so a reference straddling the boundary survives.
            i = max(i + 1, j - sentence_overlap)

    log.info("chunking.done", chunks=len(chunks))
    return chunks


# ===========================================================================
# Section 3: Embedding
#
# BGE-M3 is special: one forward pass produces BOTH a dense vector
# (1024-dim, captures meaning) AND a lexical sparse vector (token-id →
# weight dict, captures exact terms). We use both for hybrid search in
# Qdrant.
#
# Why roll our own instead of using FlagEmbedding? FlagEmbedding is the
# official BAAI package, but its import path drags in training +
# reranker code that's sensitive to transformers library versions. We
# only need the encoder's two heads — ~50 lines of torch gives us that
# cleanly and keeps the dep tree stable.
# ===========================================================================

class BGEEmbedder:
    """
    BGE-M3 embedder — dense + sparse in a single forward pass.

    ## What BGE-M3 actually is

    A single XLM-RoBERTa-based encoder fine-tuned with three heads:

      1. A *dense* head — the CLS token's last-hidden-state projected
         to 1024 dims. Standard embedding output: one vector per input,
         cosine-compared at query time.

      2. A *sparse* (lexical) head — a per-token scalar weight. For
         each non-special token in the input, the model predicts "how
         much does this token matter for retrieval?" That gives us a
         BM25-like {token_id: weight} representation — ideal for
         exact-term matching (legal text has many "Article 3(2)(a)"
         references that dense vectors smooth over).

      3. A ColBERT-style multi-vector head (we don't use this in v1).

    One forward pass produces all of them because they share the encoder.
    """

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        # Imported lazily so `import lex` stays fast — torch is heavy.
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info("embedder.loading", model=model_name)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # `trust_remote_code=True` so HF loads BGE-M3's custom model
        # class (the one with the sparse head). Without it we'd only
        # get the plain encoder output.
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Pick the best available device. MPS = Apple Silicon Metal,
        # much faster than CPU for encoder workloads.
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._model = self._model.to(self._device).eval()

        self._batch_size = batch_size
        # BGE-M3 supports up to 8192 tokens, but our chunks are ≤1500
        # chars so we cap at 1024 to save memory and keep batches large.
        self._max_length = 1024

        log.info("embedder.loaded", model=model_name, device=self._device)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        """
        Encode a batch.

        Returns (dense, sparse) where:
          dense[i]  is a 1024-dim list[float]  (unit-normalised)
          sparse[i] is a {token_id: weight} dict
        """
        torch = self._torch
        dense_out: list[list[float]] = []
        sparse_out: list[dict[int, float]] = []

        # Special-token IDs to skip when building sparse weights
        # (CLS, SEP, PAD, UNK).
        special_ids = set(self._tokenizer.all_special_ids)

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

            # --- Dense --------------------------------------------------
            # BGE-M3 exposes `dense_output` (the 1024-d projection of
            # CLS). Fall back to raw CLS if this is a differently-wrapped
            # model. L2-normalise so Qdrant cosine similarity works.
            dense = (
                outputs.dense_output
                if hasattr(outputs, "dense_output")
                else outputs.last_hidden_state[:, 0]
            )
            dense = torch.nn.functional.normalize(dense, p=2, dim=-1)
            dense_out.extend(vec.cpu().tolist() for vec in dense)

            # --- Sparse -------------------------------------------------
            # One weight per input token. We zip with input IDs + mask
            # to build {token_id: max_weight} per row, skipping special
            # tokens and padding. Max-aggregation (not sum) matches
            # BGE's own convention for handling repeated tokens.
            sparse_weights = (
                outputs.sparse_output
                if hasattr(outputs, "sparse_output")
                else outputs.last_hidden_state.new_zeros(
                    outputs.last_hidden_state.shape[:2]
                )
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            for row_ids, row_mask, row_weights in zip(
                input_ids, attention_mask, sparse_weights
            ):
                row: dict[int, float] = {}
                for tok_id, mask_flag, weight in zip(
                    row_ids.tolist(), row_mask.tolist(), row_weights.tolist()
                ):
                    if not mask_flag:              # padding
                        continue
                    if tok_id in special_ids:      # CLS/SEP/etc
                        continue
                    w = float(weight)
                    if w <= 0:                     # below threshold
                        continue
                    prev = row.get(tok_id)
                    if prev is None or w > prev:
                        row[tok_id] = w
                sparse_out.append(row)

        return dense_out, sparse_out


# ===========================================================================
# Section 4: Qdrant writer
#
# Qdrant's hybrid search uses "named vectors": you declare multiple
# vectors per point at collection-creation time, then query whichever
# combination you want. We declare:
#
#   "dense"  : DENSE_DIM float vector, cosine similarity
#   "sparse" : sparse vector, dot product
#
# M2 (retrieval) will build a query using both and fuse the results.
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

    # Payload indexes on the fields we actually filter by (see
    # commands.RetrieveFilter). Without these, filtering does a linear
    # scan — O(N) in corpus size.
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
    """
    Qdrant point IDs must be UUIDs or unsigned 64-bit ints. We keep the
    human-readable ID in the payload and derive a deterministic UUIDv5
    for the point ID. Same chunk_id → same UUID → re-ingest is an
    idempotent upsert.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


async def write_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list[dict[int, float]],
) -> int:
    """Upsert all points for one directive in one batch. Returns count written."""
    points: list[qmodels.PointStruct] = []
    for chunk, d, s in zip(chunks, dense, sparse):
        points.append(qmodels.PointStruct(
            id=_chunk_uuid(chunk.id),
            vector={
                "dense": d,
                # Qdrant sparse vectors use parallel indices/values lists.
                "sparse": qmodels.SparseVector(
                    indices=list(s.keys()),
                    values=list(s.values()),
                ),
            },
            # Payload = everything a RetrieveResult might need. We dump
            # the whole Chunk so M2 can reconstruct it without a second DB.
            payload=chunk.model_dump(),
        ))

    # `wait=True` blocks until Qdrant has persisted. For ingestion we
    # want the completion signal; high-throughput use would flip this.
    await client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


# ===========================================================================
# Section 5: The ingestion handler
#
# This is the function the Engine calls. It orchestrates the pipeline
# and publishes state transitions. Everything above is plain data
# transformation; everything here is I/O and state tracking.
# ===========================================================================

@dataclass
class IngestDeps:
    """
    Everything the handler needs, bundled for easy test injection.

    We pass this in rather than constructing inside the handler so tests
    can hand in fakes (e.g. an embedder that returns zeros).
    """
    settings: Settings
    source: Source
    embedder: BGEEmbedder
    qdrant: AsyncQdrantClient
    redis: aioredis.Redis
    # Enumeration of the state machine's states — one source of truth.
    states: list[str] = field(default_factory=lambda: [
        "queued", "fetching", "parsing", "chunking", "embedding",
        "writing", "done", "failed",
    ])


async def _set_state(
    deps: IngestDeps, cmd_id: str, state: str, **extra: str
) -> None:
    """Write the new state to Redis + publish the transition."""
    key = f"{deps.settings.redis.key_prefix}:job:{cmd_id}"
    channel = f"{deps.settings.redis.key_prefix}:job:{cmd_id}:events"
    now = datetime.now(timezone.utc).isoformat()
    payload = {"state": state, "updated_at": now, **extra}
    # `hset(mapping=...)` takes str→str; stringify everything.
    await deps.redis.hset(key, mapping={k: str(v) for k, v in payload.items()})
    await deps.redis.publish(channel, state)
    log.info("ingest.state", cmd_id=cmd_id, state=state, **extra)


def _eli_uri(celex_id: str) -> str:
    """
    Derive the ELI URI from a CELEX ID for a directive.

      CELEX 32018L1972 → http://data.europa.eu/eli/dir/2018/1972/oj

    The pattern is `3<YYYY>L<NNNN>` → `dir/<YYYY>/<NNNN>`. For
    regulations (R), decisions (D), etc. we'd add more patterns. For
    anything else we fall back to an EUR-Lex CELEX URL — still citable,
    just not ELI.
    """
    m = re.match(r"^3(\d{4})L(\d{4})$", celex_id)
    if not m:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex_id}"
    year, num = m.groups()
    return f"http://data.europa.eu/eli/dir/{year}/{int(num)}/oj"

@observe(name="ingest")
async def handle_ingest(cmd: IngestCmd, deps: IngestDeps) -> IngestResult:
    """
    Execute the full ingestion state machine for one directive.

    Structure: each `try` block is one state. On exception we record
    `failed` with the error message and return; we do NOT re-raise,
    because in M4 this will run inside a Redis queue worker and we want
    the worker to keep consuming.
    """
    cmd_id = str(cmd.cmd_id)
    collection = deps.settings.collection_name(cmd.language)

    await _set_state(deps, cmd_id, "queued",
                     celex_id=cmd.celex_id, language=cmd.language)

    # ---- Fetch ---------------------------------------------------------
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

    # ---- Parse ---------------------------------------------------------
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

    # ---- Chunk ---------------------------------------------------------
    await _set_state(deps, cmd_id, "chunking")
    chunks = chunk_nodes(
        nodes,
        celex_id=cmd.celex_id, language=cmd.language,
        eli_uri=_eli_uri(cmd.celex_id),
        max_chars=deps.settings.chunking.max_chars,
        sentence_overlap=deps.settings.chunking.sentence_overlap,
    )

    # ---- Embed ---------------------------------------------------------
    # Embedding is CPU/GPU bound; run off the event loop so we don't
    # block other concurrent work. `asyncio.to_thread` picks up a
    # worker thread from the default executor.
    await _set_state(deps, cmd_id, "embedding", chunks=len(chunks))
    texts = [c.text for c in chunks]
    dense, sparse = await asyncio.to_thread(deps.embedder.embed, texts)

    # ---- Write ---------------------------------------------------------
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

    await _set_state(deps, cmd_id, "done", chunks=written)
    return IngestResult(
        cmd_id=cmd.cmd_id, celex_id=cmd.celex_id, language=cmd.language,
        chunks_written=written, state="done",
    )


# ===========================================================================
# Section 6: Factory — build a handler with real deps
#
# The CLI and API call this. Tests skip it and build `IngestDeps`
# directly with fakes.
# ===========================================================================

def build_ingest_deps(
    settings: Settings,
    *,
    source_kind: Literal["cellar", "local"] = "cellar",
    local_dir: str | None = None,
) -> IngestDeps:
    """Construct the real dependencies. First call loads the embedding model."""
    source: Source = (
        CellarRest()
        if source_kind == "cellar"
        else LocalFile(
            directory=(local_dir or settings.data_dir / "formex")  # type: ignore[arg-type]
        )
    )
    embedder = BGEEmbedder(
        model_name=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
    )
    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    redis = aioredis.from_url(settings.redis.url, decode_responses=True)
    return IngestDeps(
        settings=settings, source=source, embedder=embedder,
        qdrant=qdrant, redis=redis,
    )
