"""
Command and Result types — the entire system contract in one file.

## The command-buffer pattern

Borrowed from GPU programming. A GPU driver doesn't expose "draw this
triangle right now"; it exposes "here's a *command buffer* — a list of
typed operations — please execute it." The caller (the game) is decoupled
from the executor (the driver) by a data structure.

We do the same thing. CLI, UI, API, and tests all build a `Command`
(a pydantic object) and hand it to the `Engine`. The Engine dispatches
to a handler based on the command's `kind`. Subsystems never call each
other directly — they only talk through these types.

## Why this works so well for RAG

- **Testability**: a test is just `await engine.submit(AnswerCmd(...))`.
  No HTTP mocks, no CLI subprocess.
- **Observability**: every command carries a `cmd_id` (UUID). That ID
  is the correlation key for logs, traces, and Redis job state.
- **Evolvability**: adding a new operation = adding a variant to the
  `Command` union + a handler. Existing callers don't need to change.

## Discriminated unions, briefly

Each command has a `kind: Literal["..."]` field with a unique value.
pydantic uses that field to decide which class to build when parsing
raw JSON, and type checkers use it to narrow the union in `match`
statements. This is how you do sum types / tagged unions in Python.
"""

from __future__ import annotations

from typing import Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

class RetrieveFilter(BaseModel):
    """
    Metadata filters applied at the vector DB level.

    Every filter here maps to a Qdrant payload filter. We keep this small
    on purpose: more fields = more index branches = slower queries. Add
    new ones when a user story actually demands them.
    """
    celex_id: str | None = None
    article: str | None = None
    language: str = "en"


class Chunk(BaseModel):
    """
    One piece of a directive, as stored and as retrieved.

    `text` is what the embedder saw and what the LLM will see.
    Everything else is metadata for citations and filtering.

    The field set mirrors SPEC §6's ancestry requirement — if you look
    at a chunk you can always reconstruct "where in the directive am I?"
    """
    id: str                       # "32018L1972:art:2:1"  (stable, reproducible)
    celex_id: str                 # "32018L1972"
    eli_uri: str                  # "http://data.europa.eu/eli/dir/2018/1972/oj"
    language: str                 # "en"
    chunk_type: Literal["recital", "article", "paragraph", "annex"]
    # Optional hierarchy — not every directive has every level.
    part: str | None = None
    title: str | None = None
    chapter: str | None = None
    article: str | None = None    # "2", "64", ...
    paragraph: str | None = None  # "1", "2a", ...
    text: str


# ---------------------------------------------------------------------------
# Commands — the input side of each operation.
# ---------------------------------------------------------------------------

class IngestCmd(BaseModel):
    """Fetch a directive, chunk it, embed it, store it."""
    kind: Literal["ingest"] = "ingest"
    cmd_id: UUID = Field(default_factory=uuid4)
    celex_id: str                 # "32018L1972"
    # Where to fetch from. "cellar" = EUR-Lex REST API. "local" = a file
    # you already downloaded (useful for offline dev and tests).
    source: Literal["cellar", "local"] = "cellar"
    language: str = "en"


class RetrieveCmd(BaseModel):
    """Hybrid search + rerank. Returns chunks, does NOT call the LLM."""
    kind: Literal["retrieve"] = "retrieve"
    cmd_id: UUID = Field(default_factory=uuid4)
    query: str
    top_k: int = 5
    filters: RetrieveFilter = Field(default_factory=RetrieveFilter)


class AnswerCmd(BaseModel):
    """Full RAG: retrieve + generate + extract citations."""
    kind: Literal["answer"] = "answer"
    cmd_id: UUID = Field(default_factory=uuid4)
    query: str
    filters: RetrieveFilter = Field(default_factory=RetrieveFilter)
    stream: bool = False


# The union itself. Any subsystem that takes "a command" takes this.
# The `kind` field is the discriminator pydantic uses to decide which
# variant to parse when given raw dict/JSON input.
Command = Union[IngestCmd, RetrieveCmd, AnswerCmd]


# ---------------------------------------------------------------------------
# Results — what handlers return.
# ---------------------------------------------------------------------------

class IngestResult(BaseModel):
    cmd_id: UUID
    celex_id: str
    language: str
    chunks_written: int
    # Terminal state of the ingestion state machine. "done" on success.
    state: Literal["done", "failed"]
    error: str | None = None


class RetrievedChunk(BaseModel):
    """A chunk plus the scores that put it here."""
    chunk: Chunk
    dense_score: float
    sparse_score: float
    rerank_score: float


class RetrieveResult(BaseModel):
    cmd_id: UUID
    query: str
    chunks: list[RetrievedChunk]


class Citation(BaseModel):
    """One citation marker in a generated answer.

    We render these as clickable cards in the UI. `span_start`/`span_end`
    are character offsets into the answer text so the UI can highlight
    exactly which phrase a citation supports.
    """
    chunk_id: str
    article: str | None
    paragraph: str | None
    span_start: int
    span_end: int


class AnswerResult(BaseModel):
    cmd_id: UUID
    query: str
    answer: str
    citations: list[Citation]
    # Keep the retrieved chunks around so the UI can show "sources used."
    chunks: list[RetrievedChunk]


Result = Union[IngestResult, RetrieveResult, AnswerResult]
