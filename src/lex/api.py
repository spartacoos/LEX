"""
FastAPI surface.

Three routes map one-to-one onto our command types:

    POST /ingest   → IngestCmd  → enqueue; return cmd_id
    POST /search   → RetrieveCmd → inline; return chunks
    POST /ask      → AnswerCmd  → inline; stream tokens via SSE

Plus one read-only route for ingestion progress:

    GET /jobs/{cmd_id}  → Redis hash for that job

## Why FastAPI

Three properties we want:
  1. Native async — our handlers are async, so no thread-pool glue.
  2. Automatic OpenAPI schema from Pydantic models — because our
     command types ARE Pydantic models, routing and schema come free.
  3. Streaming responses — SSE is two lines of code via
     `StreamingResponse` + an async generator.

## Streaming, briefly

Server-Sent Events (SSE) is the HTTP way to stream text from server to
browser. Each line in the response body starts with `data: ` and ends
with a blank line:

    data: {"token": "Hello"}

    data: {"token": " world"}

    data: {"done": true, "citations": [...]}

Browsers and most HTTP clients have built-in SSE support. We use a
tiny JSON-per-event format so the Chainlit UI in M5 can consume it
directly.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import UUID

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inspect import isawaitable
async def _maybe_await(value):
    if isawaitable(value):
        return await value
    return value

from .commands import AnswerCmd, AnswerResult, IngestCmd, RetrieveCmd
from .config import get_settings
from .engine import Engine, wire_default

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — models + engine built once at startup.
#
# FastAPI calls the lifespan context manager exactly once: entry at
# app start, exit at shutdown. Stuff you set on `app.state` is shared
# across all requests. Perfect for our heavy, load-once ML models.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load models + wire the engine. Runs before the first request."""
    settings = get_settings()
    log.info("api.startup", qdrant=settings.qdrant.url, redis=settings.redis.url)

    # `ingest_inline=False`: the API enqueues onto Redis; a separate
    # worker process consumes. Run `lex worker` in another terminal.
    app.state.settings = settings
    app.state.engine = await wire_default(settings, ingest_inline=False)
    app.state.redis = aioredis.from_url(settings.redis.url, decode_responses=True)

    try:
        yield
    finally:
        # Best-effort cleanup. The FastAPI worker process is usually
        # going away anyway, but closing these explicitly avoids noisy
        # shutdown warnings.
        await app.state.redis.aclose()
        log.info("api.shutdown")


app = FastAPI(
    title="LEX API",
    description="Natural-language Q&A over EU directives.",
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Request/response models.
#
# We take our Pydantic command types directly — the user doesn't need
# to supply cmd_id (we generate one), and /ask streams rather than
# returning a plain JSON body, so we don't declare a response_model
# for that route.
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    celex_id: str
    language: str = "en"


class IngestResponse(BaseModel):
    cmd_id: UUID
    state: str


class JobStateResponse(BaseModel):
    cmd_id: UUID
    state: str
    celex_id: str | None = None
    language: str | None = None
    chunks: int | None = None
    error: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Returns 200 if the process is alive."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def post_ingest(req: IngestRequest) -> IngestResponse:
    """
    Enqueue an ingestion job. Returns immediately.

    Poll `GET /jobs/{cmd_id}` to watch progress.
    """
    cmd = IngestCmd(celex_id=req.celex_id, language=req.language)
    engine: Engine = app.state.engine
    result = await engine.submit(cmd)
    # result.state is "done" from the engine's perspective (handed off
    # cleanly); the actual ingestion state lives in the Redis job hash.
    # We return "queued" to reflect reality from the client's POV.
    return IngestResponse(cmd_id=cmd.cmd_id, state="queued")


@app.get("/jobs/{cmd_id}", response_model=JobStateResponse)
async def get_job(cmd_id: UUID) -> JobStateResponse:
    """Current state of an ingestion job. 404 if not found."""
    settings = app.state.settings
    redis: aioredis.Redis = app.state.redis
    key = f"{settings.redis.key_prefix}:job:{cmd_id}"

    data = await _maybe_await(redis.hgetall(key))
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Redis values are strings. We coerce `chunks` to int when present
    # but leave everything else as str for forward-compat.
    chunks = data.get("chunks")
    return JobStateResponse(
        cmd_id=cmd_id,
        state=data.get("state", "unknown"),
        celex_id=data.get("celex_id"),
        language=data.get("language"),
        chunks=int(chunks) if chunks else None,
        error=data.get("error"),
        updated_at=data.get("updated_at"),
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    language: str = "en"
    celex_id: str | None = None
    article: str | None = None


@app.post("/search")
async def post_search(req: SearchRequest) -> dict:
    """Hybrid search + rerank. Returns the top-K chunks with scores."""
    from .commands import RetrieveFilter

    cmd = RetrieveCmd(
        query=req.query,
        top_k=req.top_k,
        filters=RetrieveFilter(
            language=req.language,
            celex_id=req.celex_id,
            article=req.article,
        ),
    )
    engine: Engine = app.state.engine
    result = await engine.submit(cmd)
    # Let FastAPI serialize the pydantic result via model_dump.
    return result.model_dump(mode="json")


class AskRequest(BaseModel):
    query: str
    language: str = "en"
    celex_id: str | None = None
    article: str | None = None


@app.post("/ask")
async def post_ask(req: AskRequest) -> StreamingResponse:
    """
    Full RAG with Server-Sent Events streaming.

    Response body is a sequence of SSE events:
        data: {"token": "..."}     ← one per streamed token
        data: {"done": true, "citations": [...], "chunks": [...]}  ← final

    The client reads until it sees the `done: true` event.
    """
    from .commands import RetrieveFilter

    cmd = AnswerCmd(
        query=req.query,
        filters=RetrieveFilter(
            language=req.language,
            celex_id=req.celex_id,
            article=req.article,
        ),
        stream=True,
    )

    engine: Engine = app.state.engine

    # We need to bridge two async worlds:
    #   (a) the answer handler delivers tokens via an `on_token`
    #       *synchronous* callback (it was designed for the CLI's
    #       print-as-you-go).
    #   (b) FastAPI wants an async generator for StreamingResponse.
    #
    # Bridge: an asyncio.Queue. The callback drops tokens into the
    # queue; the generator awaits and yields them. When the handler
    # returns, we push a sentinel + final metadata.
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def on_token(tok: str) -> None:
        # `put_nowait` is safe because the callback runs on the same
        # event loop as the handler (both are async). Unbounded queue
        # is fine — tokens arrive faster than they're consumed only if
        # the client stalled, in which case we want to buffer, not drop.
        queue.put_nowait(tok)

    async def runner() -> None:
        """Run the handler, then push sentinel + final payload."""
        try:
            result = await engine.submit(cmd, on_token=on_token)
            if not isinstance(result, AnswerResult):
                raise TypeError(f"Expected AnswerResult, got {type(result).__name__}")
        except Exception as e:
            log.exception("api.ask.failed", error=str(e))
            queue.put_nowait({"error": str(e)})
        else:
            # Final metadata event — the full Citation list and the
            # chunks we grounded on.
            queue.put_nowait({
                "done": True,
                "citations": [c.model_dump() for c in result.citations],
                "chunks":    [rc.model_dump() for rc in result.chunks],
                "answer":    result.answer,
            })
        finally:
            queue.put_nowait(SENTINEL)

    async def sse_generator():
        # Kick off the runner in the background. It writes to the queue
        # while we read from it.
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    return
                if isinstance(item, str):
                    # Plain token.
                    yield f"data: {json.dumps({'token': item})}\n\n"
                else:
                    # Dict (either final payload or error). Emit as-is.
                    yield f"data: {json.dumps(item)}\n\n"
        finally:
            # Make sure the runner is done before closing the response.
            await task

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        # Tell browsers and proxies not to buffer — critical for SSE.
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
