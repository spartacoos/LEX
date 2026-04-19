"""
The dispatcher.

Takes a Command, looks at its `kind`, routes to the right handler,
returns a Result. This file stays small by design — it owns routing
and dependency wiring, nothing else.

## Mental model

    caller  ──►  Engine.submit(cmd)  ──►  handler(cmd, deps)  ──►  Result

The Engine owns the routing table and the shared dependency bundle.
Handlers own their own state (models, clients). That mirrors how a GPU
command processor works: the frontend doesn't know what a shader
*does*, only which queue to stick it on.

## Inline vs queued ingestion

Ingestion is slow. The API layer wants `POST /ingest` to return in
milliseconds with a `cmd_id` so the UI can poll, while the CLI wants
`lex ingest …` to run to completion with a progress spinner.

So: `Engine` has an `ingest_inline` flag.

  - `ingest_inline=True`   → CLI. Engine calls `handle_ingest` directly.
  - `ingest_inline=False`  → API. Engine pushes onto the Redis queue
                              and returns an `IngestResult(state="queued")`
                              immediately.

Retrieve and answer are always inline — they're fast enough to sit on
an HTTP request (search ~1s, answer starts streaming in ~1s).
"""

from __future__ import annotations

from typing import Protocol

import structlog

from .commands import (
    AnswerCmd,
    AnswerResult,
    Command,
    IngestCmd,
    IngestResult,
    RetrieveCmd,
    RetrieveResult,
    Result,
)
from .config import Settings

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Handler protocols.
#
# Structural typing instead of ABCs — a handler can be a class, a
# closure, a partial'd function. Testing is trivial: a fake handler is
# a two-line lambda.
# ---------------------------------------------------------------------------

class IngestHandler(Protocol):
    async def __call__(self, cmd: IngestCmd) -> IngestResult: ...


class RetrieveHandler(Protocol):
    async def __call__(self, cmd: RetrieveCmd) -> RetrieveResult: ...


class AnswerHandler(Protocol):
    async def __call__(self, cmd: AnswerCmd, *, on_token=None) -> AnswerResult: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """
    The single entry point for all operations.

    Construction wires handlers in. `wire_default()` builds real ones;
    tests hand in fakes via the constructor.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        ingest: IngestHandler,
        retrieve: RetrieveHandler,
        answer: AnswerHandler,
    ) -> None:
        self.settings = settings
        self._ingest = ingest
        self._retrieve = retrieve
        self._answer = answer

    async def submit(self, cmd: Command, *, on_token=None) -> Result:
        """
        Dispatch a command. Structural pattern matching gives us
        exhaustive, type-narrowed routing.

        `on_token` is an extension point for AnswerCmd streaming —
        callers that want live token delivery pass a callback; the
        engine passes it through to the answer handler. Retrieve and
        ingest ignore it.
        """
        logger = log.bind(cmd_id=str(cmd.cmd_id), kind=cmd.kind)
        logger.info("engine.submit")

        try:
            match cmd:
                case IngestCmd():
                    return await self._ingest(cmd)
                case RetrieveCmd():
                    return await self._retrieve(cmd)
                case AnswerCmd():
                    return await self._answer(cmd, on_token=on_token)
                case _:
                    # `Command` is a closed union — unreachable in a
                    # well-typed program. Raise loudly so "someone
                    # added a new variant" isn't a silent bug.
                    raise TypeError(f"Unknown command kind: {cmd!r}")
        except Exception:
            logger.exception("engine.submit.failed")
            raise


# ---------------------------------------------------------------------------
# Factory — builds an Engine with real handlers.
#
# The choice of inline-vs-queued ingestion lives here because it
# affects how we construct the ingest handler, not how the Engine
# dispatches.
# ---------------------------------------------------------------------------

async def wire_default(settings: Settings, *, ingest_inline: bool) -> Engine:
    """
    Build a fully-wired Engine.

    Parameters
    ----------
    settings
        From `get_settings()`.
    ingest_inline
        True for CLI (run `handle_ingest` directly, blocking).
        False for API (enqueue into Redis, return immediately with
        state="queued").
    """
    from .generation import build_answer_deps, handle_answer
    from .ingestion import build_ingest_deps, handle_ingest
    from .retrieval import build_retrieve_deps, handle_retrieve

    # Retrieve + answer deps are always built the same way. Note that
    # `build_answer_deps` is async because it pings the LLM endpoint.
    retrieve_deps = build_retrieve_deps(settings)
    answer_deps = await build_answer_deps(settings)

    async def retrieve_handler(cmd):
        return await handle_retrieve(cmd, retrieve_deps)

    async def answer_handler(cmd, *, on_token=None):
        return await handle_answer(cmd, answer_deps, on_token=on_token)

    # Ingest handler branches on the inline flag.
    if ingest_inline:
        ingest_deps = build_ingest_deps(settings, source_kind="cellar")

        async def ingest_handler(cmd):
            return await handle_ingest(cmd, ingest_deps)
    else:
        # Queued mode: just enqueue and return. The worker process
        # picks the job up. We don't build embedder / qdrant clients
        # in-process because they'd be unused (the worker owns them).
        import redis.asyncio as aioredis
        from .worker import enqueue_ingest

        redis = aioredis.from_url(settings.redis.url, decode_responses=True)

        async def ingest_handler(cmd):
            await enqueue_ingest(redis, settings, cmd)
            return IngestResult(
                cmd_id=cmd.cmd_id,
                celex_id=cmd.celex_id,
                language=cmd.language,
                chunks_written=0,
                state="done",       # "done" here means "handed off cleanly"
                                    # — the actual done state lives in Redis
                                    # and is what `GET /jobs/{id}` exposes.
            )

    return Engine(
        settings=settings,
        ingest=ingest_handler,
        retrieve=retrieve_handler,
        answer=answer_handler,
    )
