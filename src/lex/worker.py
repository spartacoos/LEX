"""
Redis-queue consumer for ingestion jobs.

## Why a worker at all?

Ingestion is slow — fetch from CELLAR (seconds), parse + chunk (seconds),
embed ~800 chunks on MPS (minute+), write to Qdrant (seconds). If the
HTTP handler ran that inline, the POST /ingest request would hold the
connection for a minute or more and the UI would look hung.

Instead we use the classic producer/consumer split:

    POST /ingest  ──►  Engine  ──►  LPUSH on Redis list
                                          │
                                          ▼
                                    BLPOP in worker  ──►  handle_ingest()
                                          │
                                          ▼
                                    state machine writes
                                    to Redis hash + pubsub

The API returns immediately with `{cmd_id, state: "queued"}`. The UI
polls `GET /jobs/{cmd_id}` (or subscribes to the pubsub channel) to
track progress.

## Why not RQ / Celery / Dramatiq?

Those libraries are great. They bring:
  - Result backends (we don't need — Redis hash is our result)
  - Cron-style scheduling (we don't need)
  - Retry policies (we'd rather handle in the state machine)
  - Multiple queues with priorities (overkill for one job type)

We'd be paying complexity for features we don't use. A raw BLPOP loop
is ~80 lines, handles exactly our case, and stays trivial to reason
about. If we ever need priority queues or cron scheduling, we graduate
to RQ — but not before.

## Running it

    uv run lex worker

That's it. Ctrl-C stops it cleanly (finishes any in-flight job first).
In production you'd run this under a process supervisor (systemd,
supervisord, `docker compose`) that restarts on crash. For now, one
terminal per worker.
"""

from __future__ import annotations

import asyncio
import json
import signal

import redis.asyncio as aioredis
import structlog

from .commands import IngestCmd
from .config import Settings, get_settings
from .ingestion import IngestDeps, build_ingest_deps, handle_ingest

log = structlog.get_logger(__name__)


# The Redis list name that producers LPUSH to and workers BLPOP from.
# Kept as a function of settings so `lex:queue:ingest` stays under the
# same `lex:` prefix as the job state hashes.
def _queue_key(settings: Settings) -> str:
    return f"{settings.redis.key_prefix}:queue:ingest"


# ---------------------------------------------------------------------------
# Enqueue (called by the engine from inside the HTTP layer)
# ---------------------------------------------------------------------------

async def enqueue_ingest(redis: aioredis.Redis, settings: Settings, cmd: IngestCmd) -> None:
    """
    Push an IngestCmd onto the ingestion queue.

    We also immediately write `state=queued` to the job hash, so a UI
    polling `GET /jobs/{cmd_id}` right after submission sees a real
    state rather than "not found until the worker picks it up."
    """
    job_key = f"{settings.redis.key_prefix}:job:{cmd.cmd_id}"
    queue_key = _queue_key(settings)

    # Write the "queued" state hash first so the UI has something to
    # read between enqueue and worker pickup.
    await redis.hset(job_key, mapping={
        "state": "queued",
        "celex_id": cmd.celex_id,
        "language": cmd.language,
    })

    # LPUSH the serialized command. Workers BLPOP it.
    payload = cmd.model_dump_json()
    await redis.lpush(queue_key, payload)
    log.info("worker.enqueue", cmd_id=str(cmd.cmd_id),
             celex_id=cmd.celex_id, language=cmd.language)


# ---------------------------------------------------------------------------
# Consume (the worker loop itself)
# ---------------------------------------------------------------------------

async def _worker_loop(settings: Settings, deps: IngestDeps, stop: asyncio.Event) -> None:
    """
    Main consume loop. Runs until `stop` is set.

    Uses BLPOP with a 1s timeout so we can check the stop flag
    regularly without spinning. A pure blocking BLPOP would leave the
    worker impossible to shut down gracefully.
    """
    queue_key = _queue_key(settings)
    log.info("worker.start", queue=queue_key)

    while not stop.is_set():
        # BLPOP returns (key, value) or None on timeout.
        msg = await deps.redis.blpop(queue_key, timeout=1.0)
        if msg is None:
            continue   # timeout, re-check stop flag

        _key, raw = msg

        # Parse the IngestCmd. If the payload is malformed, log and
        # move on — we never want one poisoned message to break the
        # worker.
        try:
            data = json.loads(raw)
            cmd = IngestCmd.model_validate(data)
        except Exception as e:
            log.exception("worker.parse_failed", error=str(e), raw=raw[:200])
            continue

        # Run the ingest. `handle_ingest` is defensive: it records
        # failures into the job hash rather than raising. Even so we
        # wrap in try/except as belt-and-braces — the worker must
        # survive any handler mishap.
        try:
            result = await handle_ingest(cmd, deps)
            log.info("worker.job_done",
                     cmd_id=str(cmd.cmd_id),
                     state=result.state,
                     chunks=result.chunks_written)
        except Exception as e:
            log.exception("worker.handler_crash",
                          cmd_id=str(cmd.cmd_id), error=str(e))

    log.info("worker.stop")


# ---------------------------------------------------------------------------
# Entry point used by the CLI
# ---------------------------------------------------------------------------

async def run_worker() -> None:
    """
    Start a worker. Loads the embedding model once, then consumes forever.

    Handles SIGINT/SIGTERM so `Ctrl-C` and `docker stop` both produce
    a clean shutdown: the in-flight job (if any) finishes before the
    loop exits.
    """
    settings = get_settings()
    deps = build_ingest_deps(settings, source_kind="cellar")

    stop = asyncio.Event()

    # Wire signals to the stop flag. add_signal_handler works on
    # Unix; on Windows we'd fall back to KeyboardInterrupt.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass   # Windows — user can still Ctrl-C, just less gracefully

    try:
        await _worker_loop(settings, deps, stop)
    finally:
        # Close async clients so their connection pools don't leak.
        await deps.qdrant.close()
        await deps.redis.aclose()
