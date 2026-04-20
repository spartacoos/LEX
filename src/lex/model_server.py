"""
Model server daemon — loads BGE-M3 and the reranker once, serves requests
over a Unix domain socket.

## Why a Unix socket

Unix domain sockets are:
  - Faster than TCP loopback (no network stack, kernel copies directly)
  - Invisible to the network (no port conflicts, no firewall issues)
  - Automatically cleaned up on process death (via atexit)
  - Supported on Linux, macOS, and WSL2

## Protocol

Request and response are newline-delimited JSON. One request per connection.
Client sends the full JSON payload then shuts down the write side (SHUT_WR)
to signal EOF. Server reads until EOF, processes, writes response, closes.

Request types:

  {"op": "embed",        "texts": ["...", "..."], "idf_path": "..."}
  {"op": "query_sparse", "text": "...",           "idf_path": "..."}
  {"op": "rerank",       "query": "...", "passages": ["...", "..."]}
  {"op": "ping"}

Response:

  {"ok": true,  "dense": [[...], ...], "sparse": [{...}, ...]}  # embed
  {"ok": true,  "weights": {"123": 0.5, ...}}                   # query_sparse
  {"ok": true,  "scores": [0.9, 0.1, ...]}                      # rerank
  {"ok": true,  "pong": true}                                    # ping
  {"ok": false, "error": "..."}                                  # any error

## Client design

ModelClient uses only stdlib socket — no asyncio on the client side.
This means it can be called safely from:
  - Synchronous code
  - asyncio.to_thread() (which is how the retrieval/ingestion handlers
    call embed and rerank — they already dispatch to a thread pool)
  - Any context without a running event loop

ping() is the one exception — it is async because it is only ever called
from async contexts (services command, health checks).

## Lifecycle

  lex serve-models          starts the daemon (blocking, Ctrl-C to stop)
  lex services              pings the daemon to check if it's running
  lex search / lex ask      connect as thin clients, no model loading
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket as _socket
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Maximum message size: 50 MB covers 921 chunks × 1024-dim float32 vectors
# (921 × 1024 × 4 bytes ≈ 3.8 MB) with headroom for JSON encoding overhead.
_MAX_MSG = 50 * 1024 * 1024


# ===========================================================================
# Server
# ===========================================================================

class ModelServer:
    """
    Async Unix-socket server owning the embedding and reranking models.

    Loads models once at startup. Handles concurrent requests via asyncio —
    CPU-bound work (embed, rerank) is dispatched to a thread pool so the
    event loop stays free for accepting new connections.
    """

    def __init__(self, socket_path: Path, embedder, reranker) -> None:
        self._socket_path = socket_path
        self._embedder = embedder
        self._reranker = reranker

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # Read until client sends EOF (SHUT_WR).
            raw = await reader.read(_MAX_MSG)
            req: dict[str, Any] = json.loads(raw)
            op = req.get("op")

            if op == "ping":
                resp: dict[str, Any] = {"ok": True, "pong": True}

            elif op == "embed":
                texts: list[str] = req["texts"]
                idf_path_str: str | None = req.get("idf_path")
                idf_path = Path(idf_path_str) if idf_path_str else None
                dense, sparse = await asyncio.to_thread(
                    self._embedder.embed, texts, idf_path
                )
                resp = {"ok": True, "dense": dense, "sparse": sparse}

            elif op == "query_sparse":
                text: str = req["text"]
                idf_path_str = req.get("idf_path")
                idf_path = Path(idf_path_str) if idf_path_str else None
                weights = await asyncio.to_thread(
                    self._embedder.query_sparse, text, idf_path
                )
                resp = {"ok": True, "weights": weights}

            elif op == "rerank":
                query: str = req["query"]
                passages: list[str] = req["passages"]
                scores = await asyncio.to_thread(
                    self._reranker.rerank, query, passages
                )
                resp = {"ok": True, "scores": scores}

            else:
                resp = {"ok": False, "error": f"unknown op: {op!r}"}

        except Exception as e:
            log.exception("model_server.handle_error")
            resp = {"ok": False, "error": str(e)}

        writer.write(json.dumps(resp).encode())
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def serve(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._socket_path.exists():
            self._socket_path.unlink()

        server = await asyncio.start_unix_server(
            self._handle, path=str(self._socket_path)
        )
        log.info("model_server.listening", socket=str(self._socket_path))

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass  # Windows / WSL edge cases

        async with server:
            await stop.wait()

        log.info("model_server.stopping")
        if self._socket_path.exists():
            self._socket_path.unlink()


# ===========================================================================
# Client
# ===========================================================================

class ModelClient:
    """
    Thin synchronous client for the model server.

    All public methods (embed, query_sparse, rerank) are synchronous and
    use stdlib socket directly — no asyncio, no event loop dependency.
    This makes them safe to call from asyncio.to_thread() (which is how
    the retrieval and ingestion handlers call them) or from plain
    synchronous code.

    ping() is async because it is only ever called from async contexts
    (the services health check).

    Falls back gracefully: if the socket doesn't exist, callers should
    detect this via the `available` property and use in-process models
    instead (see build_retrieve_deps / build_ingest_deps).
    """

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path

    @property
    def available(self) -> bool:
        """True if the model server socket file exists."""
        return self._socket_path.exists()

    def _call_sync(self, req: dict[str, Any]) -> dict[str, Any]:
        """
        Send a request and receive a response over a Unix socket.

        Pure stdlib — no asyncio, no event loop. Safe from any thread.

        Protocol:
          1. Connect to socket
          2. Send full JSON payload
          3. Shutdown write side (SHUT_WR) — tells server we're done sending
          4. Read response until server closes connection
          5. Parse and return JSON response
        """
        raw = json.dumps(req).encode()
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as sock:
            sock.connect(str(self._socket_path))
            sock.sendall(raw)
            # SHUT_WR signals EOF to the server's reader.read() call.
            sock.shutdown(_socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        resp: dict[str, Any] = json.loads(b"".join(chunks))
        if not resp.get("ok"):
            raise RuntimeError(f"model_server error: {resp.get('error')}")
        return resp

    # ------------------------------------------------------------------
    # Public interface — mirrors BGEEmbedder / BGEReranker exactly so
    # RetrieveDeps and IngestDeps can hold either without knowing which.
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        idf_save_path: Path | None = None,
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """Encode texts to dense + sparse vectors via the model server."""
        req: dict[str, Any] = {"op": "embed", "texts": texts}
        if idf_save_path:
            req["idf_path"] = str(idf_save_path)
        resp = self._call_sync(req)
        # JSON round-trip turns int keys to strings — convert back.
        sparse = [
            {int(k): v for k, v in row.items()}
            for row in resp["sparse"]
        ]
        return resp["dense"], sparse

    def query_sparse(
        self,
        text: str,
        idf_load_path: Path | None = None,
    ) -> dict[int, float]:
        """Encode a single query to a sparse BM25 vector."""
        req: dict[str, Any] = {"op": "query_sparse", "text": text}
        if idf_load_path:
            req["idf_path"] = str(idf_load_path)
        resp = self._call_sync(req)
        return {int(k): v for k, v in resp["weights"].items()}

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Score passages against query using the cross-encoder."""
        resp = self._call_sync(
            {"op": "rerank", "query": query, "passages": passages}
        )
        return resp["scores"]

    async def ping(self) -> bool:
        """
        Async health check — safe to call from async contexts only.
        Returns True if the server is alive and responding.
        """
        try:
            # Run the sync call in a thread to avoid blocking the event loop.
            resp = await asyncio.to_thread(
                self._call_sync, {"op": "ping"}
            )
            return resp.get("pong", False)
        except Exception:
            return False
