"""
Chainlit chat UI for LEX.

## Why Chainlit?

Chainlit is a Python framework purpose-built for LLM chat UIs. It gives
us, without writing frontend code:
  - a streaming chat widget (tokens appear as they arrive)
  - native "source cards" beneath messages (for our citations)
  - a sidebar with action buttons (for "Add a directive")
  - session state, history, and message threading

The alternative would be Streamlit + a lot of custom work, or two
weeks of React. Chainlit covers the 90% we need.

## How this UI talks to LEX

For v1 we import our handlers directly and call them in-process. That
means `lex ui` spins up everything it needs — no need to also have
the API server running. At deploy time we'd switch to calling the
FastAPI layer over HTTP; the structure here is unchanged, just the
handler bodies would become httpx calls.

## Streaming pattern

Chainlit messages are async. For a streaming message:

    msg = cl.Message(content="")
    await msg.send()
    async for token in stream:
        await msg.stream_token(token)
    await msg.update()

We bridge our handler's sync `on_token` callback to this by pushing
tokens onto a queue and letting the message loop drain it.

## Running it

    uv run lex ui            # opens http://localhost:8100

You still need: Qdrant + Redis (`docker compose up -d`), LLM server
(`uv run mlx_lm.server ...`). The UI will check these at startup.
"""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

import chainlit as cl
import redis.asyncio as aioredis
import structlog

from lex.commands import AnswerCmd, IngestCmd, RetrieveFilter
from lex.config import get_settings
from lex.generation import LLMUnreachable, build_answer_deps, handle_answer
from lex.worker import enqueue_ingest

log = structlog.get_logger(__name__)


# ===========================================================================
# Section 1: Startup — load models once per session
#
# Chainlit calls `@on_chat_start` for every new browser session. We
# stash the heavy stuff (answer handler deps, redis client) in the
# user session so each request reuses them. A fresh tab gets a fresh
# model load, which is fine for local dev.
#
# For production/cloud we'd promote these to process-global and use
# a dependency container — but that's over-engineering for v1.
# ===========================================================================

@cl.on_chat_start
async def on_chat_start() -> None:
    """Load models, wire handler, greet the user."""
    settings = get_settings()

    # Show a loading message while we warm up. Loading BGE-M3, the
    # reranker, and pinging the LLM takes ~10s on first run.
    loading = cl.Message(
        content="Loading models and connecting to services... "
                "(BGE-M3, reranker, LLM)."
    )
    await loading.send()

    try:
        answer_deps = await build_answer_deps(settings)
    except LLMUnreachable as e:
        loading.content = (
            f"**LLM endpoint unreachable.**\n\n{e}\n\n"
            f"Start the server and refresh this page."
        )
        await loading.update()
        return

    redis = aioredis.from_url(settings.redis.url, decode_responses=True)

    # Store per-session state. `user_session` is a dict Chainlit
    # maintains for us, keyed by the WebSocket connection.
    cl.user_session.set("settings", settings)
    cl.user_session.set("answer_deps", answer_deps)
    cl.user_session.set("redis", redis)

    # Replace the loading message with a welcome.
    loading.content = (
        "**LEX — EU directive Q&A**\n\n"
        "Ask a question about the ingested directives. Answers are "
        "grounded in the text and cite Article/Recital numbers.\n\n"
        "Use the **Add Directive** action below to ingest a new one."
    )
    # The "action" buttons underneath the welcome message — clicking
    # one fires an `@cl.action_callback`.
    loading.actions = [
        cl.Action(
            name="add_directive",
            label="➕ Add Directive",
            description="Ingest a new directive by CELEX ID", # ty: ignore[unknown-argument] 
            payload={},
        )
    ]
    await loading.update()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Close the Redis client when the session ends."""
    redis = cl.user_session.get("redis")
    if redis is not None:
        await redis.aclose()


# ===========================================================================
# Section 2: Main message handler — full RAG with streaming citations
# ===========================================================================

def _build_citation_graph(chunks: list, citations: list):
    """
    Build a Plotly force-directed graph of retrieved chunks and their
    cross-references. Returns a plotly Figure suitable for cl.Plotly.

    Nodes: retrieved chunks coloured by type, sized by rerank score.
    Edges: explicit cross-references between chunks (from cross_refs).
    Gold border: chunks actually cited in the answer.
    """
    import networkx as nx
    import plotly.graph_objects as go

    cited_ids = {c.chunk_id for c in citations}
    chunk_by_id = {rc.chunk.id: rc.chunk for rc in chunks}

    G = nx.Graph()

    # Add nodes
    for rc in chunks:
        c = rc.chunk
        if c.chunk_type == "recital":
            label = f"Recital {c.paragraph or '?'}"
        elif c.chunk_type == "annex":
            label = f"Annex {c.article or '?'}"
        else:
            label = f"Art. {c.article or '?'}"
            if c.paragraph:
                label += f"({c.paragraph})"
        G.add_node(c.id,
                   label=label,
                   cited=c.id in cited_ids,
                   chunk_type=c.chunk_type,
                   rerank=rc.rerank_score)

    # Add edges from cross_refs
    node_ids = set(G.nodes)
    seen: set[tuple[str, str]] = set()
    for rc in chunks:
        c = rc.chunk
        for ref_art in getattr(c, 'cross_refs', []):
            targets = [
                n for n in node_ids
                if chunk_by_id.get(n) and
                chunk_by_id[n].article == ref_art and
                n != c.id
            ]
            for t in targets:
                edge = tuple(sorted([c.id, t]))
                if edge not in seen:
                    seen.add(edge)
                    G.add_edge(c.id, t)

    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='#555'),
        hoverinfo='none',
    )

    # Nodes — split into cited and uncited for different styling
    type_colors = {
        'paragraph': '#4cc9f0',
        'article':   '#4361ee',
        'recital':   '#7209b7',
        'annex':     '#f72585',
    }

    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []
    border_colors = []

    for node_id in G.nodes:
        x, y = pos[node_id]
        d = G.nodes[node_id]
        node_x.append(x)
        node_y.append(y)
        rerank = d.get('rerank', 0.0)
        score_str = f" (score: {rerank:.2f})" if rerank > 0 else " (graph hop)"
        node_text.append(f"{d['label']}{' ★' if d['cited'] else ''}{score_str}")
        node_color.append(type_colors.get(d.get('chunk_type', 'recital'), '#888'))
        node_size.append(20 if d['cited'] else 14)
        border_colors.append('#f0a500' if d['cited'] else '#333')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n]['label'] for n in G.nodes],
        textposition='top center',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color=border_colors),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='Citation graph — gold = cited, size = rerank score',
                font=dict(size=12, color='#ccc'),
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=350,
            font=dict(color='#ccc'),
        )
    )
    return fig

@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    The main chat handler. If the user was prompted for a CELEX ID
    (via the Add Directive action), treat their message as the ID.
    Otherwise run full RAG.
    """
    # If the add-directive flow is in-flight, take this message as a
    # CELEX ID instead of a question.
    if cl.user_session.get("awaiting_celex"):
        cl.user_session.set("awaiting_celex", False)
        celex_id = message.content.strip()
        if not celex_id:
            await cl.Message(content="No CELEX ID provided. Cancelled.").send()
            return
        await _ingest_with_progress(celex_id)
        return

    answer_deps = cl.user_session.get("answer_deps")
    if answer_deps is None:
        await cl.Message(
            content="Models not loaded. Refresh the page."
        ).send()
        return

    # Build the command. Filters are default (language=en, no celex/article
    # restrictions). A fancier UI could expose filter controls; out of scope.
    cmd = AnswerCmd(
        query=message.content,
        filters=RetrieveFilter(language="en"),
        stream=True,
    )

    # Create the streaming reply message. `await msg.send()` makes it
    # appear in the UI immediately (empty). We fill it in as tokens arrive.
    reply = cl.Message(content="")
    await reply.send()

    # Bridge our handler's sync on_token callback → Chainlit's async
    # stream_token. We use an asyncio.Queue: callback pushes, the
    # consumer loop awaits-and-forwards.
    token_queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def _on_token(tok: str) -> None:
        token_queue.put_nowait(tok)

    async def _runner() -> None:
        """Run the handler to completion, then signal DONE."""
        try:
            result = await handle_answer(cmd, answer_deps, on_token=_on_token)
            token_queue.put_nowait(("RESULT", result))
        except Exception as e:
            log.exception("ui.answer.failed")
            token_queue.put_nowait(("ERROR", str(e)))
        finally:
            token_queue.put_nowait(DONE)

    # Kick off the handler concurrently with the token consumer loop.
    runner_task = asyncio.create_task(_runner())

    final_result = None
    final_error = None
    while True:
        item = await token_queue.get()
        if item is DONE:
            break
        if isinstance(item, tuple):
            tag, value = item
            if tag == "RESULT":
                final_result = value
            elif tag == "ERROR":
                final_error = value
            continue
        # Plain string token — push to the Chainlit message.
        await reply.stream_token(item)

    # Consumer loop exited — make sure the background task is clean.
    await runner_task

    if final_error:
        reply.content = f"**Error:** {final_error}"
        await reply.update()
        return

    # Finalise the message — Chainlit needs this to stop the "typing"
    # indicator and commit the final content.
    await reply.update()

    # Render citations as separate message elements. Chainlit displays
    # these as source cards below the message.
    if final_result and final_result.citations:
        seen_chunk_ids: set[str] = set()
        elements: list = []
        chunk_by_id = {rc.chunk.id: rc.chunk for rc in final_result.chunks}

        for cite in final_result.citations:
            if cite.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cite.chunk_id)
            chunk = chunk_by_id.get(cite.chunk_id)
            if chunk is None:
                continue
            if chunk.chunk_type == "recital":
                label = f"Recital {chunk.paragraph or '?'}"
            elif chunk.chunk_type == "annex":
                label = f"Annex {chunk.article or '?'}"
            else:
                label = f"Article {chunk.article or '?'}"
                if chunk.paragraph:
                    label += f"({chunk.paragraph})"
            label += f" — {chunk.celex_id}"
            elements.append(cl.Text(
                name=label,
                content=chunk.text,
                display="inline",
            ))

        # Citation graph visualisation
        fig = _build_citation_graph(
            final_result.chunks, final_result.citations
        )
        elements.append(cl.Plotly(
            name="Citation graph",
            figure=fig,
            display="inline",
        ))
        reply.elements = elements
        await reply.update()

# ===========================================================================
# Section 3: Add Directive action — ingest from the UI
#
# Clicking the "➕ Add Directive" button prompts for a CELEX ID, then
# fires an ingest command. We subscribe to the job's Redis pubsub
# channel and stream state transitions into a live message.
# ===========================================================================

@cl.action_callback("add_directive")
async def on_add_directive(action: cl.Action) -> None:
    """
    Handoff: set a flag in session state, prompt the user inline.

    Note: we deliberately do NOT use cl.AskUserMessage here. Using
    Ask*Message inside an action_callback is a known Chainlit bug
    that leaves the chat input stuck on "Stop Task" after the callback
    returns (see github.com/Chainlit/chainlit/issues/2204 and #2209).

    Instead, we flip a flag. The next user message is interpreted as
    a CELEX ID in on_message, not as a question. This keeps the
    action_callback tiny and synchronous-feeling, which Chainlit
    handles fine.
    """
    cl.user_session.set("awaiting_celex", True)
    await cl.Message(
        content="Enter the CELEX ID of the directive to ingest "
                "(e.g. `32018L1972`) as your next message."
    ).send()

async def _ingest_with_progress(celex_id: str) -> None:
    """
    Enqueue a CELEX ID, stream progress into a Chainlit message.

    Called from on_message when the `awaiting_celex` flag is set.
    Running in the on_message handler (not an action_callback) avoids
    the known Chainlit bug that causes the "Stop Task" button to hang.
    """
    settings = cl.user_session.get("settings")
    redis = cl.user_session.get("redis")
    if settings is None or redis is None:
        await cl.Message(content="Session not initialised. Refresh.").send()
        return

    cmd = IngestCmd(celex_id=celex_id, language="en")
    await enqueue_ingest(redis, settings, cmd)

    progress = cl.Message(content=f"**Queued** `{celex_id}` — waiting for worker...")
    await progress.send()

    job_key = f"{settings.redis.key_prefix}:job:{cmd.cmd_id}"
    channel = f"{settings.redis.key_prefix}:job:{cmd.cmd_id}:events"
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)

    TERMINAL = {"done", "failed"}
    OVERALL_TIMEOUT_S = 600.0

    async def _loop_until_terminal() -> str | None:
        data = await redis.hgetall(job_key)
        if data.get("state") in TERMINAL:
            progress.content = _render_progress(celex_id, data["state"], data)
            await progress.update()
            return data["state"]

        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            state = msg["data"]
            data = await redis.hgetall(job_key)
            progress.content = _render_progress(celex_id, state, data)
            await progress.update()
            if state in TERMINAL:
                return state
        return None

    try:
        try:
            await asyncio.wait_for(_loop_until_terminal(), timeout=OVERALL_TIMEOUT_S)
        except asyncio.TimeoutError:
            progress.content = (
                f"⌛ **Timed out** waiting for `{celex_id}`. "
                f"The worker may still be running — check its logs."
            )
            await progress.update()
    finally:
        try:
            await pubsub.reset()
        except Exception:
            pass

def _render_progress(celex_id: str, state: str, data: dict) -> str:
    """Format the progress message content."""
    # One-line status with a progress-ish indicator.
    emoji = {
        "queued":    "⏳",
        "fetching":  "⬇️",
        "parsing":   "📄",
        "chunking":  "✂️",
        "embedding": "🧮",
        "writing":   "💾",
        "done":      "✅",
        "failed":    "❌",
    }.get(state, "•")

    lines = [f"{emoji} **{state}** — `{celex_id}`"]
    if state == "done" and data.get("chunks"):
        lines.append(f"  {data['chunks']} chunks written to Qdrant.")
    if state == "failed" and data.get("error"):
        lines.append(f"  Error: `{data['error']}`")
    return "\n".join(lines)
