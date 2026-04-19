"""
Generation: retrieve context, prompt the LLM, extract citations.

## Retrieval-Augmented Generation (RAG), briefly

A vanilla LLM answers from its training data — reliable for common
knowledge, unreliable for anything specific (a recent directive, an
internal document, last quarter's numbers). It also has no way to
point at *where* a fact came from.

RAG fixes both by prepending retrieved passages to the prompt. The
LLM answers *from* those passages, and we can cite them by reference
because we know exactly which ones we handed it.

## The citation mechanism

There are two broad approaches for "which source backed this sentence?"

1. **Inline markers.** We number each retrieved chunk `[1]`, `[2]`,
   ..., include the numbered sources in the prompt, and instruct the
   LLM to write `[N]` immediately after any claim it's drawing from
   source N. We then post-process the answer: scan for `[N]`, look up
   the chunk by its number, record the character offsets.

2. **Post-hoc attribution.** Generate the answer first, then run an
   entailment / embedding model over (claim, source) pairs to decide
   which source best supports each claim.

(1) is dramatically simpler, works on small models, and is what
Perplexity/Claude Citations/etc. use. We do (1). Post-hoc attribution
can be layered on later if evaluation shows the model cheats on
markers.

## Streaming

The OpenAI protocol supports streaming via Server-Sent Events. We use
the async SDK so we can yield tokens as they arrive without blocking
the event loop. The CLI prints them in real time; the Chainlit UI
will bind them to its own streaming widget in M5.

## "I don't know" is a valid answer

Critical for legal Q&A: we'd rather have the model refuse than
hallucinate a plausible-sounding Article number. The system prompt
explicitly tells it to say so when the context is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import structlog
from openai import AsyncOpenAI

from .commands import (
    AnswerCmd,
    AnswerResult,
    Citation,
    RetrieveCmd,
    RetrievedChunk,
)
from .config import Settings
from .retrieval import RetrieveDeps, handle_retrieve

from .tracing import observe

log = structlog.get_logger(__name__)


# ===========================================================================
# Section 1: Prompt construction
#
# The prompt is the interface between "what we know" and "what the
# model sees." Keeping it in one function makes it obvious what we're
# telling the LLM, and easy to A/B different phrasings during eval.
# ===========================================================================

# The system prompt does three jobs:
#   1. Sets the role ("answer questions about EU directives").
#   2. Mandates citing sources with [N] markers.
#   3. Authorises (and expects) "I don't know" when context is thin.
#
# Kept short because small models pay attention to brevity. Anything
# longer and Gemma-4 E4B starts drifting into generic legal-assistant
# boilerplate.
SYSTEM_PROMPT = """\
You are a legal research assistant specialising in European Union law. \
You answer questions about EU directives based STRICTLY on the numbered \
sources provided in each user message.

Rules:
1. Every factual claim MUST be followed by a bracketed citation like [1] \
or [3] referring to the numbered sources.
2. Use ONLY the provided sources. Do not draw on outside knowledge.
3. If the sources do not contain enough information to answer, say so \
plainly: "The provided sources do not answer this question." Do not \
speculate.
4. Prefer direct, factual answers over general discussion. When quoting, \
keep quotes short.
5. Never invent Article or Recital numbers. Only cite what is shown."""


def _format_source(idx: int, rc: RetrievedChunk) -> str:
    """
    Render one retrieved chunk as a numbered source for the prompt.

    Example output:
        [1] Article 82 of Directive 2018/1972 (VHCN):
        Article 82 BEREC guidelines on very high capacity networks...
    """
    chunk = rc.chunk

    # Build the human-readable reference (where the text lives).
    if chunk.chunk_type == "recital":
        ref = f"Recital {chunk.paragraph or '?'}"
    elif chunk.chunk_type == "annex":
        ref = f"Annex {chunk.article or '?'}"
    else:
        ref = f"Article {chunk.article or '?'}"
        if chunk.paragraph:
            ref += f"({chunk.paragraph})"

    # Identify the directive. For v1 we always have a CELEX-shaped
    # citation; later versions might show a human name if available.
    return f"[{idx}] {ref} of {chunk.celex_id}:\n{chunk.text}"

@observe(name="build_prompt")
def build_prompt(query: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """
    Assemble the chat messages list sent to the LLM.

    Returns a list in OpenAI chat format:
        [{"role": "system", "content": ...},
         {"role": "user",   "content": ...}]
    """
    # Number sources from 1 so citation markers line up with natural
    # reading order. 1-indexed matters — humans write "[1]", not "[0]".
    sources_block = "\n\n".join(
        _format_source(i + 1, rc) for i, rc in enumerate(chunks)
    )

    user_msg = (
        f"Question: {query}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Answer the question using ONLY these sources. "
        f"Cite with [N] after each claim."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ===========================================================================
# Section 2: LLM client
#
# We talk OpenAI protocol over HTTP. The local MLX-served Gemma is one
# possible backend; OpenAI proper, vLLM, Together, Groq, etc. are all
# drop-in replacements — change the base_url in .env, zero code change.
# ===========================================================================

class LLMClient:
    """
    Thin async wrapper around `openai.AsyncOpenAI` for our streaming use.

    The OpenAI SDK does everything we need — retries, timeouts, SSE
    parsing. We just narrow it to the one call shape we use.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            timeout=settings.llm.timeout_s,
        )
        self._model = settings.llm.model
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens

    async def stream(self, messages: list[dict[str, str]]):
        """
        Stream completion tokens as they arrive.

        Async generator yielding strings (one per SSE delta). Caller
        concatenates to form the full answer.
        """
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        async for event in stream:
            # Each event has a choices[0].delta.content with the new
            # token text (or None if the delta is something else, e.g.
            # a role header on the first chunk).
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    async def ping(self) -> None:
        """
        Fail fast if the endpoint isn't reachable.

        Called at handler construction so we don't wait the full
        generation timeout before reporting that the LLM server isn't
        running. Uses a cheap `GET /models` on the OpenAI-compatible
        endpoint (every server implementing the protocol exposes it).
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._settings.llm.base_url}/models")
                r.raise_for_status()
        except Exception as e:
            raise LLMUnreachable(
                f"LLM endpoint {self._settings.llm.base_url} is unreachable: {e}. "
                f"Start the local server — for MLX on Apple Silicon: "
                f"`uv run mlx_lm.server --model mlx-community/gemma-4-E4B-it-4bit "
                f"--port 8080`"
            ) from e


class LLMUnreachable(RuntimeError):
    """Raised when the LLM endpoint isn't responding."""


# ===========================================================================
# Section 3: Citation extraction
#
# The model was instructed to write `[1]`, `[2]`, ..., markers. We scan
# the finished answer for those markers, record their character spans,
# and map each back to the corresponding chunk.
#
# Edge cases we handle:
#   - Multi-citation groups like "[1,3]" or "[1][2]" — we split into
#     individual Citations so the UI can highlight each source
#     separately.
#   - Markers referring to numbers the model invented ("[7]" when we
#     only gave 5 sources) — we drop these; they're hallucinations,
#     and rendering them as broken links is worse than ignoring them.
# ===========================================================================

# Matches `[N]` or `[N, M]` or `[N,M]` or `[N,M,O]` — the whole group.
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(
    answer: str, chunks: list[RetrievedChunk]
) -> list[Citation]:
    """
    Scan the answer for [N] markers, convert each to a Citation.

    A marker like `[1,3]` produces two Citations — one for chunk 1 and
    one for chunk 3 — both pointing at the same character span in the
    answer. The UI can draw one underline and attach two source cards,
    or stack two highlights; that's the UI's choice.
    """
    citations: list[Citation] = []

    for match in _CITE_RE.finditer(answer):
        span_start = match.start()
        span_end = match.end()
        # Parse the comma-separated numbers inside the brackets.
        nums = [int(n.strip()) for n in match.group(1).split(",")]

        for n in nums:
            # Indices in the prompt were 1-based, chunks list is 0-based.
            if not (1 <= n <= len(chunks)):
                # Hallucinated citation — ignore.
                continue
            chunk = chunks[n - 1].chunk
            citations.append(Citation(
                chunk_id=chunk.id,
                article=chunk.article,
                paragraph=chunk.paragraph,
                span_start=span_start,
                span_end=span_end,
            ))

    return citations


# ===========================================================================
# Section 4: Handler
#
# Orchestrates retrieve → prompt → stream → citations. Supports both
# streaming and non-streaming modes: CLI and UI use streaming, tests
# use non-streaming for deterministic assertions.
# ===========================================================================

@dataclass
class AnswerDeps:
    """Dependencies for the answer handler, bundled for test injection."""
    settings: Settings
    retrieve: RetrieveDeps
    llm: LLMClient

@observe(name="answer")
async def handle_answer(
    cmd: AnswerCmd,
    deps: AnswerDeps,
    *,
    on_token=None,
) -> AnswerResult:
    """
    Full RAG: retrieve, prompt, generate (streaming), extract citations.

    Parameters
    ----------
    cmd : AnswerCmd
        The user's query + any filters.
    deps : AnswerDeps
        Built via build_answer_deps.
    on_token : callable or None
        If provided, called with each streamed token as a str. The CLI
        uses this to print tokens live; tests pass None and just read
        the final AnswerResult. Note: this is always invoked under
        streaming; `cmd.stream` only affects whether the CALLER of the
        engine is expected to consume an iterator — we leave that API
        to M4's HTTP surface. For now, on_token is the sole extension
        point.
    """
    cmd_id = str(cmd.cmd_id)
    logger = log.bind(cmd_id=cmd_id, query=cmd.query[:80])

    # ---- 1. Retrieve --------------------------------------------------
    # Reuse the M2 handler rather than duplicating the pipeline here.
    # RetrieveCmd has no `stream` so just construct it from AnswerCmd.
    retrieve_cmd = RetrieveCmd(
        cmd_id=cmd.cmd_id,      # same correlation ID
        query=cmd.query,
        top_k=deps.settings.retrieval.top_k_rerank,
        filters=cmd.filters,
    )
    retrieve_result = await handle_retrieve(retrieve_cmd, deps.retrieve)
    logger.info("answer.retrieved", chunks=len(retrieve_result.chunks))

    if not retrieve_result.chunks:
        # Nothing to ground the answer in. Return a plain refusal
        # without calling the LLM at all — saves tokens and time.
        return AnswerResult(
            cmd_id=cmd.cmd_id,
            query=cmd.query,
            answer="The provided sources do not answer this question.",
            citations=[],
            chunks=[],
        )

    # ---- 2. Build prompt ----------------------------------------------
    messages = build_prompt(cmd.query, retrieve_result.chunks)

    # ---- 3. Stream from LLM -------------------------------------------
    # Accumulate tokens into the final answer while optionally forwarding
    # them to the caller's on_token callback.
    pieces: list[str] = []
    async for token in deps.llm.stream(messages):
        pieces.append(token)
        if on_token is not None:
            on_token(token)
    answer = "".join(pieces)
    logger.info("answer.streamed", chars=len(answer))

    # ---- 4. Extract citations -----------------------------------------
    citations = extract_citations(answer, retrieve_result.chunks)
    logger.info("answer.cited", citations=len(citations))

    return AnswerResult(
        cmd_id=cmd.cmd_id,
        query=cmd.query,
        answer=answer,
        citations=citations,
        chunks=retrieve_result.chunks,
    )


# ===========================================================================
# Section 5: Factory
# ===========================================================================

async def build_answer_deps(settings: Settings) -> AnswerDeps:
    """
    Construct real dependencies for the answer handler.

    Async because we ping the LLM endpoint at construction time — this
    turns "LLM server isn't running" from a 120s hang into a clear
    error the caller can display immediately.
    """
    from .retrieval import build_retrieve_deps

    retrieve_deps = build_retrieve_deps(settings)
    llm = LLMClient(settings)
    await llm.ping()
    return AnswerDeps(settings=settings, retrieve=retrieve_deps, llm=llm)
