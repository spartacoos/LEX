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
    RetrieveFilter,
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
plainly: "The provided sources do not answer this question." If the \
sources contain related but partial information, summarise what is \
available and note what is missing — for example: "The sources describe \
functions of national regulatory authorities but do not provide a \
formal definition. The definition may appear in a directive or regulation \
not currently ingested."
4. Prefer direct, factual answers over general discussion. When quoting, \
keep quotes short.
5. Never invent Article or Recital numbers. Only cite what is shown."""

_HYDE_PROMPT = """\
Expand the following legal question into a short passage (2-4 sentences) \
using formal EU legal terminology. Do NOT invent or cite any article \
numbers, directive names, or specific legal references.
- Expand any abbreviations to their full legal terms
- Rephrase in the formal language that would appear in a directive text
- Do not answer the question — only rephrase and expand it

Question: {query}

Expansion:"""

from typing import Literal

QueryType = Literal["definition", "procedural", "negative", "general"]


def _classify_query(query: str) -> QueryType:
    """
    Lightweight heuristic query classifier — zero latency, no model call.

    Classifies into four types that require different retrieval strategies:

      definition  — "What is X?" / "Define X" / "What does X mean?"
                    Retrieval favours Art. 2 sub-chunks and article chunks.
                    HyDE is skipped — definitions are precise and short;
                    a hypothetical passage would match explanatory recitals
                    rather than the definition chunk we want.

      procedural  — "How does X work?" / "What is the procedure for Y?"
                    Retrieval strongly prefers operative articles.
                    HyDE is applied to bridge formal procedural language.

      negative    — "Does this directive cover X?" / "Is X regulated?"
                    Retrieval casts wide — scope exclusions appear in both
                    recitals and articles. HyDE is skipped — it would
                    generate a positive passage, the opposite of what we
                    want to retrieve for a scope/exclusion question.

      general     — everything else
                    Default: HyDE + hybrid search + chunk type boost.

    Design principles:
      - Procedural check runs before definition to handle "What is the
        X procedure?" correctly (procedural, not definitional).
      - Negative check runs before general to catch scope questions that
        start with auxiliary verbs.
      - All matching is case-insensitive on a stripped, lowercased copy.
      - No regex is applied to the original query string — only to `q`,
        the normalised lowercase version.
      - Patterns are ordered from most specific to least specific within
        each category.
    """
    # Normalise once — all pattern matching uses `q`
    q = query.lower().strip().rstrip('?').strip()

    # ------------------------------------------------------------------
    # 1. PROCEDURAL
    # Must run before definition because "What is the X procedure/process/
    # mechanism?" looks definitional by its opening but is procedural by
    # its subject. We check for procedural-subject words first.
    # ------------------------------------------------------------------
    _PROCEDURAL_SUBJECTS = (
        'procedure', 'process', 'mechanism', 'framework', 'regime',
        'system', 'steps', 'rules', 'requirements', 'conditions',
        'criteria', 'methodology', 'approach', 'method', 'workflow',
        'authorisation', 'authorization', 'notification', 'consultation',
        'designation', 'assessment', 'analysis', 'review', 'appeal',
        'enforcement', 'compliance', 'implementation',
    )

    # "What is the [procedural subject]..." — procedural despite "what is"
    if re.match(r'^what (is|are) (the|a|an)\b', q) and any(
        w in q for w in _PROCEDURAL_SUBJECTS
    ):
        return "procedural"

    # "How is/are/does/do/can/should/must/will..."
    if re.match(r'^how (is|are|does|do|can|should|must|will|would|shall)\b', q):
        return "procedural"

    # "How to..." / "How do I..." / "How does one..."
    if re.match(r'^how (to|do i|does one|do you)\b', q):
        return "procedural"

    # Explicit procedural openers
    if re.match(
        r'^(what is the procedure|what is the process|what are the steps|'
        r'what are the rules|what are the requirements|what are the conditions|'
        r'what are the criteria|describe the procedure|explain the procedure|'
        r'outline the procedure|what happens (when|if|after|before|during)|'
        r'under what conditions|in what circumstances|when (can|may|must|shall|should|will))',
        q
    ):
        return "procedural"

    # "Who may/can/must/shall..." — procedural (about authorisation/obligation)
    if re.match(r'^who (may|can|must|shall|should|is|are|has|have)\b', q):
        return "procedural"

    # ------------------------------------------------------------------
    # 2. DEFINITION
    # Plain "What is X?" / "What are X?" without procedural subjects.
    # Also "Define X", "What does X mean?", "What is meant by X?"
    # ------------------------------------------------------------------

    # "What is/are [a/an/the] X?" — classic definition query
    if re.match(r'^what (is|are)( a| an| the)?\b', q):
        return "definition"

    # "What does X mean?" / "What do X mean?"
    if re.match(r'^what do(es)?\b', q) and 'mean' in q:
        return "definition"

    # "Define X" / "Definition of X"
    if re.match(r'^(define|definition of|meaning of|concept of)\b', q):
        return "definition"

    # "What is meant by X?"
    if re.match(r'^what is meant by\b', q):
        return "definition"

    # "How is X defined?" — looks procedural but is definitional
    if re.match(r'^how is\b', q) and 'defined' in q:
        return "definition"

    # ------------------------------------------------------------------
    # 3. NEGATIVE / SCOPE
    # "Does this directive cover X?" / "Is X regulated?" / "Are X included?"
    # Also multi-word scope questions: "Does X apply to Y?"
    # ------------------------------------------------------------------
    _SCOPE_VERBS = (
        'cover', 'apply', 'include', 'regulate', 'govern', 'concern',
        'affect', 'address', 'deal with', 'extend to', 'fall within',
        'fall under', 'come within', 'encompass', 'pertain',
    )

    # "Does/do/is/are/can/will/would this/the directive..."
    if re.match(r'^(does|do|is|are|can|will|would|has|have)\b', q) and any(
        w in q for w in _SCOPE_VERBS
    ):
        return "negative"

    # "Is X covered/regulated/included/governed...?"
    if re.match(r'^is\b', q) and any(
        w in q for w in ('covered', 'regulated', 'included', 'governed',
                         'regulated', 'addressed', 'applicable', 'excluded')
    ):
        return "negative"

    # "Are X regulated/covered...?"
    if re.match(r'^are\b', q) and any(
        w in q for w in ('covered', 'regulated', 'included', 'governed',
                         'addressed', 'applicable', 'excluded')
    ):
        return "negative"

    # Explicit scope/exclusion openers
    if re.match(
        r'^(does this directive|does the directive|is this directive|'
        r'do these rules|do the rules|is .* excluded|are .* excluded|'
        r'is .* exempt|are .* exempt|does .* fall (under|within|outside))',
        q
    ):
        return "negative"

    # ------------------------------------------------------------------
    # 4. GENERAL — default
    # ------------------------------------------------------------------
    return "general"

def _hyde_is_usable(original: str, expanded: str) -> bool:
    """
    Heuristic guard: use the expanded query only if it's genuinely better
    than the original. Falls back to original if expansion looks hallucinated
    or degenerate.

    Rules (all generally applicable regardless of directive or model):
      1. Must be longer than the original — pure expansion, never contraction
      2. Must not contain bracketed placeholders like [Article X] or [Insert]
      3. Must not contain specific number patterns that look invented:
         "Article 15", "Article 2(3)" etc. — we want terminology not citations
      4. Must not be a near-copy of the original (model just repeated the query)
    """
    if len(expanded) <= len(original):
        return False
    # Bracketed placeholders
    if re.search(r'\[.{2,40}\]', expanded):
        return False
    # Invented article citations (number after "Article")
    if re.search(r'\bArticle\s+\d+', expanded):
        return False
    # Near-copy: if >80% of original words appear verbatim in same order
    orig_words = original.lower().split()
    exp_words = expanded.lower().split()
    if len(orig_words) > 3:
        matches = sum(1 for w in orig_words if w in exp_words)
        if matches / len(orig_words) > 0.85:
            return False
    return True

async def _hyde_expand(query: str, llm: LLMClient) -> str:
    """
    Hypothetical Document Embedding: generate a synthetic directive passage
    that would answer the query, then use it as the retrieval query.

    Why this works: the embedding of "what obligations apply to SMP undertakings?"
    is semantically distant from Arts. 68-74 which say "designated as having
    significant market power... shall... transparency... non-discrimination".
    The hypothetical passage generated from the question will use the same
    legal phrasing as the actual articles, bringing the query vector close
    to the document vectors.

    The original query is still used for reranking (cross-encoder sees the
    real question vs candidate passage, which is correct).
    """
    messages = [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}]
    tokens: list[str] = []
    async for tok in llm.stream(messages):
        tokens.append(tok)
    expanded = "".join(tokens).strip()

    if _hyde_is_usable(query, expanded):
        log.debug("hyde.expanded", query=query[:60], hyde_chars=len(expanded))
        return expanded
    else:
        log.debug("hyde.rejected", query=query[:60], hyde_chars=len(expanded),
                  reason="failed validity check — using original query")
        return query

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

def _extract_article_refs(text: str) -> set[str]:
    """
    Extract article numbers mentioned in generated text.

    Matches patterns like:
      "Article 68", "Articles 69 to 74", "Art. 76", "article 80"
      "pursuant to Articles 69, 70, 71"

    Returns a set of string article numbers e.g. {"68", "69", "70", "76"}.
    Used in two-stage retrieval to fetch articles the model referenced
    in a draft answer that did not appear in the initial top-5.
    """
    # Match "Article(s) N" or "Art. N" with optional comma/range continuations
    pattern = re.compile(
        r'\b[Aa]rt(?:icle)?s?\.?\s+(\d+(?:\s*(?:,|to|and)\s*\d+)*)',
    )
    refs: set[str] = set()
    for m in pattern.finditer(text):
        # Parse the captured group which may be "68", "69, 70, 71", "69 to 74"
        raw = m.group(1)
        nums = re.findall(r'\d+', raw)
        # For ranges like "69 to 74", expand to all integers
        if 'to' in raw.lower() and len(nums) == 2:
            start, end = int(nums[0]), int(nums[1])
            refs.update(str(n) for n in range(start, end + 1))
        else:
            refs.update(nums)
    return refs

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
    # ---- A. Classify query and decide retrieval strategy --------------
    query_type = _classify_query(cmd.query)
    logger.debug("answer.query_type", query_type=query_type,
                 query=cmd.query[:60])

    # ---- B. HyDE expansion (skipped for definitions and negatives) ----
    # Definitions: the answer is a short precise text — HyDE would
    # generate a passage that matches explanatory recitals rather than
    # the Art. 2 sub-chunk we want.
    # Negatives: HyDE would generate a positive passage ("this directive
    # covers X") which is the opposite of what we want to retrieve.
    use_hyde = query_type in ("procedural", "general")
    if use_hyde:
        hyde_text = await _hyde_expand(cmd.query, deps.llm)
        retrieval_query = hyde_text
    else:
        retrieval_query = cmd.query
        logger.debug("answer.hyde_skipped", reason=query_type)

    # ---- C. Build retrieve command with type-aware top_k --------------
    # Definitions get more candidates so Art. 2 sub-chunks have more
    # chance to surface even if dense score is slightly lower than a
    # recital that discusses the same concept.
    top_k = deps.settings.retrieval.top_k_rerank
    if query_type == "definition":
        top_k = min(top_k + 3, 10)  # slightly wider net for definitions

    retrieve_cmd = RetrieveCmd(
        cmd_id=cmd.cmd_id,
        query=retrieval_query,
        top_k=top_k,
        filters=cmd.filters,
    )
    retrieve_result = await handle_retrieve(retrieve_cmd, deps.retrieve)
    logger.info("answer.retrieved", chunks=len(retrieve_result.chunks))

    # ---- 2. Build prompt ----------------------------------------------
    messages = build_prompt(cmd.query, retrieve_result.chunks)
    # ---- 3. Stream from LLM — first pass (draft) ----------------------
    # We generate once to get a draft, extract article references from it,
    # then do a targeted second retrieval to pull in any articles the model
    # cited that were not in the initial top-5. If the second pass finds
    # new chunks, we regenerate with the enriched context.
    # If it finds nothing new, the draft becomes the final answer.
    messages = build_prompt(cmd.query, retrieve_result.chunks)

    pieces: list[str] = []
    # First pass: collect silently (no on_token streaming yet)
    async for token in deps.llm.stream(messages):
        pieces.append(token)
    draft = "".join(pieces)
    logger.info("answer.draft", chars=len(draft))

    # ---- 4. Second retrieval pass -------------------------------------
    # Extract article numbers mentioned in the draft. Fetch any that are
    # not already in our retrieved set.
    # Articles already in our retrieved set
    existing_articles = {
        rc.chunk.article for rc in retrieve_result.chunks
        if rc.chunk.article is not None
    }
    # Articles the model cited via [N] markers — these are already covered
    cited_indices = {
        int(n) - 1
        for n in re.findall(r'\[(\d+)\]', draft)
        if n.isdigit() and 1 <= int(n) <= len(retrieve_result.chunks)
    }
    cited_articles = {
        retrieve_result.chunks[i].chunk.article
        for i in cited_indices
        if retrieve_result.chunks[i].chunk.article is not None
    }
    # Articles mentioned in prose but not retrieved or cited — genuinely missing
    draft_refs = _extract_article_refs(draft)
    missing_refs = draft_refs - existing_articles - cited_articles
    logger.debug("answer.two_stage",
                 draft_refs=sorted(draft_refs),
                 existing=sorted(existing_articles),
                 cited=sorted(cited_articles),
                 missing=sorted(missing_refs))

    if missing_refs:
        # Fetch each missing article directly by metadata filter.
        # We collect all new chunks and merge with the original set,
        # keeping the original ranking order and appending new chunks.
        new_chunks: list[RetrievedChunk] = []
        for article_num in sorted(missing_refs):
            targeted_cmd = RetrieveCmd(
                cmd_id=cmd.cmd_id,
                query=cmd.query,
                top_k=2,  # at most 2 paragraphs per article
                filters=RetrieveFilter(
                    celex_id=cmd.filters.celex_id,
                    language=cmd.filters.language,
                    article=article_num,
                ),
            )
            targeted_result = await handle_retrieve(targeted_cmd, deps.retrieve)
            for rc in targeted_result.chunks:
                # Only add if not already present
                if rc.chunk.id not in {c.chunk.id for c in retrieve_result.chunks}:
                    new_chunks.append(rc)

        if new_chunks:
            logger.info("answer.two_stage_enriched",
                        new_chunks=len(new_chunks),
                        articles=sorted(missing_refs))
            # Merge: original chunks first (preserve ranking), new chunks appended.
            # Rebuild prompt with enriched context and regenerate.
            from .commands import RetrieveResult
            enriched_chunks = retrieve_result.chunks + new_chunks
            retrieve_result = retrieve_result.model_copy(
                update={"chunks": enriched_chunks}
            )
            messages = build_prompt(cmd.query, enriched_chunks)

            # Regenerate with enriched context — this time stream to caller.
            pieces = []
            async for token in deps.llm.stream(messages):
                pieces.append(token)
                if on_token is not None:
                    on_token(token)
            answer = "".join(pieces)
            logger.info("answer.regenerated", chars=len(answer))
        else:
            # No new chunks found — use the draft, stream it to caller.
            answer = draft
            if on_token is not None:
                for token in draft:
                    on_token(token)
    else:
        # No missing refs — draft is final, stream it.
        answer = draft
        if on_token is not None:
            for token in draft:
                on_token(token)

    logger.info("answer.streamed", chars=len(answer))

    # ---- 5. Extract citations -----------------------------------------
    citations = extract_citations(answer, retrieve_result.chunks)
    logger.info("answer.cited", citations=len(citations))

    return AnswerResult(
        cmd_id=cmd.cmd_id,
        query=cmd.query,
        answer=answer,
        citations=citations,
        chunks=retrieve_result.chunks,
    )

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
