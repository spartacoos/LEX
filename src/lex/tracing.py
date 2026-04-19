"""
Langfuse tracing shim.

## What Langfuse is

An LLM observability platform — think "Datadog for RAG pipelines."
Every AnswerCmd becomes a trace: you see the query, which chunks were
retrieved (and with what scores), the exact prompt, the completion,
and the extracted citations. Invaluable when an answer goes sideways
and you need to figure out whether retrieval or generation was at
fault.

## Why a shim?

We want Langfuse to be *optional*: LEX runs fine without it, and
turning it on shouldn't require code changes. Langfuse's own SDK
mostly achieves this — the `@observe` decorator silently no-ops when
LANGFUSE_PUBLIC_KEY isn't set — but we centralise the import here so
the rest of the code imports one symbol (`observe`) and never needs
to know whether Langfuse is available.

## Enabling

Uncomment the Langfuse service in docker-compose.yml, then set:

    LANGFUSE_PUBLIC_KEY=pk-...
    LANGFUSE_SECRET_KEY=sk-...
    LANGFUSE_HOST=http://localhost:3000

(Keys are minted in the Langfuse UI after first start.)
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def _have_langfuse() -> bool:
    """Langfuse is 'on' when the SDK has real credentials to work with."""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


# Lazy import: only pull Langfuse into memory if it's actually wanted.
if _have_langfuse():
    from langfuse.decorators import observe as _lf_observe
    observe = _lf_observe
else:
    def observe(*dargs, **dkwargs):
        """
        No-op stand-in for langfuse.decorators.observe.

        Langfuse's real decorator is callable both as `@observe` and
        `@observe(name="...")`. We match that shape so call sites don't
        have to care which mode they're in.
        """
        # Case: bare `@observe` (first arg is the function to decorate).
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        # Case: `@observe(name="foo")` — return an identity decorator.
        def _identity(fn: F) -> F:
            return fn
        return _identity
