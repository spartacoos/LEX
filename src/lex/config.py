"""
Typed configuration for LEX.

## Why pydantic-settings?

Twelve-factor config says "config comes from the environment." Python
has `os.getenv`, which is fine for one or two strings but collapses
when you want types, defaults, nesting, or validation. pydantic-settings
gives us all of that for free: a `BaseSettings` subclass reads from
env vars (and `.env` files) and hands back a typed object.

## Reading this file

Each settings *group* is a nested model (Qdrant, Redis, LLM, ...). The
top-level `Settings` class composes them. Env vars map to fields using
the `LEX_` prefix and `__` as a nesting separator:

    LEX_QDRANT__URL=http://localhost:6333  →  settings.qdrant.url
    LEX_LLM__MODEL=mlx-community/gemma-4-E4B-it-4bit             →  settings.llm.model

We build a single `get_settings()` singleton so config is read once.
Every subsystem imports `get_settings()` — nobody reads env vars directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Nested groups. Each group corresponds to one subsystem.
# ---------------------------------------------------------------------------

class QdrantSettings(BaseModel):
    """Connection info for the Qdrant vector DB."""
    url: str = "http://localhost:6333"
    # Collection naming convention from SPEC §6: eurlex_{lang}_{version}.
    # We store the *prefix* and the *version*; the language comes from the
    # command at runtime so a single deployment can serve many languages.
    collection_prefix: str = "eurlex"
    collection_version: str = "v1"


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    # Prefix for all LEX keys so we don't collide with anything else that
    # might share this Redis.
    key_prefix: str = "lex"


class LLMSettings(BaseModel):
    """
    LLM endpoint config. Deliberately OpenAI-shaped: the local
    llama-cpp-python server, OpenAI proper, a cloud vLLM endpoint, and
    Together/Groq all speak this protocol, so swapping = change base_url.
    """
    base_url: str = "http://localhost:8080/v1"
    model: str = "mlx-community/gemma-4-E4B-it-4bit"
    api_key: str = "not-needed-for-local"
    # Generation knobs. Low temp because legal Q&A wants determinism.
    temperature: float = 0.1
    max_tokens: int = 1024
    # How long to wait for a completion before giving up. Generous
    # because CPU inference on a laptop is slow.
    timeout_s: float = 120.0


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-m3"
    # BGE-M3 is trained to handle up to 8192 tokens; we cap at 1024 because
    # our chunks are at most ~1500 chars and we'd rather batch more than
    # pad more.
    max_tokens: int = 1024
    # Batch size tuned for laptop CPU. Raise on GPU hosts.
    batch_size: int = 16
    # Dense dimension for BGE-M3. Hardcoded because changing the model
    # means changing the collection schema — not a silent env-var flip.
    dense_dim: int = 1024


class RerankerSettings(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 4 #was 16 - dropped for MPS stability on longer chunks


class ChunkingSettings(BaseModel):
    """See SPEC §6."""
    max_chars: int = 1500
    # When a paragraph exceeds max_chars and we fall back to sentence
    # splitting, we keep this many sentences of overlap between chunks
    # so context isn't sliced through a reference.
    sentence_overlap: int = 2


class RetrievalSettings(BaseModel):
    top_k_dense: int = 20        # hybrid search: initial candidate pool
    top_k_rerank: int = 5        # final answer context
    # Hybrid fusion weights. 0.5/0.5 is a safe default; tune per language.
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


# ---------------------------------------------------------------------------
# Top-level settings. This is the thing the rest of the code imports.
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Read at startup. Everything downstream takes a `Settings` or reaches
    for `get_settings()`.
    """
    model_config = SettingsConfigDict(
        env_prefix="LEX_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore stray env vars rather than crashing on them.
        extra="ignore",
    )

    # Where on disk we cache anything that isn't a Docker volume
    # (e.g. local copies of fetched Formex for debugging).
    data_dir: Path = Path(".lex")

    # Default language for commands that don't specify one.
    default_language: str = "en"

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    # ---- Helpers ----------------------------------------------------------

    def collection_name(self, language: str) -> str:
        """Resolve the Qdrant collection name for a given language.

        Centralised so the naming scheme lives in exactly one place. If we
        ever change the scheme (e.g. to include embedding model), we flip
        it here and every subsystem picks it up.
        """
        return f"{self.qdrant.collection_prefix}_{language}_{self.qdrant.collection_version}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Read the env once, reuse forever."""
    return Settings()
