"""
Typed configuration for LEX.

## Reading order

1.  A profile is loaded if LEX_PROFILE is set.  The profile YAML
    provides baseline defaults for the hardware/model combination.
2.  Individual LEX_* env vars (and .env) override the profile.
3.  Hard-coded Python defaults are the last resort.

This gives three layers of override, from coarsest to finest:
    profile  <  .env file  <  shell env vars

## Env-var mapping

Nested settings use __ as the delimiter:

    LEX_LLM__BASE_URL       → settings.llm.base_url
    LEX_EMBEDDING__DEVICE   → settings.embedding.device

The profile name itself is:

    LEX_PROFILE=gemma4-e4b-mlx

Run `lex config` to write a .env interactively, or copy .env.example.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Nested settings groups
# ---------------------------------------------------------------------------

class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection_prefix: str = "eurlex"
    collection_version: str = "v1"


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "lex"


class LLMSettings(BaseModel):
    """
    LLM endpoint.  Talks OpenAI protocol regardless of backend.

    Extra fields (backend, model_file, hf_repo, port, ctx_size,
    n_gpu_layers) are used by `lex serve-llm` to *start* the local
    server.  They are ignored when backend=remote.
    """
    base_url: str = "http://localhost:8080/v1"
    model: str = "gemma-4-e4b-it"
    api_key: str = "not-needed-for-local"
    # Which key to read from the *environment* when backend=remote.
    # e.g. "OPENAI_API_KEY" → os.environ["OPENAI_API_KEY"]
    api_key_env: str = ""

    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_s: float = 120.0

    # Server-launch parameters (used by `lex serve-llm` only).
    backend: Literal["llamacpp", "mlx", "remote"] = "llamacpp"
    model_file: str = ""          # local GGUF filename inside models/
    hf_repo: str = ""             # HF repo to download from
    port: int = 8080
    ctx_size: int = 32768
    n_gpu_layers: int = -1        # -1 = all layers to GPU

    @model_validator(mode="after")
    def resolve_api_key(self) -> "LLMSettings":
        """If api_key_env is set, read the actual key from the environment."""
        if self.api_key_env:
            val = os.environ.get(self.api_key_env, "")
            if val:
                self.api_key = val
            else:
                log.warning(
                    "config.api_key_env_missing",
                    env_var=self.api_key_env,
                    hint="Set that env var or use a local backend",
                )
        return self


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-m3"
    max_tokens: int = 1024
    batch_size: int = 16
    dense_dim: int = 1024
    device: str = "auto"


class RerankerSettings(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 4
    device: str = "auto"


class ChunkingSettings(BaseModel):
    max_chars: int = 1500
    sentence_overlap: int = 2


class RetrievalSettings(BaseModel):
    top_k_dense: int = 20
    top_k_rerank: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


class EvalJudgeSettings(BaseModel):
    """
    Override the judge LLM for `pytest -m eval`.

    Leave base_url unset → falls back to the RAG LLM.
    Set LEX_EVAL_JUDGE__BASE_URL + LEX_EVAL_JUDGE__MODEL to point at a
    bigger, smarter model without touching the RAG config.
    """
    base_url: str | None = None
    model: str | None = None
    api_key: str = "sk-local"
    api_key_env: str = ""
    timeout_s: float = 600.0
    temperature: float = 0.0
    max_tokens: int = 4096

    @model_validator(mode="after")
    def resolve_api_key(self) -> "EvalJudgeSettings":
        if self.api_key_env:
            val = os.environ.get(self.api_key_env, "")
            if val:
                self.api_key = val
        return self


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEX_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Active profile name.  `lex config` writes this.
    profile: str = ""

    data_dir: Path = Path(".lex")
    default_language: str = "en"

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    eval_judge: EvalJudgeSettings = Field(default_factory=EvalJudgeSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    def collection_name(self, language: str) -> str:
        return (
            f"{self.qdrant.collection_prefix}"
            f"_{language}"
            f"_{self.qdrant.collection_version}"
        )

    def bm25_idf_path(self, language: str) -> Path:
        """Path to the BM25 IDF table for a given language."""
        return self.data_dir / f"bm25_idf_{language}.json"

    def model_socket_path(self) -> Path:
        """Unix socket path for the model server daemon."""
        return self.data_dir / "models.sock"


def _apply_profile(name: str) -> None:
    """
    Write profile overrides into os.environ *before* pydantic-settings
    parses them.  Only sets keys that aren't already in the environment
    (env vars always win).
    """
    from .profile import profile_to_env
    for key, value in profile_to_env(name).items():
        if key not in os.environ:
            os.environ[key] = value
    log.info("config.profile_applied", profile=name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Process-wide singleton.

    Profile is applied here — before pydantic-settings reads env vars —
    so that individual LEX_* vars still override the profile.
    """
    # Peek at the env directly to avoid a partial Settings construction.
    profile_name = os.environ.get("LEX_PROFILE", "")
    if not profile_name:
        # Also check .env manually (pydantic hasn't read it yet).
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("LEX_PROFILE="):
                    profile_name = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if profile_name:
        try:
            _apply_profile(profile_name)
        except FileNotFoundError as e:
            log.warning("config.profile_not_found", error=str(e))

    return Settings()
