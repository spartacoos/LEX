"""
Profile loader — maps a YAML profile to Settings overrides.

Profiles live in profiles/ at the project root. Each YAML file fully
describes one hardware/model combination. The `lex config` wizard writes
the chosen profile name into .env as LEX_PROFILE=<name>; Settings picks
it up at startup and applies the overrides before any other env var.

Env vars always win over profile defaults — so LEX_LLM__MAX_TOKENS=8192
in .env still overrides whatever the profile says, letting users tweak
without touching the profile files.

Design note: profiles are *data*, not code. A new model = a new YAML
file. Zero Python changes required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Profiles directory relative to this file's package root (src/lex/ → root)
_PROFILES_DIR = Path(__file__).parent.parent.parent / "profiles"


def list_profiles() -> list[str]:
    """Return names of all available profiles (stem of each .yaml file)."""
    if not _PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str) -> dict[str, Any]:
    """
    Load a profile by name (filename stem, no extension).

    Returns the raw YAML dict. Callers — specifically apply_profile() —
    are responsible for mapping keys to Settings fields.

    Raises FileNotFoundError with a helpful message if the profile
    doesn't exist.
    """
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_profiles()
        raise FileNotFoundError(
            f"Profile '{name}' not found at {path}.\n"
            f"Available profiles: {available or '(none — check profiles/ directory)'}"
        )

    try:
        import yaml  # PyYAML, already a transitive dep via deepeval
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for profile loading. "
            "Run: pip install pyyaml"
        ) from exc

    with path.open() as f:
        data = yaml.safe_load(f)

    log.debug("profile.loaded", name=name, path=str(path))
    return data or {}


def profile_to_env(name: str) -> dict[str, str]:
    """
    Translate a profile into a flat dict of LEX_ env-var overrides.

    This dict is what `lex config` writes to .env. Individual env vars
    can still override individual values; this just sets the defaults.
    """
    p = load_profile(name)
    out: dict[str, str] = {}

    llm = p.get("llm", {})
    backend: str = llm.get("backend", "llamacpp")

    # Base URL depends on backend.
    if backend == "remote":
        out["LEX_LLM__BASE_URL"] = llm.get("base_url", "https://api.openai.com/v1")
        key_env = llm.get("api_key_env", "OPENAI_API_KEY")
        out["LEX_LLM__API_KEY_ENV"] = key_env
    else:
        port = llm.get("port", 8080)
        out["LEX_LLM__BASE_URL"] = f"http://localhost:{port}/v1"
        out["LEX_LLM__API_KEY"] = "not-needed-for-local"

    out["LEX_LLM__MODEL"] = llm.get("model_name", "gemma-4-e4b-it")
    out["LEX_LLM__MAX_TOKENS"] = str(llm.get("max_tokens", 2048))
    out["LEX_LLM__TEMPERATURE"] = str(llm.get("temperature", 0.1))
    out["LEX_LLM__CTX_SIZE"] = str(llm.get("ctx_size", 32768))
    out["LEX_LLM__N_GPU_LAYERS"] = str(llm.get("n_gpu_layers", -1))
    out["LEX_LLM__BACKEND"] = backend

    if llm.get("model_file"):
        out["LEX_LLM__MODEL_FILE"] = llm["model_file"]
    if llm.get("hf_repo"):
        out["LEX_LLM__HF_REPO"] = llm["hf_repo"]
    if llm.get("port"):
        out["LEX_LLM__PORT"] = str(llm["port"])

    emb = p.get("embedding", {})
    out["LEX_EMBEDDING__DEVICE"] = emb.get("device", "auto")

    rnk = p.get("reranker", {})
    out["LEX_RERANKER__DEVICE"] = rnk.get("device", "cpu")

    return out
