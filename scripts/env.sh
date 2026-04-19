#!/usr/bin/env bash
# Source this from your shell before working on LEX on Linux:
#   source scripts/env.sh
#
# Purpose: add all nvidia-* lib directories from the venv to
# LD_LIBRARY_PATH so llama-cpp-python can find libcudart, libcublas,
# libcudnn, etc. at runtime.
#
# Harmless on macOS — the find returns nothing and the var just stays
# whatever it was. Idempotent — re-sourcing doesn't duplicate entries.

if [[ ! -d "$(dirname "${BASH_SOURCE[0]}")/../.venv" ]]; then
  echo "No .venv found. Run 'uv sync --extra <platform>' first." >&2
  return 1 2>/dev/null || exit 1
fi

_LEX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_NV_LIBS=$(find "$_LEX_ROOT/.venv" -path '*/nvidia/*/lib' -type d 2>/dev/null | tr '\n' ':')

if [[ -n "$_NV_LIBS" ]]; then
  # Prepend, but strip any previous instances first to stay idempotent.
  _CLEAN_LDL=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$_LEX_ROOT/.venv" | tr '\n' ':' | sed 's/:$//')
  export LD_LIBRARY_PATH="${_NV_LIBS%:}${_CLEAN_LDL:+:$_CLEAN_LDL}"
  echo "LEX: LD_LIBRARY_PATH configured with $(echo "$_NV_LIBS" | tr ':' '\n' | grep -c /) nvidia lib dirs."
fi

unset _LEX_ROOT _NV_LIBS _CLEAN_LDL
