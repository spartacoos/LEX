# LEX

Natural-language Q&A over EU directives. Local-first, reproducible,
incrementally deployable.

Ask "What penalties are provided for?" — get an answer grounded in the
actual text, with citations back to the Article and Paragraph.

## Stack

- **Ingestion** — Formex XML from EUR-Lex CELLAR, structure-preserving chunker
- **Retrieval** — BGE-M3 dense + sparse embeddings in Qdrant, BGE-reranker-v2-m3 cross-encoder
- **Generation** — any OpenAI-protocol LLM (MLX on Mac, llama.cpp on Linux)
- **API** — FastAPI with SSE streaming
- **Worker** — Redis BLPOP consumer for background ingestion
- **UI** — Chainlit chat with inline citations
- **Eval** — DeepEval + custom citation-correctness metric

## Platform matrix

|                        | macOS (Apple Silicon)       | Linux + NVIDIA GPU             | Linux CPU-only              |
|------------------------|------------------------------|-------------------------------|------------------------------|
| LLM server             | `mlx-lm` on host (Metal)     | `llama-cpp-python` on host (CUDA) | `llama-cpp-python` on host (CPU) |
| Embedder / reranker    | PyTorch MPS                  | PyTorch CUDA                  | PyTorch CPU                  |
| Qdrant / Redis         | Docker                       | Docker                        | Docker                       |
| Install extra          | `--extra llm-mlx`            | `--extra llm-llamacpp`        | `--extra llm-llamacpp-cpu`   |

Windows is not supported directly — use WSL2 with the Linux path.

## Quick start

### 1. Clone and install

```bash
git clone git@github.com:YOUR/LEX.git
cd LEX

# Pick ONE extra matching your platform:
uv sync --extra llm-mlx              # macOS Apple Silicon
uv sync --extra llm-llamacpp         # Linux + NVIDIA GPU
uv sync --extra llm-llamacpp-cpu     # Linux CPU-only
```

### 2. Start services

```bash
# Qdrant + Redis (all platforms).
docker compose up -d
```

### 3. Start the LLM server

**macOS Apple Silicon:**

```bash
uv run mlx_lm.server \
  --model mlx-community/gemma-4-E4B-it-4bit \
  --port 8080 --host 127.0.0.1
```

**Linux + NVIDIA GPU (llama.cpp):**

```bash
# Download a GGUF model first (example: Gemma-2 2B Q4_K_M, ~1.5 GB).
mkdir -p models
huggingface-cli download \
  bartowski/gemma-2-2b-it-GGUF \
  gemma-2-2b-it-Q4_K_M.gguf \
  --local-dir models

# Run the server. --n_gpu_layers -1 offloads everything to GPU; reduce if OOM.
uv run python -m llama_cpp.server \
  --model models/gemma-2-2b-it-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  --n_gpu_layers -1 \
  --chat_format gemma
```

Set `.env` accordingly:

```bash
# Mac
LEX_LLM__BASE_URL=http://localhost:8080/v1
LEX_LLM__MODEL=mlx-community/gemma-4-E4B-it-4bit
LEX_LLM__MAX_TOKENS=2048

# Linux
LEX_LLM__BASE_URL=http://localhost:8080/v1
LEX_LLM__MODEL=gemma-2-2b-it         # llama.cpp uses the model basename
LEX_LLM__MAX_TOKENS=2048
```

### 4. Ingest a directive

```bash
uv run lex ingest 32018L1972
```

### 5. Ask questions

```bash
uv run lex ask "What is a very high capacity network?"

# Or run the UI:
uv run lex ui   # opens http://localhost:8100
```

## Evaluation

The eval suite runs 30 gold-standard questions through the RAG pipeline
and scores each with DeepEval metrics. Takes ~30-45 min on a local
judge.

### Simple case — one model for both RAG and judging

Use the same LLM server for both. Scores won't be great with a small
model as judge (4B models produce invalid JSON frequently), but
everything works.

```bash
uv run pytest -m eval -v
```

### Split RAG (small) + judge (big)

This is the intended workflow for local dev. Small model for everyday
RAG; large model stood up temporarily for evals.

Terminal 1 — the RAG LLM:
```bash
uv run mlx_lm.server --model mlx-community/gemma-4-E4B-it-4bit --port 8080
```

Terminal 2 — the judge LLM:
```bash
# macOS example:
uv run mlx_lm.server --model mlx-community/gemma-4-31B-it-4bit --port 8081
# Linux example (llama.cpp):
uv run python -m llama_cpp.server \
  --model models/gemma-2-27b-it-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8081 --n_gpu_layers 40 --chat_format gemma
```

Terminal 3 — run eval with the judge configured:
```bash
export LEX_EVAL_JUDGE__BASE_URL=http://localhost:8081/v1
export LEX_EVAL_JUDGE__MODEL=gemma-2-27b-it
uv run pytest -m eval -v
```

Reports land in `tests/reports/eval-YYYYMMDD-HHMMSS.{csv,md}`.

### Remote judge (OpenAI, Anthropic, etc.)

Any OpenAI-protocol endpoint works. Point the judge config at a remote
service:

```bash
export LEX_EVAL_JUDGE__BASE_URL=https://api.openai.com/v1
export LEX_EVAL_JUDGE__MODEL=gpt-4o-mini
export LEX_EVAL_JUDGE__API_KEY=sk-...
uv run pytest -m eval -v
```

## Production deploy

The `docker-compose.cuda.yml` overlay adds a containerised llama.cpp
server with GPU access. Requires the NVIDIA Container Toolkit on the
host.

```bash
# Assumes NVIDIA Container Toolkit is installed.
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# Then run the API + worker as usual.
uv run lex serve
uv run lex worker
```

For cloud deploys, the same compose file works on any machine with
Docker + NVIDIA Container Toolkit.

## Repository layout

```
src/lex/
  commands.py      Pydantic typed commands (Ingest, Retrieve, Answer)
  engine.py        Dispatcher
  config.py        Settings from env
  sources.py       CELLAR fetcher (RDF → Formex XML)
  ingestion.py     Parser, chunker, embedder, Qdrant writer
  retrieval.py     Hybrid search + reranker
  generation.py    RAG prompt + streaming + citation extraction
  worker.py        Redis-queue consumer
  api.py           FastAPI surface
  ui.py            Chainlit chat UI
  cli.py           Typer CLI
  tracing.py       Langfuse shim

tests/
  gold_standard.json   30 curated Q/A pairs
  test_lex.py          Smoke tests + parametrised eval
  conftest.py          Writes CSV + Markdown reports

SPEC.md              Design doc (reference)
```

## Troubleshooting

**`Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.`** — Your judge model is too small to produce well-formed JSON. Use the split-RAG-and-judge workflow with a bigger judge, or point `LEX_EVAL_JUDGE__BASE_URL` at a remote API.

**`MPS backend out of memory`** — macOS only. Other processes are using Metal. Quit anything GPU-heavy, or drop `LEX_RERANKER__BATCH_SIZE=2` in `.env`.

**`llama-cpp-python` fails to install on Linux** — The prebuilt CUDA wheels are indexed via `pyproject.toml`'s `[tool.uv.sources]`. If uv still tries to build from source, it means your system lacks a compatible prebuilt. Easiest fix: install the CPU extra (`uv sync --extra llm-llamacpp-cpu`) first, then swap after verifying.

**`docker compose` not found** — On Ubuntu with the Ubuntu-shipped `docker.io`, the compose plugin isn't bundled. Install via the official binary release (see Docker's docs) or switch to Docker's own apt repo.
