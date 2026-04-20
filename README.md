# LEX

Natural-language Q&A over EU directives. Local-first, reproducible,
incrementally deployable.

Ask _"What penalties are provided for?"_ — get an answer grounded in
the actual directive text, with citations back to the Article and
Paragraph.

---

## Quick start (5 commands)

```bash
# 1. Clone + install (pick the extra matching your platform)
git clone https://github.com/YOUR/LEX.git && cd LEX

uv sync --extra llm-mlx          # macOS Apple Silicon  ← recommended on Mac
uv sync --extra llm-llamacpp     # Linux + NVIDIA GPU
uv sync --extra llm-llamacpp-cpu # Linux CPU-only / WSL without GPU

# On Linux/WSL with CUDA, source this once per shell:
source scripts/env.sh

# 2. Choose your model + hardware profile (interactive wizard)
uv run lex config
# → writes .env with the right settings for your machine
# → run `lex config --list` to see all available profiles

# 3. Start Qdrant + Redis
docker compose up -d

# 4. Download model + start LLM server (skipped if profile=remote-*)
uv run lex serve-llm             # terminal 1 — keep running

# 5. In a new terminal: ingest a directive + open the chat UI
uv run lex ingest 32018L1972     # terminal 2
uv run lex ui                    # opens http://localhost:8100
```

That's it. The chat UI handles everything else.

---

## Platform matrix

| Platform | Profile | Extra |
|---|---|---|
| macOS Apple Silicon | `gemma4-e4b-mlx` | `--extra llm-mlx` |
| Linux + NVIDIA GPU  | `gemma4-e4b-gpu` | `--extra llm-llamacpp` |
| Linux CPU-only / WSL| `gemma4-e2b-cpu` | `--extra llm-llamacpp-cpu` |
| Any (remote API)    | `remote-openai`  | _(none)_ |

Windows native is not supported — use WSL2 with the Linux path.

---

## Available profiles

| Profile | Model | VRAM / RAM | Best for |
|---|---|---|---|
| `gemma4-e2b-cpu` | Gemma 4 E2B Q4 | CPU only | Any machine, slowest |
| `gemma4-e4b-gpu` | Gemma 4 E4B Q4 | ~6 GB VRAM | Default GPU recommendation |
| `gemma4-e4b-mlx` | Gemma 4 E4B 4bit | Metal (unified) | macOS M-series |
| `gemma4-31b-gpu` | Gemma 4 31B Q4 | 24+ GB VRAM | High-quality local judge |
| `qwen35-9b-gpu`  | Qwen 3.5 9B Q4 | ~8 GB VRAM | Strong reasoning alternative |
| `remote-openai`  | Any remote model | — | OpenAI / Together / Groq / Anthropic |

Profiles live in `profiles/*.yaml`. Adding a new model = adding a new
YAML file, zero Python changes.

### Switching profiles

```bash
# Re-run the wizard any time
uv run lex config

# Or set directly in .env:
LEX_PROFILE=qwen35-9b-gpu

# Override a single parameter without touching the profile:
LEX_LLM__N_GPU_LAYERS=10 uv run lex serve-llm   # partial GPU offload

# Split RAG model (small+fast) from eval judge (large+accurate):
uv run lex config --profile gemma4-e4b-gpu --judge gemma4-31b-gpu
```

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| Deps | uv | Deterministic, fast |
| Containers | docker compose | Reproducible dev/prod |
| Config | pydantic-settings + YAML profiles | Typed, env-backed, wizard-friendly |
| Types/commands | pydantic v2 | Discriminated unions, validation |
| XML parsing | lxml | XPath over Formex |
| Embeddings | BGE-M3 (transformers) | Dense + sparse, multilingual |
| Reranking | BGE-reranker-v2-m3 | Cross-encoder, strong on legal text |
| Vector store | Qdrant | Hybrid search, metadata filters |
| LLM (local) | llama-cpp-python / mlx-lm | OpenAI-protocol, GGUF, zero daemon |
| Queue/cache | Redis | Job state machine, result cache |
| API | FastAPI | Async, SSE streaming |
| UI | Chainlit | Streaming, inline citations |
| CLI | typer + rich | Subcommands, progress spinners |
| Eval | DeepEval + pytest | RAGAS-compatible metrics |

---

## CLI reference

```
lex config              Interactive wizard — choose profile, write .env
lex config --list       Show available profiles
lex serve-llm           Download model (if needed) + start local LLM server
lex serve-llm --gpu-layers 0   Force CPU (override profile)
lex services            Ping Qdrant, Redis, LLM endpoint
lex smoke               Verify config + print sample commands
lex ingest 32018L1972   Ingest the EECC directive
lex search "query"      Hybrid search + rerank
lex ask "question"      Full RAG with streaming output
lex serve               Start FastAPI server (port 8000)
lex worker              Start Redis ingestion worker
lex ui                  Start Chainlit UI (port 8100)
```

---

## Evaluation

The eval suite runs 30 gold-standard Q/A pairs through the full RAG
pipeline and scores each answer with DeepEval metrics. Takes 30–45 min
on a local judge; much faster with a remote judge.

```bash
# Simple: one model for both RAG and judging (scores will be lower)
uv run pytest -m eval -v

# Recommended: small model for RAG, big model for judging
# Terminal 1: RAG LLM already running on port 8080
# Terminal 2: judge LLM
uv run lex serve-llm --profile gemma4-31b-gpu --port 8081

# Terminal 3: run eval
export LEX_EVAL_JUDGE__BASE_URL=http://localhost:8081/v1
export LEX_EVAL_JUDGE__MODEL=gemma-4-31b-it
uv run pytest -m eval -v

# Or use a remote judge (fastest, no extra download):
export LEX_EVAL_JUDGE__BASE_URL=https://api.openai.com/v1
export LEX_EVAL_JUDGE__MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
uv run pytest -m eval -v
```

Reports land in `tests/reports/eval-YYYYMMDD-HHMMSS.{csv,md}`.

### Eval targets

| Metric | Target | What it measures |
|---|---|---|
| Context precision | > 0.80 | Retrieved chunks are relevant |
| Context recall | > 0.80 | Needed info is in retrieved set |
| Faithfulness | > 0.90 | Answer grounded in context |
| Answer relevancy | > 0.85 | Answer addresses the question |
| Citation correctness | > 0.90 | Cited articles match source chunks |

---

## Production deploy (GPU server)

```bash
# Assumes NVIDIA Container Toolkit installed
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# Run API + worker
uv run lex serve
uv run lex worker
```

The CUDA compose overlay reads `LEX_LLM__MODEL_FILE`, `LEX_LLM__PORT`,
`LEX_LLM__N_GPU_LAYERS`, and `LEX_LLM__CTX_SIZE` from your `.env`.

---

## Repository layout

```
profiles/                  Hardware/model profiles (YAML)
  gemma4-e4b-gpu.yaml
  gemma4-e4b-mlx.yaml
  gemma4-e2b-cpu.yaml
  gemma4-31b-gpu.yaml
  qwen35-9b-gpu.yaml
  remote-openai.yaml

src/lex/
  config.py          Settings (pydantic-settings, profile-aware)
  profile.py         YAML profile loader + env-var translator
  commands.py        All command + result types (system contract)
  engine.py          Dispatcher
  sources.py         CELLAR fetcher + local file adapter
  ingestion.py       Parse → chunk → embed → write
  retrieval.py       Hybrid search + rerank
  generation.py      RAG prompt + streaming + citation extraction
  worker.py          Redis queue consumer
  api.py             FastAPI surface (SSE streaming)
  cli.py             typer CLI (config wizard, serve-llm, etc.)
  ui.py              Chainlit chat UI
  tracing.py         Langfuse shim (no-op when unset)

tests/
  test_lex.py        Smoke tests + parametrised eval
  conftest.py        CSV/Markdown report writer
  gold_standard.json 30 curated Q/A pairs

scripts/
  env.sh             Linux CUDA LD_LIBRARY_PATH helper

docker-compose.yml        Qdrant + Redis
docker-compose.cuda.yml   GPU LLM overlay (production)
```

---

## Troubleshooting

**`LLM endpoint unreachable`** — Run `lex serve-llm` in a separate
terminal before `lex ask` or `lex ui`.

**`Profile 'x' not found`** — Run `lex config --list` to see available
profiles, or run `lex config` to re-run the wizard.

**`Evaluation LLM outputted an invalid JSON`** — Your judge model is
too small. Use a split-judge setup with Gemma 4 31B or a remote API.
See the Evaluation section above.

**`MPS backend out of memory`** (macOS) — Quit GPU-heavy apps, or set
`LEX_RERANKER__DEVICE=cpu` in `.env`.

**`libcudart.so.12: cannot open shared object file`** (Linux) — Run
`source scripts/env.sh` before any `uv run` command.

**`hf: command not found`** — The old `huggingface-cli` is deprecated.
Install `huggingface_hub[cli]`: `uv pip install huggingface_hub[cli]`.

**`llama-cpp-python` fails to install on Linux** — Try the CPU extra
first: `uv sync --extra llm-llamacpp-cpu`, verify it works, then
switch to `--extra llm-llamacpp` once you confirm your CUDA setup.

**`docker compose` not found** (Ubuntu with `docker.io`) — Install the
compose plugin:
```bash
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL \
  https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```
