"""
Command-line interface for LEX.

Subcommands
-----------
smoke         Sanity check: config, commands, service connectivity.
services      Ping Qdrant, Redis, and the LLM endpoint.
config        Interactive wizard — writes .env from a profile.
serve-llm     Download (if needed) and start the local LLM server.
ingest        Fetch, parse, chunk, embed, and store a directive.
search        Hybrid search + rerank.
ask           Full RAG with streaming output.
serve         Start the FastAPI server.
worker        Start the Redis ingestion worker.
ui            Start the Chainlit chat UI.

Design rule: no business logic lives here.  Each command builds a typed
Command object (or calls a factory) and delegates.  If you find yourself
writing logic in cli.py, move it to the relevant handler.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .commands import AnswerCmd, IngestCmd, RetrieveCmd, RetrieveFilter
from .config import get_settings

app = typer.Typer(
    name="lex",
    help="LEX — natural-language Q&A over EU directives.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

@app.command()
def smoke() -> None:
    """Sanity check: load config, build sample commands, print as JSON."""
    settings = get_settings()
    rprint("[bold green]LEX smoke test[/bold green]")
    rprint(f"  Profile:           {settings.profile or '(none)'}")
    rprint(f"  LLM backend:       {settings.llm.backend}")
    rprint(f"  LLM base URL:      {settings.llm.base_url}")
    rprint(f"  LLM model:         {settings.llm.model}")
    rprint(f"  Embedding model:   {settings.embedding.model}")
    rprint(f"  Embedding device:  {settings.embedding.device}")
    rprint(f"  Reranker device:   {settings.reranker.device}")
    rprint(f"  Qdrant URL:        {settings.qdrant.url}")
    rprint(f"  Redis URL:         {settings.redis.url}")
    rprint(f"  Collection (en):   {settings.collection_name('en')}")
    rprint("\n[bold]Sample commands (as JSON):[/bold]")
    for cmd in [
        IngestCmd(celex_id="32018L1972"),
        RetrieveCmd(query="What is a VHCN?"),
        AnswerCmd(query="Define electronic communications service"),
    ]:
        rprint(f"\n[cyan]{cmd.kind}[/cyan]:")
        rprint(json.dumps(cmd.model_dump(mode="json"), indent=2))


# ---------------------------------------------------------------------------
# services
# ---------------------------------------------------------------------------

@app.command()
def services() -> None:
    """Ping Qdrant, Redis, the LLM endpoint, and the model server."""
    settings = get_settings()

    async def _check() -> None:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{settings.qdrant.url}/")
                rprint(f"[green]✓ Qdrant[/green]: {r.json().get('title', 'ok')}")
            except Exception as e:
                rprint(f"[red]✗ Qdrant[/red]: {e}")

        import redis.asyncio as aioredis
        try:
            r = aioredis.from_url(settings.redis.url)
            pong = await r.ping()
            rprint(f"[green]✓ Redis[/green]: {pong}")
            await r.aclose()
        except Exception as e:
            rprint(f"[red]✗ Redis[/red]: {e}")

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{settings.llm.base_url}/models")
                r.raise_for_status()
            rprint(f"[green]✓ LLM[/green] @ {settings.llm.base_url} ({settings.llm.model})")
        except Exception as e:
            rprint(
                f"[yellow]○ LLM[/yellow] @ {settings.llm.base_url} "
                f"(not running — run `lex serve-llm`): {e}"
            )

        try:
            from .model_server import ModelClient
            client = ModelClient(settings.model_socket_path())
            if await client.ping():
                rprint(f"[green]✓ Model server[/green] @ {settings.model_socket_path()}")
            else:
                rprint(
                    f"[yellow]○ Model server[/yellow] not running "
                    f"(run `lex serve-models` for faster CLI)"
                )
        except Exception as e:
            rprint(f"[yellow]○ Model server[/yellow]: {e}")

    asyncio.run(_check())

# ---------------------------------------------------------------------------
# config  (interactive wizard)
# ---------------------------------------------------------------------------

@app.command(name="config")
def config_wizard(
    profile: str | None = typer.Option(
        None, "--profile", "-p",
        help="Profile name to apply directly (skips interactive prompt)",
    ),
    judge_profile: str | None = typer.Option(
        None, "--judge", "-j",
        help="Separate profile for the eval judge LLM (optional)",
    ),
    list_profiles: bool = typer.Option(
        False, "--list", "-l", help="List available profiles and exit",
    ),
) -> None:
    """
    Interactive wizard — choose a profile and write .env.

    Profiles encode hardware/model combinations.  Individual settings
    can still be overridden in .env after running this wizard.
    """
    from .profile import list_profiles as _list, profile_to_env

    available = _list()

    if list_profiles:
        rprint("[bold]Available profiles:[/bold]")
        for p in available:
            rprint(f"  [cyan]{p}[/cyan]")
        raise typer.Exit()

    if not available:
        rprint("[red]No profiles found in profiles/ directory.[/red]")
        raise typer.Exit(1)

    # ---- Choose RAG profile -------------------------------------------
    if profile is None:
        rprint("\n[bold]Available profiles:[/bold]")
        for i, p in enumerate(available, 1):
            rprint(f"  [{i}] [cyan]{p}[/cyan]")
        rprint()
        choice = typer.prompt(
            f"Choose RAG model profile (1-{len(available)})",
            default="1",
        )
        try:
            idx = int(choice) - 1
            profile = available[idx]
        except (ValueError, IndexError):
            rprint(f"[red]Invalid choice.[/red]")
            raise typer.Exit(1)

    # ---- Optional judge profile ---------------------------------------
    if judge_profile is None:
        rprint(
            "\n[dim]Optionally choose a separate (usually larger) model "
            "for eval judging.\nPress Enter to use the same model as RAG.[/dim]"
        )
        for i, p in enumerate(available, 1):
            rprint(f"  [{i}] [cyan]{p}[/cyan]")
        rprint(f"  [Enter] Same as RAG ({profile})")
        choice = typer.prompt("Judge profile", default="")
        if choice.strip():
            try:
                idx = int(choice) - 1
                judge_profile = available[idx]
            except (ValueError, IndexError):
                rprint("[yellow]Invalid choice — using same model as RAG.[/yellow]")
                judge_profile = None

    # ---- Build .env block --------------------------------------------
    rag_env = profile_to_env(profile)
    lines: list[str] = [
        "# Generated by `lex config`.",
        f"# RAG profile: {profile}",
        "# Edit individual values here; they override the profile.",
        "",
        f"LEX_PROFILE={profile}",
        "",
        "# --- RAG LLM ---",
    ]
    for k, v in sorted(rag_env.items()):
        lines.append(f"{k}={v}")

    if judge_profile and judge_profile != profile:
        judge_env = profile_to_env(judge_profile)
        # Map LLM keys → EVAL_JUDGE keys
        _key_map = {
            "LEX_LLM__BASE_URL": "LEX_EVAL_JUDGE__BASE_URL",
            "LEX_LLM__MODEL":    "LEX_EVAL_JUDGE__MODEL",
            "LEX_LLM__API_KEY":  "LEX_EVAL_JUDGE__API_KEY",
        }
        lines += ["", "# --- Eval judge (separate model) ---",
                  f"# Judge profile: {judge_profile}"]
        for src_k, dst_k in _key_map.items():
            if src_k in judge_env:
                lines.append(f"{dst_k}={judge_env[src_k]}")
        judge_raw = __import__("lex.profile", fromlist=["load_profile"]).load_profile(judge_profile)
        judge_ctx = judge_raw.get("llm", {}).get("ctx_size", 32768)
        judge_tok = judge_raw.get("llm", {}).get("max_tokens", 4096)
        lines += [
            f"LEX_EVAL_JUDGE__MAX_TOKENS={judge_tok}",
            "LEX_EVAL_JUDGE__TIMEOUT_S=3600.0",
        ]

    env_content = "\n".join(lines) + "\n"

    env_path = Path(".env")
    if env_path.exists():
        overwrite = typer.confirm(
            f"\n.env already exists. Overwrite?", default=False
        )
        if not overwrite:
            rprint("[yellow]Aborted — .env not changed.[/yellow]")
            raise typer.Exit()

    env_path.write_text(env_content)
    rprint(f"\n[green]✓[/green] Wrote [bold]{env_path}[/bold] with profile [cyan]{profile}[/cyan].")

    if settings_need_llm_server(profile):
        rprint(
            f"\n[bold]Next step:[/bold] start the LLM server:\n"
            f"  [cyan]uv run lex serve-llm[/cyan]"
        )
    rprint(
        "Then: [cyan]docker compose up -d && "
        "uv run lex ingest 32018L1972 && "
        "uv run lex ui[/cyan]"
    )


def settings_need_llm_server(profile_name: str) -> bool:
    """True when the profile requires a local server (not remote)."""
    try:
        from .profile import load_profile
        p = load_profile(profile_name)
        return p.get("llm", {}).get("backend", "llamacpp") != "remote"
    except Exception:
        return True


# ---------------------------------------------------------------------------
# serve-llm  (download model if needed, then start the server)
# ---------------------------------------------------------------------------

@app.command(name="serve-llm")
def serve_llm(
    profile: str | None = typer.Option(
        None, "--profile", "-p",
        help="Override the active profile for LLM serving",
    ),
    port: int | None = typer.Option(None, "--port", help="Override port"),
    n_gpu_layers: int | None = typer.Option(
        None, "--gpu-layers", "-g",
        help="Override n_gpu_layers (-1 = all, 0 = CPU only)",
    ),
    ctx_size: int | None = typer.Option(
        None, "--ctx-size", "-c", help="Override context size"
    ),
) -> None:
    """
    Download the model (if needed) and start the local LLM server.

    Uses the active profile from .env, or the profile specified with
    --profile.  The server runs in the foreground; Ctrl-C to stop.

    Examples:
      lex serve-llm
      lex serve-llm --profile gemma4-31b-gpu --gpu-layers 20
      lex serve-llm --gpu-layers 0          # force CPU
    """
    import subprocess

    # If a profile override was given, temporarily inject it.
    if profile:
        import os
        os.environ["LEX_PROFILE"] = profile

    settings = get_settings()
    llm = settings.llm

    if llm.backend == "remote":
        rprint("[yellow]Active profile uses a remote LLM — nothing to start.[/yellow]")
        raise typer.Exit()

    # Apply CLI overrides.
    _n_gpu = n_gpu_layers if n_gpu_layers is not None else llm.n_gpu_layers
    _ctx   = ctx_size     if ctx_size     is not None else llm.ctx_size
    _port  = port         if port         is not None else llm.port

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ---- Download GGUF if needed (llamacpp only) ----------------------
    if llm.backend == "llamacpp":
        model_path = models_dir / llm.model_file
        if not model_path.exists():
            if not llm.hf_repo or not llm.model_file:
                rprint(
                    "[red]model_file or hf_repo not set in profile.\n"
                    "Download the GGUF manually into models/ and set "
                    "LEX_LLM__MODEL_FILE in .env.[/red]"
                )
                raise typer.Exit(1)

            rprint(
                f"[bold]Downloading[/bold] [cyan]{llm.model_file}[/cyan] "
                f"from [dim]{llm.hf_repo}[/dim] → [dim]{models_dir}/[/dim]"
            )
            subprocess.run(
                ["hf", "download", llm.hf_repo, llm.model_file,
                 "--local-dir", str(models_dir)],
                check=True,
            )
        else:
            rprint(f"[green]✓[/green] Model already at {model_path}")

        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(_port),
            "--n_gpu_layers", str(_n_gpu),
            "--n_ctx", str(_ctx),
        ]
        rprint(
            f"\n[bold]Starting llama.cpp server[/bold] on port {_port} "
            f"(gpu_layers={_n_gpu}, ctx={_ctx})\n"
            f"[dim]{' '.join(cmd)}[/dim]\n"
            f"Ctrl-C to stop.\n"
        )
        subprocess.run(cmd)

    elif llm.backend == "mlx":
        model_id = llm.hf_repo or llm.model
        cmd = [
            sys.executable, "-m", "mlx_lm.server",
            "--model", model_id,
            "--port", str(_port),
            "--host", "127.0.0.1",
        ]
        rprint(
            f"\n[bold]Starting MLX server[/bold] on port {_port} "
            f"(model={model_id})\n"
            f"[dim]{' '.join(cmd)}[/dim]\n"
            f"Ctrl-C to stop.\n"
        )
        subprocess.run(cmd)

    else:
        rprint(f"[red]Unknown backend: {llm.backend}[/red]")
        raise typer.Exit(1)


@app.command(name="serve-models")
def serve_models(
    embedding_device: str = typer.Option(
        "cpu", "--embedding-device",
        help="Device for BGE-M3 embedder: cpu | cuda | mps | auto",
    ),
    reranker_device: str = typer.Option(
        "cpu", "--reranker-device",
        help="Device for BGE reranker: cpu | cuda | mps | auto",
    ),
) -> None:
    """
    Start the model server daemon (BGE-M3 + reranker).

    Defaults to CPU for both models — safe regardless of what else is
    running on the GPU. Pass --reranker-device cuda explicitly if you
    have confirmed free VRAM after your LLM server has loaded.

      Terminal 1:  lex serve-llm
      Terminal 2:  lex serve-models
      Terminal 3:  lex search / lex ask   (instant, no model loading)
    """
    from .ingestion import BGEEmbedder
    from .retrieval import BGEReranker
    from .model_server import ModelServer

    settings = get_settings()
    rprint(
        f"[bold]Loading models[/bold] "
        f"(embedder={embedding_device}, reranker={reranker_device})..."
    )

    embedder = BGEEmbedder(
        model_name=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        device=embedding_device,
    )
    reranker = BGEReranker(
        model_name=settings.reranker.model,
        batch_size=settings.reranker.batch_size,
        device=reranker_device,
    )

    rprint(
        f"[green]✓[/green] Models loaded. "
        f"Serving on [cyan]{settings.model_socket_path()}[/cyan]\n"
        f"Ctrl-C to stop."
    )

    server = ModelServer(
        socket_path=settings.model_socket_path(),
        embedder=embedder,
        reranker=reranker,
    )
    asyncio.run(server.serve())

# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    celex_id: str = typer.Argument(..., help="CELEX ID, e.g. 32018L1972"),
    language: str = typer.Option("en", "--language", "-l"),
    source: str = typer.Option("cellar", "--source", "-s",
                               help="'cellar' or 'local'"),
    local_dir: str | None = typer.Option(None, "--local-dir"),
) -> None:
    """
    Fetch, parse, chunk, embed, and store a directive.

    Examples:
      lex ingest 32018L1972
      lex ingest 32018L1972 --language fr
      lex ingest 32018L1972 --source local --local-dir ./fixtures
    """
    from .ingestion import build_ingest_deps, handle_ingest

    settings = get_settings()
    cmd = IngestCmd(celex_id=celex_id, language=language,
                    source=source)  # type: ignore[arg-type]

    async def _run() -> None:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      transient=True) as progress:
            task = progress.add_task(
                f"Ingesting {celex_id} ({language})...", total=None,
            )
            deps = build_ingest_deps(settings, source_kind=source,
                                     local_dir=local_dir)
            try:
                result = await handle_ingest(cmd, deps)
            finally:
                await deps.qdrant.close()
                await deps.redis.aclose()
            progress.remove_task(task)

        if result.state == "done":
            rprint(
                f"[green]✓[/green] Ingested {result.celex_id} "
                f"({result.language}) — {result.chunks_written} chunks."
            )
        else:
            rprint(f"[red]✗[/red] Ingest failed: {result.error}")
            raise typer.Exit(code=1)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    language: str = typer.Option("en", "--language", "-l"),
    celex_id: str | None = typer.Option(None, "--celex-id", "-c"),
    article: str | None = typer.Option(None, "--article", "-a"),
) -> None:
    """
    Hybrid search + rerank. Returns top-K chunks.

    Examples:
      lex search "what is a very high capacity network"
      lex search "penalties" --article 29 --top-k 3
    """
    from .retrieval import build_retrieve_deps, handle_retrieve

    settings = get_settings()
    cmd = RetrieveCmd(
        query=query, top_k=top_k,
        filters=RetrieveFilter(language=language, celex_id=celex_id,
                               article=article),
    )

    async def _run() -> None:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      transient=True) as progress:
            task = progress.add_task("Searching...", total=None)
            deps = build_retrieve_deps(settings)
            try:
                result = await handle_retrieve(cmd, deps)
            finally:
                await deps.qdrant.close()
            progress.remove_task(task)

        if not result.chunks:
            rprint("[yellow]No results.[/yellow]")
            return

        rprint(f"\n[bold]Query:[/bold] {result.query}\n")
        table = Table(show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Where", style="cyan", no_wrap=True)
        table.add_column("Rerank", justify="right", style="green")
        table.add_column("Dense", justify="right", style="blue")
        table.add_column("Sparse", justify="right", style="magenta")
        table.add_column("Text")

        for i, rc in enumerate(result.chunks, start=1):
            c = rc.chunk
            if c.chunk_type == "recital":
                where = f"Recital {c.paragraph or '?'}"
            elif c.chunk_type == "annex":
                where = f"Annex {c.article or '?'}"
            else:
                where = f"Art. {c.article or '?'}"
                if c.paragraph:
                    where += f"({c.paragraph})"
            text = c.text if len(c.text) < 300 else c.text[:297] + "..."
            table.add_row(
                str(i), where,
                f"{rc.rerank_score:+.3f}",
                f"{rc.dense_score:.3f}",
                f"{rc.sparse_score:.3f}",
                text,
            )
        rprint(table)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@app.command()
def ask(
    query: str = typer.Argument(...),
    language: str = typer.Option("en", "--language", "-l"),
    celex_id: str | None = typer.Option(None, "--celex-id", "-c"),
    article: str | None = typer.Option(None, "--article", "-a"),
) -> None:
    """
    Full RAG: retrieve, generate, cite.  Streams tokens live.

    Requires the LLM server to be running:
      lex serve-llm

    Examples:
      lex ask "What is a very high capacity network?"
      lex ask "Who may impose penalties?" --article 29
    """
    from .generation import LLMUnreachable, build_answer_deps, handle_answer

    settings = get_settings()
    cmd = AnswerCmd(
        query=query,
        filters=RetrieveFilter(language=language, celex_id=celex_id,
                               article=article),
        stream=True,
    )

    async def _run() -> None:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      transient=True) as progress:
            task = progress.add_task(
                "Loading models + connecting to LLM...", total=None)
            try:
                deps = await build_answer_deps(settings)
            except LLMUnreachable as e:
                progress.remove_task(task)
                rprint(f"[red]✗[/red] {e}")
                rprint("[dim]Tip: run `lex serve-llm` in another terminal.[/dim]")
                raise typer.Exit(code=1)
            progress.remove_task(task)

        rprint(f"[bold]Q:[/bold] {query}\n")
        rprint("[bold]A:[/bold] ", end="")

        def _on_token(tok: str) -> None:
            sys.stdout.write(tok)
            sys.stdout.flush()

        try:
            result = await handle_answer(cmd, deps, on_token=_on_token)
        finally:
            await deps.retrieve.qdrant.close()

        print()

        if result.citations:
            seen: set[str] = set()
            rprint("\n[bold]Sources:[/bold]")
            for c in result.citations:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                ref = (
                    f"Article {c.article}({c.paragraph})" if c.paragraph and c.article
                    else f"Article {c.article}" if c.article
                    else c.chunk_id
                )
                rprint(f"  [cyan]{ref}[/cyan]  [dim]({c.chunk_id})[/dim]")
        else:
            rprint("\n[yellow]No citations emitted.[/yellow]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# serve / worker / ui
# ---------------------------------------------------------------------------

@app.command()
def worker() -> None:
    """Run the ingestion worker (consumes from Redis queue)."""
    from .worker import run_worker
    asyncio.run(run_worker())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run("lex.api:app", host=host, port=port, reload=reload)


@app.command()
def ui(
    port: int = typer.Option(8100),
    host: str = typer.Option("127.0.0.1"),
) -> None:
    """Launch the Chainlit chat UI (http://127.0.0.1:8100)."""
    import subprocess
    ui_file = Path(__file__).parent / "ui.py"
    subprocess.run(
        [sys.executable, "-m", "chainlit", "run", str(ui_file),
         "--host", host, "--port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
