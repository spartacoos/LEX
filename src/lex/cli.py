"""
Command-line interface.

Each subcommand is a thin wrapper: build a command object, hand it to
the Engine (or a handler), print the result. No business logic lives
here. If you find yourself writing logic in cli.py, it belongs in a
handler instead.
"""

from __future__ import annotations

import asyncio
import json
import sys

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


@app.command()
def smoke() -> None:
    """Sanity check: load config, build one of each command, print as JSON."""
    settings = get_settings()

    rprint("[bold green]LEX smoke test[/bold green]")
    rprint(f"  Qdrant URL:        {settings.qdrant.url}")
    rprint(f"  Redis URL:         {settings.redis.url}")
    rprint(f"  LLM base URL:      {settings.llm.base_url}")
    rprint(f"  LLM model:         {settings.llm.model}")
    rprint(f"  Embedding model:   {settings.embedding.model}")
    rprint(f"  Reranker model:    {settings.reranker.model}")
    rprint(f"  Collection (en):   {settings.collection_name('en')}")

    rprint("\n[bold]Sample commands (as JSON):[/bold]")
    for cmd in [
        IngestCmd(celex_id="32018L1972"),
        RetrieveCmd(query="What is a VHCN?"),
        AnswerCmd(query="Define electronic communications service"),
    ]:
        rprint(f"\n[cyan]{cmd.kind}[/cyan]:")
        rprint(json.dumps(cmd.model_dump(mode="json"), indent=2))


@app.command()
def services() -> None:
    """Ping Qdrant and Redis. Verifies the Docker stack is actually up."""
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

        # Bonus: check the LLM endpoint too, but don't make it fatal —
        # ingest and search don't need the LLM, so having it offline
        # is a legitimate state for this command to report.
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{settings.llm.base_url}/models")
                r.raise_for_status()
            rprint(f"[green]✓ LLM[/green] @ {settings.llm.base_url}")
        except Exception as e:
            rprint(f"[yellow]○ LLM[/yellow] @ {settings.llm.base_url} "
                   f"(not running — that's fine unless you need `lex ask`): {e}")

    asyncio.run(_check())


@app.command()
def ingest(
    celex_id: str = typer.Argument(..., help="CELEX ID, e.g. 32018L1972"),
    language: str = typer.Option("en", "--language", "-l", help="ISO 639-1 code"),
    source: str = typer.Option(
        "cellar", "--source", "-s",
        help="'cellar' (network) or 'local' (disk)"
    ),
    local_dir: str | None = typer.Option(
        None, "--local-dir", help="Directory for --source=local"
    ),
) -> None:
    """
    Fetch, parse, chunk, embed, and store a directive.

    Example:
      lex ingest 32018L1972
      lex ingest 32018L1972 --language fr
      lex ingest 32018L1972 --source local --local-dir ./fixtures
    """
    from .ingestion import build_ingest_deps, handle_ingest

    settings = get_settings()
    cmd = IngestCmd(
        celex_id=celex_id, language=language,
        source=source,  # type: ignore[arg-type]
    )

    async def _run() -> None:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), transient=True,
        ) as progress:
            task = progress.add_task(
                f"Ingesting {celex_id} ({language})...", total=None,
            )
            deps = build_ingest_deps(
                settings, source_kind=source, local_dir=local_dir,
            )
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


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query"),
    top_k: int = typer.Option(5, "--top-k", "-k",
                              help="Number of results after reranking"),
    language: str = typer.Option("en", "--language", "-l", help="ISO 639-1 code"),
    celex_id: str | None = typer.Option(
        None, "--celex-id", "-c",
        help="Restrict to a specific directive"
    ),
    article: str | None = typer.Option(
        None, "--article", "-a",
        help="Restrict to a specific article number"
    ),
) -> None:
    """
    Hybrid search + rerank. Returns top-K chunks.

    Example:
      lex search "what is a very high capacity network"
      lex search "market analysis procedure" --celex-id 32018L1972
      lex search "penalties" --article 29 --top-k 3
    """
    from .retrieval import build_retrieve_deps, handle_retrieve

    settings = get_settings()
    cmd = RetrieveCmd(
        query=query,
        top_k=top_k,
        filters=RetrieveFilter(
            language=language, celex_id=celex_id, article=article,
        ),
    )

    async def _run() -> None:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), transient=True,
        ) as progress:
            task = progress.add_task("Loading models + searching...", total=None)
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


@app.command()
def ask(
    query: str = typer.Argument(..., help="Natural-language question"),
    language: str = typer.Option("en", "--language", "-l", help="ISO 639-1 code"),
    celex_id: str | None = typer.Option(
        None, "--celex-id", "-c",
        help="Restrict to a specific directive"
    ),
    article: str | None = typer.Option(
        None, "--article", "-a",
        help="Restrict to a specific article number"
    ),
) -> None:
    """
    Full RAG: retrieve, generate, cite. Streams tokens as the LLM produces them.

    Requires the LLM server to be running:
      uv run mlx_lm.server --model mlx-community/gemma-4-E4B-it-4bit --port 8080

    Example:
      lex ask "What is a very high capacity network?"
      lex ask "Who may impose penalties?" --article 29
      lex ask "Define electronic communications service"
    """
    from .generation import LLMUnreachable, build_answer_deps, handle_answer

    settings = get_settings()
    cmd = AnswerCmd(
        query=query,
        filters=RetrieveFilter(
            language=language, celex_id=celex_id, article=article,
        ),
        stream=True,
    )

    async def _run() -> None:
        # Phase 1: load models + verify LLM is reachable. Show a spinner
        # during the slow parts.
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), transient=True,
        ) as progress:
            task = progress.add_task("Loading models + connecting to LLM...",
                                     total=None)
            try:
                deps = await build_answer_deps(settings)
            except LLMUnreachable as e:
                progress.remove_task(task)
                rprint(f"[red]✗[/red] {e}")
                raise typer.Exit(code=1)
            progress.remove_task(task)

        # Phase 2: stream the answer directly to stdout.
        rprint(f"[bold]Q:[/bold] {query}\n")
        rprint("[bold]A:[/bold] ", end="")

        # We flush each token so users see the LLM thinking in real time.
        def _on_token(tok: str) -> None:
            sys.stdout.write(tok)
            sys.stdout.flush()

        try:
            result = await handle_answer(cmd, deps, on_token=_on_token)
        finally:
            await deps.retrieve.qdrant.close()

        # Trailing newline after the streamed answer.
        print()

        # Phase 3: citation summary.
        if result.citations:
            # Deduplicate by chunk_id — many markers may point at the
            # same source, and showing the source card once is enough.
            seen_chunk_ids: set[str] = set()
            rprint("\n[bold]Sources:[/bold]")
            for c in result.citations:
                if c.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(c.chunk_id)
                if c.article and c.paragraph:
                    ref = f"Article {c.article}({c.paragraph})"
                elif c.article:
                    ref = f"Article {c.article}"
                else:
                    ref = c.chunk_id
                rprint(f"  [cyan]{ref}[/cyan]  [dim]({c.chunk_id})[/dim]")
        else:
            rprint("\n[yellow]No citations emitted.[/yellow]")

    asyncio.run(_run())

@app.command()
def worker() -> None:
    """
    Run the ingestion worker. Consumes IngestCmds from the Redis queue.

    Start this in a separate terminal before using the API's /ingest
    endpoint. Ctrl-C to stop (any in-flight job finishes first).
    """
    from .worker import run_worker
    asyncio.run(run_worker())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload on file changes (dev)"),) -> None:
    """
    Run the FastAPI app with uvicorn.

    By default binds to 127.0.0.1 (local only). Use --host 0.0.0.0
    to expose on the network.
    """
    import uvicorn
    uvicorn.run(
        "lex.api:app",
        host=host, port=port, reload=reload,
    )

@app.command()
def ui(
    port: int = typer.Option(8100, help="Port for the Chainlit server"),
    host: str = typer.Option("127.0.0.1", help="Bind address"),) -> None:
    """
    Launch the Chainlit chat UI.

    By default serves on http://127.0.0.1:8100. The UI loads models
    in-process, so you need:
      - Qdrant + Redis running (docker compose up -d)
      - LLM server running (uv run mlx_lm.server ...)
      - A worker if you want to add directives from the UI (lex worker)
    """
    import subprocess
    import sys
    from pathlib import Path

    # Chainlit ships its own CLI; simplest is to shell out to it.
    # `--headless` prevents browser auto-open when running under an
    # orchestrator. Locally, users see the URL in the terminal.
    ui_file = Path(__file__).parent / "ui.py"
    cmd = [
        sys.executable, "-m", "chainlit", "run", str(ui_file),
        "--host", host,
        "--port", str(port),
    ]
    # `chainlit run` takes over stdin/stdout; we don't need to capture.
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    app()
