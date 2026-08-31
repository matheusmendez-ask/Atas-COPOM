"""Unified Command-Line Interface (CLI) and Orchestrator for the COPOM RAG Lakehouse Pipeline.

Commands:
  ingest     - Fetch raw Copom minutes from BCB API to Bronze layer.
  transform  - Sanitize and semantically chunk Bronze data into Silver layer.
  index      - Generate vector embeddings and upsert Silver chunks into Gold (Qdrant).
  run-all    - Execute the full end-to-end lakehouse pipeline with Phoenix tracing.
  query      - Perform semantic search against the vectorstore (retrieval only).
  ask        - Answer a question from the indexed minutes, citing the passages used.
  evaluate   - Score retrieval and grounding against the golden set in evaluation/.
"""

import contextlib
import logging
import sys
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Reconfigure stdout/stderr for Windows UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import settings
from src.evaluation import load_golden_set, run_evaluation
from src.generation.answerer import (
    CopomAnswerer,
    GenerationUnavailableError,
    retrieval_span_attributes,
)
from src.ingestion.collector import BronzeCollector
from src.observability.tracer import set_span_attributes, tracer
from src.processing.chunker import AtaChunker
from src.vectorstore.qdrant_manager import QdrantManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("copom-pipeline")

app = typer.Typer(
    name="copom-lakehouse",
    help="Enterprise COPOM RAG Lakehouse & Data Observability Pipeline CLI",
    add_completion=False,
)
console = Console(legacy_windows=False)


@app.command()
def ingest(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Number of Copom minutes to fetch.")
    ] = settings.DEFAULT_INGEST_LIMIT,
) -> None:
    """Ingest raw Copom meeting minutes from BCB API to Bronze Layer (idempotent)."""
    settings.ensure_directories()
    console.print(
        Panel.fit(f"[bold cyan]🥉 Starting Bronze Ingestion (Limit: {limit})[/bold cyan]")
    )

    with tracer.span(
        "pipeline_bronze_ingestion",
        {"openinference.span.kind": "CHAIN", "requested_limit": limit},
    ) as span:
        start_time = time.perf_counter()
        collector = BronzeCollector()
        summary = collector.run(limit=limit)
        elapsed = time.perf_counter() - start_time

        if span:
            span.set_attribute("catalog_items", summary.total_catalog_items)
            span.set_attribute("new_ingested", summary.new_ingested)
            span.set_attribute("skipped_existing", summary.skipped_existing)
            span.set_attribute("failed", summary.failed)

    # Render results table
    table = Table(
        title="🥉 Bronze Layer Ingestion Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Catalog Items Found", str(summary.total_catalog_items))
    table.add_row("Newly Ingested (Written to Disk)", str(summary.new_ingested))
    table.add_row("Skipped (Hash Already Existed)", str(summary.skipped_existing))
    table.add_row("Failed / Incomplete", str(summary.failed))
    table.add_row("Execution Time", f"{elapsed:.2f}s")

    console.print(table)


@app.command()
def transform() -> None:
    """Sanitize, normalize, and semantically chunk Bronze documents into Silver Layer."""
    settings.ensure_directories()
    console.print(Panel.fit("[bold cyan]🥈 Starting Silver Layer Transformation[/bold cyan]"))

    collector = BronzeCollector()
    history = collector.load_all_records()
    # Bronze keeps every version ever ingested; Silver keys on doc_id alone, so
    # only the current version of each document may be promoted.
    records = collector.select_current_versions(history)
    superseded = len(history) - len(records)

    if not records:
        console.print("[yellow]No Bronze records found on disk. Run 'ingest' first.[/yellow]")
        return

    with tracer.span(
        "pipeline_silver_transformation",
        {
            "openinference.span.kind": "CHAIN",
            "bronze_records_count": len(records),
            "superseded_versions": superseded,
        },
    ) as span:
        start_time = time.perf_counter()
        chunker = AtaChunker()
        silver_docs = chunker.process_all_bronze(records)
        elapsed = time.perf_counter() - start_time

        total_chunks = sum(len(doc.chunks) for doc in silver_docs)
        total_tokens = sum(doc.total_tokens for doc in silver_docs)

        if span:
            span.set_attribute("silver_docs_count", len(silver_docs))
            span.set_attribute("total_chunks_produced", total_chunks)
            span.set_attribute("total_tokens_processed", total_tokens)

    # Render results table
    table = Table(
        title="🥈 Silver Layer Processing Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bronze Documents Promoted (Current Version)", str(len(records)))
    table.add_row("Superseded Versions Skipped", str(superseded))
    table.add_row("Silver Documents Created", str(len(silver_docs)))
    table.add_row("Total Chunks Generated", str(total_chunks))
    table.add_row("Total Document Tokens", f"{total_tokens:,}")
    table.add_row("Execution Time", f"{elapsed:.2f}s")

    console.print(table)


@app.command()
def index(
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for embeddings & upserts.")
    ] = settings.EMBEDDING_BATCH_SIZE,
) -> None:
    """Generate dense embeddings and upsert Silver chunks into Gold Layer (Qdrant)."""
    settings.ensure_directories()
    console.print(
        Panel.fit(
            f"[bold cyan]🥇 Starting Gold Layer Vector Indexing (Batch Size: {batch_size})[/bold cyan]"
        )
    )

    chunker = AtaChunker()
    silver_docs = chunker.load_all_silver_documents()

    if not silver_docs:
        console.print("[yellow]No Silver documents found. Run 'transform' first.[/yellow]")
        return

    all_chunks = []
    for doc in silver_docs:
        all_chunks.extend(doc.chunks)

    with tracer.span(
        "pipeline_gold_indexing",
        {"openinference.span.kind": "EMBEDDING", "total_chunks": len(all_chunks)},
    ) as span:
        start_time = time.perf_counter()
        qdrant = QdrantManager()
        upsert_summary = qdrant.upsert_chunks(all_chunks, batch_size=batch_size)
        elapsed = time.perf_counter() - start_time

        if span:
            span.set_attribute("upserted_points", upsert_summary.upserted_points)
            span.set_attribute("batch_count", upsert_summary.batch_count)
            span.set_attribute("failed", upsert_summary.failed)

    # Render results table
    table = Table(
        title="🥇 Gold Layer Vector Indexing Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Collection Name", upsert_summary.collection_name)
    table.add_row("Total Chunks Submitted", str(upsert_summary.total_chunks))
    table.add_row("Indexed Vector Points (Idempotent)", str(upsert_summary.upserted_points))
    table.add_row("Batch Count", str(upsert_summary.batch_count))
    table.add_row("Failed Vectors", str(upsert_summary.failed))
    table.add_row("Total Vectors in Collection", str(qdrant.count_points()))
    table.add_row("Execution Time", f"{elapsed:.2f}s")

    console.print(table)


@app.command(name="run-all")
def run_all(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Number of Copom minutes to ingest.")
    ] = settings.DEFAULT_INGEST_LIMIT,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Embedding batch size.")
    ] = settings.EMBEDDING_BATCH_SIZE,
) -> None:
    """Execute complete end-to-end pipeline: Bronze -> Silver -> Gold with Arize Phoenix Tracing."""
    settings.ensure_directories()
    console.print(
        Panel.fit(
            "[bold green]🚀 Executing Full Enterprise Lakehouse Pipeline (Bronze -> Silver -> Gold)[/bold green]"
        )
    )

    with tracer.span(
        "copom_lakehouse_pipeline_full_run",
        {"openinference.span.kind": "CHAIN", "limit": limit, "batch_size": batch_size},
    ):
        # Step 1: Bronze Ingestion
        console.print("\n[bold]Step 1/3: Ingesting Raw Data (Bronze Layer)...[/bold]")
        ingest(limit=limit)

        # Step 2: Silver Transformation
        console.print("\n[bold]Step 2/3: Sanitizing and Chunking (Silver Layer)...[/bold]")
        transform()

        # Step 3: Gold Indexing
        console.print("\n[bold]Step 3/3: Embedding and Indexing to Qdrant (Gold Layer)...[/bold]")
        index(batch_size=batch_size)

    console.print("\n[bold green]✅ Pipeline execution completed successfully![/bold green]")
    console.print("[dim]Traces and metrics recorded in Arize Phoenix (http://localhost:6006)[/dim]")


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="Natural language query string for semantic search.")],
    limit: Annotated[int, typer.Option("--limit", "-k", help="Top-k matching results.")] = 3,
    year: Annotated[
        int | None, typer.Option("--year", "-y", help="Filter by publication year.")
    ] = None,
    meeting: Annotated[
        int | None, typer.Option("--meeting", "-m", help="Filter by meeting number.")
    ] = None,
) -> None:
    """Search the Gold Layer Vectorstore for relevant Copom passages."""
    console.print(Panel.fit(f"[bold cyan]🔍 Semantic Search Query: '{text}'[/bold cyan]"))

    with tracer.span(
        "copom_rag_retrieval", {"query": text, "top_k": limit, "year": year, "meeting": meeting}
    ):
        start_time = time.perf_counter()
        qdrant = QdrantManager()
        results = qdrant.search(
            query=text,
            limit=limit,
            filter_year=year,
            filter_meeting=meeting,
        )
        elapsed = time.perf_counter() - start_time

    if not results:
        console.print("[yellow]No matching chunks found in Qdrant.[/yellow]")
        return

    console.print(f"[green]Found {len(results)} matching chunks in {elapsed:.3f}s:[/green]\n")

    for i, res in enumerate(results, start=1):
        score = res["score"]
        payload = res["payload"]
        chunk_text = payload.get("text", "").strip()
        snippet = (chunk_text[:350] + "...") if len(chunk_text) > 350 else chunk_text

        table = Table(title=f"Result #{i} (Cosine Score: {score:.4f})", show_header=False)
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Document ID", payload.get("doc_id", "N/A"))
        table.add_row("Chunk ID", payload.get("chunk_id", "N/A"))
        table.add_row(
            "Meeting", f"#{payload.get('nro_reuniao', 'N/A')} - {payload.get('titulo', '')}"
        )
        table.add_row("Publication Date", payload.get("data_publicacao", "N/A"))
        table.add_row(
            "Tokens / Chars",
            f"{payload.get('token_count', 0)} tokens / {payload.get('char_count', 0)} chars",
        )
        table.add_row("Chunk Text", snippet)

        console.print(table)
        console.print()


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Pergunta em linguagem natural sobre as atas.")],
    limit: Annotated[
        int, typer.Option("--limit", "-k", help="Trechos recuperados como contexto.")
    ] = 5,
    year: Annotated[
        int | None, typer.Option("--year", "-y", help="Filtrar por ano de publicação.")
    ] = None,
    meeting: Annotated[
        int | None, typer.Option("--meeting", "-m", help="Filtrar por número da reunião.")
    ] = None,
) -> None:
    """Answer a question from the indexed minutes, citing the passages used."""
    console.print(Panel.fit(f"[bold cyan]💬 Pergunta: '{question}'[/bold cyan]"))

    answerer = CopomAnswerer()

    # Nested spans mirror the RAG structure so Phoenix separates retrieval
    # latency from generation latency instead of showing one opaque block.
    with tracer.span(
        "copom_rag_ask", {"openinference.span.kind": "CHAIN", "input.value": question}
    ) as chain_span:
        start_time = time.perf_counter()

        with tracer.span(
            "copom_rag_retrieval",
            {"openinference.span.kind": "RETRIEVER", "top_k": limit},
        ) as retrieval_span:
            sources = answerer.retrieve(
                question, limit=limit, filter_year=year, filter_meeting=meeting
            )
            set_span_attributes(retrieval_span, retrieval_span_attributes(sources))

        if not sources:
            console.print("[yellow]Nenhum trecho encontrado. Rode 'index' primeiro.[/yellow]")
            return

        with tracer.span(
            "copom_rag_generation",
            {"openinference.span.kind": "LLM", "llm.provider": settings.LLM_BASE_URL},
        ) as gen:
            try:
                answer = answerer.generate(question, sources)
            except GenerationUnavailableError as err:
                console.print(f"\n[red]Geração indisponível:[/red] {err}")
                raise typer.Exit(code=1) from err
            set_span_attributes(gen, answer.llm_span_attributes())

        elapsed = time.perf_counter() - start_time
        set_span_attributes(chain_span, {"output.value": answer.text})

    console.print(Panel(answer.text, title="Resposta", border_style="green"))

    table = Table(title="Fontes citadas", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Reunião", style="white")
    table.add_column("Publicação", style="white")
    table.add_column("Score", style="green")
    for source in answer.sources:
        table.add_row(
            f"[{source.index}]",
            f"#{source.nro_reuniao}",
            source.data_publicacao,
            f"{source.score:.4f}",
        )
    console.print(table)
    console.print(
        f"[dim]{answer.model} · {answer.total_tokens} tokens · {elapsed:.2f}s · "
        "trace em http://localhost:6006[/dim]"
    )


@app.command()
def evaluate(
    limit: Annotated[
        int, typer.Option("--limit", "-k", help="Trechos recuperados por pergunta.")
    ] = 5,
    with_generation: Annotated[
        bool | None,
        typer.Option(
            "--with-generation/--retrieval-only",
            help="Avaliar também a resposta gerada. Padrão: só se houver LLM_API_KEY.",
        ),
    ] = None,
    delay: Annotated[
        float,
        typer.Option(
            "--delay",
            help="Segundos entre chamadas ao modelo. Free tiers costumam exigir 15-30s.",
        ),
    ] = 0.0,
) -> None:
    """Score retrieval and grounding against the golden set in evaluation/."""
    golden = load_golden_set()
    generate = bool(settings.LLM_API_KEY) if with_generation is None else with_generation
    if generate and not settings.LLM_API_KEY:
        console.print("[red]--with-generation exige LLM_API_KEY.[/red]")
        raise typer.Exit(code=1)

    corpus = golden.get("corpus", {})
    console.print(
        Panel.fit(
            f"[bold cyan]📏 Avaliação · {len(golden['questions'])} perguntas · "
            f"top-{limit}[/bold cyan]\n"
            f"[dim]gabarito escrito sobre {corpus.get('chunks', '?')} chunks de "
            f"{len(corpus.get('meetings', []))} reuniões[/dim]"
        )
    )

    with tracer.span(
        "copom_rag_evaluation",
        {"openinference.span.kind": "EVALUATOR", "questions": len(golden["questions"])},
    ):
        report = run_evaluation(
            CopomAnswerer(), golden, limit=limit, generate=generate, delay=delay
        )

    detail = Table(title="Por pergunta", show_header=True, header_style="bold magenta")
    detail.add_column("Pergunta", style="cyan")
    detail.add_column("Tipo", style="white")
    detail.add_column("Rank", style="white", justify="right")
    detail.add_column("Resultado", style="white")
    for result in report.results:
        if result.kind == "unanswerable":
            verdict = (
                "—" if result.refused is None else ("recusou ✓" if result.refused else "INVENTOU ✗")
            )
            rank = "—"
        else:
            verdict = "✓" if result.retrieval_ok else "não recuperado ✗"
            rank = str(result.hit_rank) if result.hit_rank else "—"
        detail.add_row(result.id, result.kind, rank, verdict)
    console.print(detail)

    summary = Table(title="Métricas", show_header=True, header_style="bold magenta")
    summary.add_column("Métrica", style="cyan")
    summary.add_column("Valor", style="green", justify="right")
    summary.add_row("hit@1", f"{report.hit_at(1):.0%}")
    summary.add_row("hit@3", f"{report.hit_at(3):.0%}")
    summary.add_row(f"hit@{limit}", f"{report.hit_at(limit):.0%}")
    summary.add_row("MRR", f"{report.mrr:.3f}")
    summary.add_row(
        "[dim]hit@1 de um recuperador aleatório[/dim]", f"{report.random_hit1_baseline:.1%}"
    )
    if report.provenance_accuracy is not None:
        summary.add_row("Reunião esperada recuperada", f"{report.provenance_accuracy:.0%}")
    if generate:
        for label, value in (
            ("Fatos esperados na resposta", report.facts_accuracy),
            ("Citações dentro da faixa", report.citation_validity),
            ("Recusa nas armadilhas", report.refusal_rate),
        ):
            if value is not None:
                summary.add_row(label, f"{value:.0%}")
    console.print(summary)

    if report.generation_failures:
        failed = report.generation_failures
        console.print(
            f"[yellow]{len(failed)} de {len(report.results)} perguntas ficaram sem nota de "
            f"geração (as métricas acima são só das que responderam).[/yellow]\n"
            f"[dim]primeiro erro: {failed[0].generation_error[:140]}[/dim]"
        )

    if not generate:
        console.print(
            "[dim]Só recuperação. Defina LLM_API_KEY (e use --with-generation) para "
            "avaliar fundamentação, citações e recusa.[/dim]"
        )


if __name__ == "__main__":
    app()
