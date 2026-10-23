from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rag_eval_kit import __version__

app = typer.Typer()


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rag-eval-kit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)
) -> None:
    """RAG Eval Kit — automated evaluation for retrieval-augmented generation."""


def _print_report(scores: list, min_score: float | None) -> bool:
    """Print a rich table of per-question scores and per-metric averages. Returns True if passed."""
    console = Console()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Question", max_width=50, no_wrap=True)
    table.add_column("Faithfulness", justify="center")
    table.add_column("Answer Relevance", justify="center")
    table.add_column("Context Recall", justify="center")

    def _colour(v: float) -> str:
        if v >= 0.7:
            return f"[green]{v:.2f}[/green]"
        elif v >= 0.4:
            return f"[yellow]{v:.2f}[/yellow]"
        else:
            return f"[red]{v:.2f}[/red]"

    for s in scores:
        table.add_row(
            s.question,
            _colour(s.faithfulness),
            _colour(s.answer_relevance),
            _colour(s.context_recall),
        )

    if scores:
        avg_f = sum(s.faithfulness for s in scores) / len(scores)
        avg_ar = sum(s.answer_relevance for s in scores) / len(scores)
        avg_cr = sum(s.context_recall for s in scores) / len(scores)
    else:
        avg_f = avg_ar = avg_cr = 0.0

    table.add_row(
        "[bold]AVERAGE[/bold]",
        f"[bold]{avg_f:.2f}[/bold]",
        f"[bold]{avg_ar:.2f}[/bold]",
        f"[bold]{avg_cr:.2f}[/bold]",
    )

    console.print(table, highlight=False)

    passed = min_score is None or all(v >= min_score for v in (avg_f, avg_ar, avg_cr))
    if passed:
        if min_score is not None:
            console.print(f"[bold green]PASS[/bold green]  (all metrics ≥ {min_score:.2f})")
        else:
            console.print("[bold green]PASS[/bold green]")
    else:
        console.print(f"[bold red]FAIL[/bold red]  (one or more metrics < {min_score:.2f})")

    return passed


@app.command()
def run(
    docs_dir: str = typer.Argument(..., help="Path to folder with .pdf/.txt/.md files"),
    n: int = typer.Option(20, "--n", help="Q&A pairs to generate"),
    difficulty: str = typer.Option("factual", help="factual or multi-hop"),
    llm: str = typer.Option("openai", help="openai or anthropic"),
    top_k: int = typer.Option(4, "--top-k", help="Chunks retrieved per question"),
    chunk_size: int = typer.Option(512, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(64, help="Overlap between chunks"),
    collection: str = typer.Option("rag_eval", help="ChromaDB collection name"),
    db_path: str = typer.Option(".chroma", help="ChromaDB persist directory"),
    dataset_path: str = typer.Option("dataset.json", help="Where to save generated Q&A dataset"),
    output: str = typer.Option("results.json", help="Where to save scored results"),
    min_score: float | None = typer.Option(None, "--min-score", help="Fail if any metric avg < threshold (0.0–1.0)"),
) -> None:
    """Run the full eval pipeline on DOCS_DIR."""
    from pathlib import Path

    from rag_eval_kit.dataset import DatasetGenerator
    from rag_eval_kit.ingestion import DocumentLoader, VectorStore
    from rag_eval_kit.judge import LLMJudge
    from rag_eval_kit.pipeline import RAGPipeline

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        typer.echo(f"Error: directory '{docs_dir}' does not exist.", err=True)
        raise typer.Exit(1)

    if difficulty not in ("factual", "multi-hop"):
        raise typer.BadParameter("difficulty must be 'factual' or 'multi-hop'", param_hint="--difficulty")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        # Phase 1: Ingest
        ingest_task = progress.add_task("Ingesting documents…", total=None)
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        supported = {".pdf", ".txt", ".md"}
        files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]
        if not files:
            typer.echo("No .pdf, .txt, or .md files found.")
            raise typer.Exit(0)
        all_chunks = []
        for file in files:
            text = loader._load_file(file)
            chunks = loader._split(text, str(file))
            all_chunks.extend(chunks)
        store = VectorStore(collection_name=collection, persist_directory=db_path)
        store.add_chunks(all_chunks, progress=progress)
        progress.update(ingest_task, description=f"Ingested {len(all_chunks)} chunk(s) from {len(files)} file(s)")
        progress.stop_task(ingest_task)

        # Phase 2: Generate
        gen_task = progress.add_task(f"Generating {n} Q&A pairs…", total=n)
        generator = DatasetGenerator(llm=llm)
        import random
        items = []
        for _ in range(n):
            if difficulty == "factual":
                chunk = random.choice(all_chunks)
                item = generator._generate_one_factual(chunk)
            else:
                chunk_a, chunk_b = random.sample(all_chunks, 2)
                item = generator._generate_one_multihop(chunk_a, chunk_b)
            items.append(item)
            progress.advance(gen_task, 1)
        DatasetGenerator.save(items, dataset_path)

        # Phase 3: Pipeline
        pipeline_task = progress.add_task("Running RAG pipeline…", total=None)
        pipeline = RAGPipeline(store, llm=llm, top_k=top_k)
        results = pipeline.run(items)
        progress.stop_task(pipeline_task)

        # Phase 4: Judge
        judge_task = progress.add_task("Scoring with LLM judge…", total=None)
        judge = LLMJudge(llm=llm)
        scores = judge.score_all(results)
        LLMJudge.save(scores, output)
        progress.stop_task(judge_task)

    typer.echo(f"Results saved to {output}")

    passed = _print_report(scores, min_score)
    if not passed:
        raise typer.Exit(1)


@app.command()
def ingest(
    docs_dir: str = typer.Argument(..., help="Folder with .pdf/.txt/.md files"),
    chunk_size: int = typer.Option(512, help="Tokens per chunk"),
    chunk_overlap: int = typer.Option(64, help="Overlap between chunks"),
    collection: str = typer.Option("rag_eval", help="ChromaDB collection name"),
    db_path: str = typer.Option(".chroma", help="ChromaDB persist directory"),
) -> None:
    """Ingest DOCS_DIR into the local ChromaDB vector store."""
    from pathlib import Path

    from rag_eval_kit.ingestion import DocumentLoader, VectorStore

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        typer.echo(f"Error: directory '{docs_dir}' does not exist.", err=True)
        raise typer.Exit(1)

    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    supported = {".pdf", ".txt", ".md"}
    files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        typer.echo("No .pdf, .txt, or .md files found.")
        raise typer.Exit(0)

    all_chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        load_task = progress.add_task("Loading files…", total=len(files))
        for file in files:
            progress.update(load_task, description=f"Loading {file.name}…")
            text = loader._load_file(file)
            chunks = loader._split(text, str(file))
            all_chunks.extend(chunks)
            progress.advance(load_task, 1)

        progress.update(load_task, description=f"Loaded {len(files)} file(s), {len(all_chunks)} chunk(s)")

        if all_chunks:
            store = VectorStore(
                collection_name=collection,
                persist_directory=db_path,
            )
            store.add_chunks(all_chunks, progress=progress)

    typer.echo(
        f"Ingested {len(all_chunks)} chunk(s) from {len(files)} file(s) into '{collection}' at '{db_path}'."
    )


@app.command()
def generate(
    docs_dir: str = typer.Argument(..., help="Folder with .pdf/.txt/.md files"),
    n: int = typer.Option(20, "--n", help="Number of Q&A pairs to generate"),
    difficulty: str = typer.Option("factual", help="factual or multi-hop"),
    llm: str = typer.Option("openai", help="openai or anthropic"),
    chunk_size: int = typer.Option(512, help="Chunk size (chars)"),
    chunk_overlap: int = typer.Option(64, help="Chunk overlap (chars)"),
    output: str = typer.Option("dataset.json", help="Output path for dataset"),
) -> None:
    """Generate a synthetic Q&A dataset from DOCS_DIR and save to OUTPUT."""
    from pathlib import Path

    from rag_eval_kit.dataset import DatasetGenerator
    from rag_eval_kit.ingestion import DocumentLoader

    if difficulty not in ("factual", "multi-hop"):
        raise typer.BadParameter("difficulty must be 'factual' or 'multi-hop'", param_hint="--difficulty")

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        typer.echo(f"Error: directory '{docs_dir}' does not exist.", err=True)
        raise typer.Exit(1)

    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = loader.load_directory(docs_path)

    if not chunks:
        typer.echo("No chunks loaded — check that DOCS_DIR contains .pdf/.txt/.md files.")
        raise typer.Exit(1)

    generator = DatasetGenerator(llm=llm)
    items = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"Generating {n} Q&A pairs…", total=n)
        # Patch generate to advance progress per item
        for i in range(n):
            if difficulty == "factual":
                import random
                chunk = random.choice(chunks)
                item = generator._generate_one_factual(chunk)
            else:
                import random
                chunk_a, chunk_b = random.sample(chunks, 2)
                item = generator._generate_one_multihop(chunk_a, chunk_b)
            items.append(item)
            progress.advance(task, 1)

    DatasetGenerator.save(items, output)
    typer.echo(f"Generated {len(items)} Q&A pairs → {output}")
