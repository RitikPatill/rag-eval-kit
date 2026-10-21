from dotenv import load_dotenv

load_dotenv()

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

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


@app.command()
def run(docs_dir: str = typer.Argument(..., help="Path to folder containing .pdf/.txt/.md files")) -> None:
    """Run the full eval pipeline on DOCS_DIR."""
    typer.echo("Pipeline not yet implemented. Stay tuned.")


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
