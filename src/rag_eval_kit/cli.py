import typer
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
