# RAG Eval Kit

Automated evaluation for retrieval-augmented generation pipelines. Point it at a folder of documents and it generates a ground-truth Q&A dataset, runs your RAG pipeline against every question, and scores the results on faithfulness, answer relevance, and context recall — all from the command line.

## Why it exists

Every team building RAG ships it without a repeatable eval loop. Frameworks like RAGAS exist but require a pre-built dataset. RAG Eval Kit *generates* the dataset from your own docs so you can benchmark before you have real users — critical for enterprise AI adoption where explainability and correctness matter.

## Status

**M1 shipped.** The project skeleton is in place and installable from source.

What works today:

- Python package (`src` layout, `rag_eval_kit` namespace) installable via `pip install -e .`
- Dependencies pinned in `pyproject.toml`: `typer`, `rich`, `openai`, `anthropic`, `chromadb`, `pypdf`, `python-dotenv`
- CLI entry point `rag-eval` registered via `pyproject.toml` scripts
- `rag-eval --version` prints `rag-eval-kit 0.1.0`
- `rag-eval run <docs_dir>` accepts a path argument and prints a stub message (pipeline not yet implemented)
- MIT license, `.gitignore`, test scaffold with `tests/test_version.py`

M2 onward (document ingestion, dataset generation, scoring) is not yet implemented.

## Quickstart

### Install from source (current)

```bash
git clone https://github.com/<!-- TODO: add repo URL -->
cd rag-eval-kit
pip install -e .

rag-eval --version
# rag-eval-kit 0.1.0
```

### Install from PyPI (once published)

```bash
pip install rag-eval-kit

# Set your LLM API key
export OPENAI_API_KEY=sk-...

# Run the full eval pipeline on a folder of documents
rag-eval run ./docs
```

> **Windows + ChromaDB note:** If ChromaDB fails with a SQLite error on Python 3.10+, run `pip install pysqlite3-binary` as a workaround.

## Architecture

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Scaffold, README, CLI stub (`rag-eval --version`) | shipped |
| M2 | Document ingestion + ChromaDB vector store | planned |
| M3 | Synthetic Q&A dataset generation via LLM | planned |
| M4 | RAG pipeline runner (chunking, retrieval, answer generation) | planned |
| M5 | LLM-as-judge scoring (faithfulness, relevance, recall) | planned |
| M6 | Rich terminal report + `results.json` output | planned |

## License

MIT © 2024 Ritik
