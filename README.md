# RAG Eval Kit

Automated evaluation for retrieval-augmented generation pipelines. Point it at a folder of documents and it generates a ground-truth Q&A dataset, runs your RAG pipeline against every question, and scores the results on faithfulness, answer relevance, and context recall — all from the command line.

## Why it exists

Every team building RAG ships it without a repeatable eval loop. Frameworks like RAGAS exist but require a pre-built dataset. RAG Eval Kit *generates* the dataset from your own docs so you can benchmark before you have real users — critical for enterprise AI adoption where explainability and correctness matter.

## Status

**M2 shipped.** Document ingestion and local ChromaDB vector store are fully implemented.

What works today:

- Python package (`src` layout, `rag_eval_kit` namespace) installable via `pip install -e .`
- Dependencies pinned in `pyproject.toml`: `typer`, `rich`, `openai`, `anthropic`, `chromadb`, `pypdf`, `python-dotenv`
- CLI entry point `rag-eval` registered via `pyproject.toml` scripts
- `rag-eval --version` prints `rag-eval-kit 0.1.0`
- `rag-eval ingest <docs_dir>` reads `.pdf`, `.txt`, and `.md` files recursively, splits them into overlapping chunks, embeds with OpenAI `text-embedding-3-small`, and persists to a local ChromaDB collection — with a `rich` progress bar
- `rag-eval run <docs_dir>` accepts a path argument and prints a stub message (pipeline not yet implemented)
- `DocumentLoader` class handles file reading (pypdf for PDFs, plain text for .txt/.md) and sliding-window chunking with configurable size and overlap
- `VectorStore` class wraps ChromaDB with upsert-based idempotency; embeds in batches of 100
- MIT license, `.gitignore`, unit tests for chunking logic in `tests/test_ingestion.py`

M3 onward (dataset generation, scoring) is not yet implemented.

## Quickstart

### Install from source (current)

```bash
git clone https://github.com/<!-- TODO: add repo URL -->
cd rag-eval-kit
pip install -e .

rag-eval --version
# rag-eval-kit 0.1.0
```

### Ingest documents (M2)

```bash
# Set your OpenAI API key (or put it in a .env file)
export OPENAI_API_KEY=sk-...

# Ingest a folder of docs into the local vector store
rag-eval ingest ./docs

# Options
rag-eval ingest ./docs --chunk-size 256 --chunk-overlap 32 --collection my_eval --db-path .chroma
```

### Install from PyPI (once published)

```bash
pip install rag-eval-kit

# Set your LLM API key
export OPENAI_API_KEY=sk-...

# Run the full eval pipeline on a folder of documents
rag-eval run ./docs
```

## Architecture

### Components (as of M2)

```
docs/
  *.pdf  *.txt  *.md
       │
       ▼
 DocumentLoader          src/rag_eval_kit/ingestion.py
   • reads files (pypdf for PDF, plain text for .txt/.md)
   • sliding-window chunking (configurable size + overlap)
       │  List[str] chunks
       ▼
  VectorStore            src/rag_eval_kit/ingestion.py
   • embeds via OpenAI text-embedding-3-small (batches of 100)
   • upserts into a local ChromaDB collection (idempotent)
       │
       ▼
  .chroma/               persistent local vector store
```

The `rag-eval ingest` CLI command wires these two classes together with a `rich` progress bar.

### Milestone map

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Scaffold, README, CLI stub (`rag-eval --version`) | shipped |
| M2 | Document ingestion + ChromaDB vector store | shipped |
| M3 | Synthetic Q&A dataset generation via LLM | planned |
| M4 | RAG pipeline runner (chunking, retrieval, answer generation) | planned |
| M5 | LLM-as-judge scoring (faithfulness, relevance, recall) | planned |
| M6 | Rich terminal report + `results.json` output | planned |

## Development

```bash
pip install -e ".[dev]"   # installs pytest
pytest                    # runs chunking + ingestion unit tests
```

## License

MIT © 2024 Ritik
