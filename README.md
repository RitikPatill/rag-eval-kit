# RAG Eval Kit

Automated evaluation for retrieval-augmented generation pipelines. Point it at a folder of documents and it generates a ground-truth Q&A dataset, runs your RAG pipeline against every question, and scores the results on faithfulness, answer relevance, and context recall — all from the command line.

## Why it exists

Every team building RAG ships it without a repeatable eval loop. Frameworks like RAGAS exist but require a pre-built dataset. RAG Eval Kit *generates* the dataset from your own docs so you can benchmark before you have real users — critical for enterprise AI adoption where explainability and correctness matter.

## Status

**M4 shipped.** RAG pipeline and LLM-judge scoring are fully implemented.

What works today:

- Python package (`src` layout, `rag_eval_kit` namespace) installable via `pip install -e .`
- Dependencies pinned in `pyproject.toml`: `typer`, `rich`, `openai`, `anthropic`, `chromadb`, `pypdf`, `python-dotenv`
- CLI entry point `rag-eval` registered via `pyproject.toml` scripts
- `rag-eval --version` prints `rag-eval-kit 0.1.0`
- `rag-eval ingest <docs_dir>` reads `.pdf`, `.txt`, and `.md` files recursively, splits them into overlapping chunks, embeds with OpenAI `text-embedding-3-small`, and persists to a local ChromaDB collection — with a `rich` progress bar
- `rag-eval generate <docs_dir>` synthesizes a Q&A dataset from document chunks using an LLM, and saves it to `dataset.json`
- `rag-eval run <docs_dir>` accepts a path argument and prints a stub message (CLI wiring is M5)
- `DocumentLoader` class handles file reading (pypdf for PDFs, plain text for .txt/.md) and sliding-window chunking with configurable size and overlap
- `VectorStore` class wraps ChromaDB with upsert-based idempotency; embeds in batches of 100; exposes `.query(text, top_k)` for retrieval
- `DatasetGenerator` class samples chunks, calls an LLM (OpenAI or Anthropic) to produce Q&A pairs, and persists to `dataset.json`
- `RAGPipeline` class retrieves top-k chunks per question and generates answers via LLM (OpenAI or Anthropic)
- `LLMJudge` class scores each `(question, context, generated_answer, gold_answer)` on faithfulness, answer relevance, and context recall using an LLM-as-judge pattern; saves results to `results.json`
- MIT license, `.gitignore`, unit tests in `tests/`

M5 (CLI wiring for `rag-eval run`) is not yet implemented.

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

### Generate a synthetic Q&A dataset (M3)

```bash
# Generate 20 factual Q&A pairs from ./docs (default)
rag-eval generate ./docs

# Options
rag-eval generate ./docs --n 50 --difficulty multi-hop --llm anthropic --output my_dataset.json
```

Each generated item contains: `question`, `answer`, `source_chunk`, `source` (file path), and `difficulty`.
The dataset is saved as JSON and can be loaded with `DatasetGenerator.load("dataset.json")`.

### Run the RAG pipeline and score results (M4)

CLI wiring for `rag-eval run` lands in M5. Until then, use the Python API directly:

```python
from rag_eval_kit.dataset import DatasetGenerator
from rag_eval_kit.ingestion import VectorStore
from rag_eval_kit.pipeline import RAGPipeline
from rag_eval_kit.judge import LLMJudge

# Load the ground-truth dataset produced by `rag-eval generate`
items = DatasetGenerator.load("dataset.json")

# Connect to the vector store populated by `rag-eval ingest`
store = VectorStore()

# Retrieve + generate answers
pipeline = RAGPipeline(store, llm="openai", top_k=4)
results = pipeline.run(items)

# Score on faithfulness, answer relevance, and context recall
judge = LLMJudge(llm="openai")
scores = judge.score_all(results)
LLMJudge.save(scores, "results.json")
```

Each entry in `results.json` contains: `question`, `generated_answer`, `gold_answer`,
`faithfulness`, `answer_relevance`, and `context_recall` — all normalised to `0.0–1.0`.

### Install from PyPI (once published)

```bash
pip install rag-eval-kit

# Set your LLM API key
export OPENAI_API_KEY=sk-...

# Run the full eval pipeline on a folder of documents
rag-eval run ./docs
```

## Architecture

### Components (as of M4)

```
docs/
  *.pdf  *.txt  *.md
       │
       ▼
 DocumentLoader          src/rag_eval_kit/ingestion.py
   • reads files (pypdf for PDF, plain text for .txt/.md)
   • sliding-window chunking (configurable size + overlap)
       │  List[Chunk]
       ├──────────────────────────────────────┐
       ▼                                      ▼
  VectorStore            src/rag_eval_kit/ingestion.py
   • embeds via OpenAI text-embedding-3-small (batches of 100)
   • upserts into a local ChromaDB collection (idempotent)
       │
       ▼
  .chroma/               persistent local vector store

 DatasetGenerator        src/rag_eval_kit/dataset.py
   • samples random chunks
   • calls LLM (OpenAI GPT-4o-mini or Anthropic Claude) to produce Q&A pairs
   • supports factual (1 chunk) and multi-hop (2 chunks) difficulty
       │
       ▼
  dataset.json           machine-readable Q&A ground-truth dataset
       │  List[QAItem]
       ▼
  RAGPipeline            src/rag_eval_kit/pipeline.py
   • retrieves top-k chunks from VectorStore per question
   • generates an answer via LLM (OpenAI or Anthropic)
       │  List[RAGResult]
       ▼
  LLMJudge               src/rag_eval_kit/judge.py
   • scores each result on faithfulness, answer relevance, context recall
   • uses LLM-as-judge with structured JSON output (scores clamped 0–5 → 0.0–1.0)
       │
       ▼
  results.json           per-question scores (faithfulness, answer_relevance, context_recall)
```

The `rag-eval ingest` and `rag-eval generate` CLI commands wire these classes together with `rich` progress bars. Full `rag-eval run` pipeline CLI wiring is M5.

### Milestone map

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Scaffold, README, CLI stub (`rag-eval --version`) | shipped |
| M2 | Document ingestion + ChromaDB vector store | shipped |
| M3 | Synthetic Q&A dataset generation via LLM | shipped |
| M4 | RAG pipeline runner + LLM-judge scoring | shipped |
| M5 | CLI wiring for `rag-eval run` + rich terminal report | planned |
| M6 | Rich terminal report + `results.json` output | planned |

## Development

```bash
pip install -e ".[dev]"   # installs pytest
pytest                    # runs chunking + ingestion unit tests
```

## License

MIT © 2024 Ritik
