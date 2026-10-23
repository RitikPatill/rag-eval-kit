# RAG Eval Kit

> Drop in a folder of docs, get an auto-generated Q&A benchmark, and score your RAG pipeline end-to-end in one command.

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

RAG Eval Kit automates the hardest part of building retrieval-augmented generation systems: knowing whether they actually work. Point it at a folder of `.pdf`, `.txt`, or `.md` files and it ingests the documents into a local ChromaDB vector store, uses an LLM to synthesize a ground-truth Q&A dataset directly from the source material, runs a configurable RAG pipeline against every generated question, and scores the results on three axes via an LLM-as-judge.

The three metrics are **faithfulness** (is the answer grounded in the retrieved context?), **answer relevance** (does it actually answer the question?), and **context recall** (did retrieval surface the right chunks?). Each score is normalised to 0.0–1.0 and written to `results.json`; a colour-coded `rich` table is printed to the terminal. Pass `--min-score` to gate CI on a threshold.

## Quickstart

```bash
git clone https://github.com/RitikPatill/rag-eval-kit.git
cd rag-eval-kit
pip install -e .

# Set your LLM key (or put it in a .env file)
export OPENAI_API_KEY=sk-...

# Run the full pipeline on a folder of documents
rag-eval run ./docs
```

Requires Python 3.10+. No external services — the vector store runs locally via ChromaDB.

## Usage

The main entry point is `rag-eval run`, which chains ingest → dataset generation → RAG pipeline → scoring in one shot:

```bash
# Defaults: 20 factual questions, OpenAI GPT-4o-mini, top-k 4
rag-eval run ./docs

# More control
rag-eval run ./docs --n 50 --difficulty multi-hop --llm anthropic --top-k 6 --min-score 0.7
```

The sub-commands are also available independently when you want to inspect intermediate outputs:

```bash
rag-eval ingest   ./docs                        # build the vector store only
rag-eval generate ./docs --n 30 --output qa.json  # generate a dataset only
```

`--min-score` causes the process to exit with code `1` if any per-metric average falls below the threshold — useful as a CI quality gate.

Example terminal output:

```
┌──────────────────────────────────────┬─────────────┬──────────────────┬────────────────┐
│ Question                             │ Faithfulness│ Answer Relevance │ Context Recall │
├──────────────────────────────────────┼─────────────┼──────────────────┼────────────────┤
│ What is the retention period for … ? │    0.80     │      0.60        │      1.00      │
│ Which regulation governs … ?         │    1.00     │      1.00        │      0.80      │
├──────────────────────────────────────┼─────────────┼──────────────────┼────────────────┤
│ AVERAGE                              │    0.90     │      0.80        │      0.90      │
└──────────────────────────────────────┴─────────────┴──────────────────┴────────────────┘

PASS  (all metrics ≥ 0.70)
```

## Architecture

```
docs/*.pdf / *.txt / *.md
        │
        ▼
  DocumentLoader ──► chunks ──► VectorStore (.chroma/)
                                      │
  DatasetGenerator ◄────────────────┘   (samples chunks → LLM → Q&A pairs)
        │ dataset.json
        ▼
  RAGPipeline  (top-k retrieval + LLM answer generation)
        │
        ▼
  LLMJudge ──► results.json + rich terminal table + exit code
```

LLM backend is pluggable: pass `--llm openai` (default, GPT-4o-mini) or `--llm anthropic` (Claude). Embeddings use OpenAI `text-embedding-3-small`.

## Project structure

```
rag-eval-kit/
├── src/rag_eval_kit/
│   ├── cli.py          # typer CLI — ingest / generate / run commands
│   ├── ingestion.py    # DocumentLoader (chunking) + VectorStore (ChromaDB)
│   ├── dataset.py      # DatasetGenerator — synthetic Q&A via LLM
│   ├── pipeline.py     # RAGPipeline — retrieval + answer generation
│   └── judge.py        # LLMJudge — faithfulness / relevance / recall scoring
├── tests/              # pytest unit tests for each module
├── pyproject.toml      # package metadata, pinned dependencies, entry point
└── LICENSE
```

## Roadmap

- [ ] PyPI release (`pip install rag-eval-kit`) with published package
- [ ] Local/open-source embedding support via `sentence-transformers`
- [ ] HTML and CSV report export alongside `results.json`
- [ ] Async LLM calls to speed up large eval runs
- [ ] Pre-built dataset import to skip generation when ground truth already exists

## License

MIT — see LICENSE.

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
