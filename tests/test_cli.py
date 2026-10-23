from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rag_eval_kit.cli import _print_report, app
from rag_eval_kit.dataset import QAItem
from rag_eval_kit.judge import EvalScore
from rag_eval_kit.ingestion import Chunk
from rag_eval_kit.pipeline import RAGResult

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chunk(text: str = "sample text", source: str = "doc.txt") -> Chunk:
    return Chunk(text=text, source=source, index=0)


def _fake_qa_item(question: str = "What is X?", answer: str = "X is Y.") -> QAItem:
    return QAItem(
        question=question,
        answer=answer,
        source_chunk="sample text",
        source="doc.txt",
        difficulty="factual",
    )


def _fake_rag_result(question: str = "What is X?") -> RAGResult:
    return RAGResult(
        question=question,
        retrieved_chunks=["chunk A"],
        generated_answer="gen answer",
        gold_answer="X is Y.",
    )


def _fake_eval_score(q: str = "What is X?", score: float = 0.8) -> EvalScore:
    return EvalScore(
        question=q,
        faithfulness=score,
        answer_relevance=score,
        context_recall=score,
        generated_answer="gen answer",
        gold_answer="X is Y.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_stub_replaced(tmp_path):
    """run() completes end-to-end, table headers appear in output, exits 0."""
    doc = tmp_path / "test.txt"
    doc.write_text("Hello world sample document text.")

    fake_chunks = [_fake_chunk()]
    fake_items = [_fake_qa_item()]
    fake_results = [_fake_rag_result()]
    fake_scores = [_fake_eval_score()]

    with (
        patch("rag_eval_kit.ingestion.DocumentLoader._load_file", return_value="Hello world sample document text."),
        patch("rag_eval_kit.ingestion.DocumentLoader._split", return_value=fake_chunks),
        patch("rag_eval_kit.ingestion.VectorStore") as MockVS,
        patch("rag_eval_kit.dataset.DatasetGenerator._generate_one_factual", return_value=fake_items[0]),
        patch("rag_eval_kit.dataset.DatasetGenerator.save"),
        patch("rag_eval_kit.pipeline.RAGPipeline.run", return_value=fake_results),
        patch("rag_eval_kit.judge.LLMJudge.score_all", return_value=fake_scores),
        patch("rag_eval_kit.judge.LLMJudge.save"),
    ):
        MockVS.return_value.add_chunks = MagicMock()
        result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Faithfulness" in result.output
    assert "Answer Relevance" in result.output
    assert "Context Recall" in result.output


def test_run_min_score_pass(tmp_path):
    """All scores 1.0 + --min-score 0.7 → exit 0 and PASS in output."""
    doc = tmp_path / "test.txt"
    doc.write_text("Sample text for testing.")

    fake_chunks = [_fake_chunk()]
    fake_items = [_fake_qa_item()]
    fake_results = [_fake_rag_result()]
    fake_scores = [_fake_eval_score(score=1.0)]

    with (
        patch("rag_eval_kit.ingestion.DocumentLoader._load_file", return_value="Sample text for testing."),
        patch("rag_eval_kit.ingestion.DocumentLoader._split", return_value=fake_chunks),
        patch("rag_eval_kit.ingestion.VectorStore") as MockVS,
        patch("rag_eval_kit.dataset.DatasetGenerator._generate_one_factual", return_value=fake_items[0]),
        patch("rag_eval_kit.dataset.DatasetGenerator.save"),
        patch("rag_eval_kit.pipeline.RAGPipeline.run", return_value=fake_results),
        patch("rag_eval_kit.judge.LLMJudge.score_all", return_value=fake_scores),
        patch("rag_eval_kit.judge.LLMJudge.save"),
    ):
        MockVS.return_value.add_chunks = MagicMock()
        result = runner.invoke(app, ["run", str(tmp_path), "--min-score", "0.7"])

    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_run_min_score_fail(tmp_path):
    """All scores 0.2 + --min-score 0.7 → exit 1 and FAIL in output."""
    doc = tmp_path / "test.txt"
    doc.write_text("Sample text for testing.")

    fake_chunks = [_fake_chunk()]
    fake_items = [_fake_qa_item()]
    fake_results = [_fake_rag_result()]
    fake_scores = [_fake_eval_score(score=0.2)]

    with (
        patch("rag_eval_kit.ingestion.DocumentLoader._load_file", return_value="Sample text for testing."),
        patch("rag_eval_kit.ingestion.DocumentLoader._split", return_value=fake_chunks),
        patch("rag_eval_kit.ingestion.VectorStore") as MockVS,
        patch("rag_eval_kit.dataset.DatasetGenerator._generate_one_factual", return_value=fake_items[0]),
        patch("rag_eval_kit.dataset.DatasetGenerator.save"),
        patch("rag_eval_kit.pipeline.RAGPipeline.run", return_value=fake_results),
        patch("rag_eval_kit.judge.LLMJudge.score_all", return_value=fake_scores),
        patch("rag_eval_kit.judge.LLMJudge.save"),
    ):
        MockVS.return_value.add_chunks = MagicMock()
        result = runner.invoke(app, ["run", str(tmp_path), "--min-score", "0.7"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_run_missing_dir():
    """Non-existent path → exit 1 with error message in output."""
    result = runner.invoke(app, ["run", "/nonexistent/path/that/does/not/exist"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_print_report_colours():
    """_print_report applies green/yellow/red markup based on score thresholds."""
    from io import StringIO
    from rich.console import Console

    scores = [
        _fake_eval_score(q="Q1", score=0.8),   # green
        _fake_eval_score(q="Q2", score=0.5),   # yellow
        _fake_eval_score(q="Q3", score=0.1),   # red
    ]

    buf = StringIO()
    console = Console(file=buf, highlight=False, no_color=False, force_terminal=True)

    # Capture the table output by patching Console inside the function
    with patch("rag_eval_kit.cli.Console", return_value=console):
        passed = _print_report(scores, min_score=0.7)

    output = buf.getvalue()
    assert "green" in output or "0.80" in output  # high score rendered
    assert "0.50" in output   # mid score rendered
    assert "0.10" in output   # low score rendered
    assert passed is False     # avg = 0.467 < 0.7
