from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_eval_kit.dataset import QAItem
from rag_eval_kit.pipeline import RAGPipeline, RAGResult


def _make_item(question: str = "What is X?", answer: str = "X is Y.") -> QAItem:
    return QAItem(
        question=question,
        answer=answer,
        source_chunk="chunk text",
        source="doc.txt",
        difficulty="factual",
    )


def _make_pipeline() -> RAGPipeline:
    vs = MagicMock()
    vs.query.return_value = ["chunk A", "chunk B"]
    pipeline = RAGPipeline(vector_store=vs)
    return pipeline


def test_run_returns_correct_count():
    pipeline = _make_pipeline()
    items = [_make_item(f"Q{i}") for i in range(5)]
    with patch.object(pipeline, "_call_llm", return_value="Some answer"):
        results = pipeline.run(items)
    assert len(results) == 5
    assert all(isinstance(r, RAGResult) for r in results)


def test_retrieve_calls_vector_store():
    vs = MagicMock()
    vs.query.return_value = ["chunk A", "chunk B"]
    pipeline = RAGPipeline(vector_store=vs, top_k=3)
    chunks = pipeline._retrieve("test question")
    vs.query.assert_called_once_with("test question", top_k=3)
    assert chunks == ["chunk A", "chunk B"]


def test_generate_answer_uses_chunks():
    pipeline = _make_pipeline()
    with patch.object(pipeline, "_call_llm", return_value="generated answer") as mock_llm:
        answer = pipeline._generate_answer("Q?", ["chunk A", "chunk B"])
    mock_llm.assert_called_once()
    assert answer == "generated answer"
    # Verify the chunks appear in the prompt
    call_prompt = mock_llm.call_args[0][0]
    assert "chunk A" in call_prompt
    assert "chunk B" in call_prompt


def test_gold_answer_carried_through():
    pipeline = _make_pipeline()
    item = _make_item(question="What is X?", answer="X is definitely Y.")
    with patch.object(pipeline, "_call_llm", return_value="generated"):
        results = pipeline.run([item])
    assert results[0].gold_answer == "X is definitely Y."
