from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_eval_kit.judge import EvalScore, LLMJudge
from rag_eval_kit.pipeline import RAGResult


def _make_result(
    question: str = "What is X?",
    chunks: list[str] | None = None,
    generated: str = "X is Y.",
    gold: str = "X is definitely Y.",
) -> RAGResult:
    return RAGResult(
        question=question,
        retrieved_chunks=chunks or ["chunk A"],
        generated_answer=generated,
        gold_answer=gold,
    )


def test_score_one_normalises():
    judge = LLMJudge()
    raw_json = json.dumps({"faithfulness": 5, "answer_relevance": 5, "context_recall": 5})
    with patch.object(judge, "_call_llm", return_value=raw_json):
        score = judge._score_one(_make_result())
    assert score.faithfulness == 1.0
    assert score.answer_relevance == 1.0
    assert score.context_recall == 1.0


def test_score_one_partial():
    judge = LLMJudge()
    raw_json = json.dumps({"faithfulness": 3, "answer_relevance": 2, "context_recall": 4})
    with patch.object(judge, "_call_llm", return_value=raw_json):
        score = judge._score_one(_make_result())
    assert score.faithfulness == pytest.approx(0.6)
    assert score.answer_relevance == pytest.approx(0.4)
    assert score.context_recall == pytest.approx(0.8)


def test_score_all_count():
    judge = LLMJudge()
    results = [_make_result(f"Q{i}") for i in range(4)]
    raw_json = json.dumps({"faithfulness": 4, "answer_relevance": 3, "context_recall": 5})
    with patch.object(judge, "_call_llm", return_value=raw_json):
        scores = judge.score_all(results)
    assert len(scores) == 4
    assert all(isinstance(s, EvalScore) for s in scores)


def test_save_creates_json():
    scores = [
        EvalScore(
            question="Q?",
            faithfulness=0.8,
            answer_relevance=0.6,
            context_recall=1.0,
            generated_answer="gen",
            gold_answer="gold",
        )
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.json"
        LLMJudge.save(scores, path)
        assert path.exists()
        data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["faithfulness"] == 0.8
    assert data[0]["answer_relevance"] == 0.6
    assert data[0]["context_recall"] == 1.0
    assert "question" in data[0]
    assert "generated_answer" in data[0]
    assert "gold_answer" in data[0]


def test_bad_json_raises():
    judge = LLMJudge()
    with patch.object(judge, "_call_llm", return_value="not valid json"):
        with pytest.raises(ValueError, match="invalid JSON"):
            judge._score_one(_make_result())
