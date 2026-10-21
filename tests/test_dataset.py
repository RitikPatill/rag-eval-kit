import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_eval_kit.dataset import DatasetGenerator, QAItem
from rag_eval_kit.ingestion import Chunk

_MOCK_RESPONSE = '{"question": "Q?", "answer": "A."}'


def _make_chunks(n: int = 3) -> list[Chunk]:
    return [Chunk(text=f"Passage text number {i}.", source=f"doc{i}.txt", index=i) for i in range(n)]


def test_save_load_roundtrip(tmp_path: Path) -> None:
    items = [
        QAItem(question="Q1?", answer="A1.", source_chunk="chunk text", source="file.txt", difficulty="factual"),
        QAItem(question="Q2?", answer="A2.", source_chunk="more text", source="file2.txt", difficulty="multi-hop"),
    ]
    out = tmp_path / "dataset.json"
    DatasetGenerator.save(items, out)

    assert out.exists()
    raw = json.loads(out.read_text())
    assert len(raw) == 2

    loaded = DatasetGenerator.load(out)
    assert loaded == items


def test_generate_factual_count() -> None:
    gen = DatasetGenerator(llm="openai")
    chunks = _make_chunks(3)
    with patch.object(gen, "_call_llm", return_value=_MOCK_RESPONSE):
        result = gen.generate(chunks, n=5, difficulty="factual")
    assert len(result) == 5
    assert all(item.difficulty == "factual" for item in result)


def test_generate_multihop_count() -> None:
    gen = DatasetGenerator(llm="openai")
    chunks = _make_chunks(3)
    with patch.object(gen, "_call_llm", return_value=_MOCK_RESPONSE):
        result = gen.generate(chunks, n=4, difficulty="multi-hop")
    assert len(result) == 4
    assert all(item.difficulty == "multi-hop" for item in result)


def test_multihop_requires_two_chunks() -> None:
    gen = DatasetGenerator(llm="openai")
    chunks = _make_chunks(1)
    with pytest.raises(ValueError, match="at least 2 chunks"):
        gen.generate(chunks, n=1, difficulty="multi-hop")


def test_parse_qa_bad_json() -> None:
    gen = DatasetGenerator(llm="openai")
    chunk = Chunk(text="some text", source="file.txt", index=0)
    with pytest.raises(ValueError, match="invalid JSON"):
        gen._parse_qa("not valid json{{", chunk, "factual")


def test_generate_uses_chunk_text() -> None:
    gen = DatasetGenerator(llm="openai")
    chunks = _make_chunks(2)
    with patch.object(gen, "_call_llm", return_value=_MOCK_RESPONSE):
        result = gen.generate(chunks, n=2, difficulty="factual")
    # source_chunk should match one of the chunk texts
    chunk_texts = {c.text for c in chunks}
    for item in result:
        assert item.source_chunk in chunk_texts
