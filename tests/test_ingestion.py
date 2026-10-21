import math

import pytest

from rag_eval_kit.ingestion import Chunk, DocumentLoader


def test_chunk_count() -> None:
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    text = "a " * 200  # 400 characters
    chunks = loader._split(text, "fake.txt")
    step = 100 - 20  # 80
    expected = math.ceil((len(text) - 100) / step) + 1
    assert len(chunks) == expected


def test_chunk_overlap_content() -> None:
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    text = "x" * 300
    chunks = loader._split(text, "fake.txt")
    assert len(chunks) >= 2
    # last 20 chars of chunk N should appear at start of chunk N+1
    tail = chunks[0].text[-20:]
    head = chunks[1].text[:20]
    assert tail == head


def test_load_txt_file(tmp_path) -> None:
    content = "Hello world. " * 50
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(content, encoding="utf-8")

    loader = DocumentLoader(chunk_size=50, chunk_overlap=10)
    chunks = loader.load_directory(tmp_path)

    assert len(chunks) > 0
    assert all(c.source == str(txt_file) for c in chunks)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_unsupported_extension_skipped(tmp_path) -> None:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2\n", encoding="utf-8")

    loader = DocumentLoader()
    chunks = loader.load_directory(tmp_path)

    assert chunks == []
