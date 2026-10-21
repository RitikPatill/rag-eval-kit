from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress


@dataclass
class Chunk:
    text: str
    source: str
    index: int


class DocumentLoader:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_directory(self, path: str | Path) -> list[Chunk]:
        supported = {".pdf", ".txt", ".md"}
        chunks: list[Chunk] = []
        for file in Path(path).rglob("*"):
            if file.suffix.lower() in supported:
                text = self._load_file(file)
                chunks.extend(self._split(text, str(file)))
        return chunks

    def _load_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(str(path))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n".join(pages)
        else:
            return path.read_text(encoding="utf-8", errors="replace")

    def _split(self, text: str, source: str) -> list[Chunk]:
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        chunks: list[Chunk] = []
        index = 0
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, index=index))
                index += 1
            start += step
        return chunks


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_eval",
        persist_directory: str = ".chroma",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        import chromadb

        self.embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(collection_name)

    def add_chunks(self, chunks: list[Chunk], progress: "Progress | None" = None) -> None:
        import openai

        client = openai.OpenAI()
        batch_size = 100

        task_id = None
        if progress is not None:
            task_id = progress.add_task("Embedding chunks…", total=len(chunks))

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [c.text for c in batch]
            response = client.embeddings.create(model=self.embedding_model, input=texts)
            embeddings = [e.embedding for e in response.data]
            ids = [f"{c.source}::{c.index}" for c in batch]
            documents = [c.text for c in batch]
            metadatas = [{"source": c.source, "index": c.index} for c in batch]
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            if progress is not None and task_id is not None:
                progress.advance(task_id, len(batch))

    def collection_size(self) -> int:
        return self._collection.count()
