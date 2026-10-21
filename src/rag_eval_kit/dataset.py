from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rag_eval_kit.ingestion import Chunk

Difficulty = Literal["factual", "multi-hop"]

_SYSTEM_PROMPT = 'Respond only with valid JSON matching the schema {"question": "...", "answer": "..."}'


@dataclass
class QAItem:
    question: str
    answer: str
    source_chunk: str
    source: str
    difficulty: Difficulty


class DatasetGenerator:
    def __init__(
        self,
        llm: Literal["openai", "anthropic"] = "openai",
        model: str | None = None,
    ) -> None:
        self.llm = llm
        if model is not None:
            self.model = model
        elif llm == "openai":
            self.model = "gpt-4o-mini"
        else:
            self.model = "claude-haiku-4-5-20251001"

    def generate(
        self,
        chunks: list[Chunk],
        n: int = 20,
        difficulty: Difficulty = "factual",
    ) -> list[QAItem]:
        if difficulty == "multi-hop" and len(chunks) < 2:
            raise ValueError("multi-hop difficulty requires at least 2 chunks")

        items: list[QAItem] = []
        for _ in range(n):
            if difficulty == "factual":
                chunk = random.choice(chunks)
                items.append(self._generate_one_factual(chunk))
            else:
                chunk_a, chunk_b = random.sample(chunks, 2)
                items.append(self._generate_one_multihop(chunk_a, chunk_b))
        return items

    def _generate_one_factual(self, chunk: Chunk) -> QAItem:
        prompt = (
            "Given the following passage, write one question whose answer can only be found\n"
            "in this passage, and provide the exact answer.\n\n"
            f"Passage:\n{chunk.text[:2000]}\n\n"
            'Respond with JSON: {"question": "...", "answer": "..."}'
        )
        raw = self._call_llm(prompt)
        return self._parse_qa(raw, chunk, "factual")

    def _generate_one_multihop(self, chunk_a: Chunk, chunk_b: Chunk) -> QAItem:
        prompt = (
            "Given these two passages, write one question that requires information from\n"
            "BOTH passages to answer correctly. Provide the full answer.\n\n"
            f"Passage A:\n{chunk_a.text[:1000]}\n\n"
            f"Passage B:\n{chunk_b.text[:1000]}\n\n"
            'Respond with JSON: {"question": "...", "answer": "..."}'
        )
        raw = self._call_llm(prompt)
        # Build a synthetic chunk for multi-hop
        combined_source_chunk = chunk_a.text[:500] + " | " + chunk_b.text[:500]

        data = json.loads(raw)
        if "question" not in data or "answer" not in data:
            raise ValueError(f"LLM response missing 'question' or 'answer' keys: {raw!r}")
        return QAItem(
            question=data["question"],
            answer=data["answer"],
            source_chunk=combined_source_chunk,
            source=chunk_a.source,
            difficulty="multi-hop",
        )

    def _call_llm(self, prompt: str) -> str:
        if self.llm == "openai":
            import openai

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        else:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text from first TextBlock
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

    def _parse_qa(self, raw: str, chunk: Chunk, difficulty: Difficulty) -> QAItem:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc
        if "question" not in data or "answer" not in data:
            raise ValueError(f"LLM response missing 'question' or 'answer' keys: {raw!r}")
        return QAItem(
            question=data["question"],
            answer=data["answer"],
            source_chunk=chunk.text,
            source=chunk.source,
            difficulty=difficulty,
        )

    @staticmethod
    def save(items: list[QAItem], path: str | Path = "dataset.json") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(i) for i in items], fh, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str | Path = "dataset.json") -> list[QAItem]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [QAItem(**d) for d in data]
