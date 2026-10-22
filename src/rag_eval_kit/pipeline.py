from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rag_eval_kit.dataset import QAItem
    from rag_eval_kit.ingestion import VectorStore


@dataclass
class RAGResult:
    question: str
    retrieved_chunks: list[str]
    generated_answer: str
    gold_answer: str


class RAGPipeline:
    def __init__(
        self,
        vector_store: "VectorStore",
        llm: Literal["openai", "anthropic"] = "openai",
        model: str | None = None,
        top_k: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        if model is not None:
            self.model = model
        elif llm == "openai":
            self.model = "gpt-4o-mini"
        else:
            self.model = "claude-haiku-4-5-20251001"

    def run(self, items: list["QAItem"]) -> list[RAGResult]:
        results: list[RAGResult] = []
        for item in items:
            chunks = self._retrieve(item.question)
            answer = self._generate_answer(item.question, chunks)
            results.append(
                RAGResult(
                    question=item.question,
                    retrieved_chunks=chunks,
                    generated_answer=answer,
                    gold_answer=item.answer,
                )
            )
        return results

    def _retrieve(self, question: str) -> list[str]:
        return self.vector_store.query(question, top_k=self.top_k)

    def _generate_answer(self, question: str, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)
        prompt = (
            "Use only the context below to answer the question. Be concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        if self.llm == "openai":
            import openai

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        else:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
