from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rag_eval_kit.pipeline import RAGResult

_SYSTEM_PROMPT = (
    'Respond only with valid JSON matching the schema: '
    '{"faithfulness": <int 0-5>, "answer_relevance": <int 0-5>, "context_recall": <int 0-5>}'
)


@dataclass
class EvalScore:
    question: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    generated_answer: str
    gold_answer: str


class LLMJudge:
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

    def score_all(self, results: list["RAGResult"]) -> list[EvalScore]:
        return [self._score_one(r) for r in results]

    def _score_one(self, result: "RAGResult") -> EvalScore:
        context = "\n\n".join(result.retrieved_chunks)
        prompt = (
            "Evaluate this RAG response on three dimensions (each scored 0–5):\n"
            "- faithfulness: is the generated answer grounded in the retrieved context (no hallucinations)?\n"
            "- answer_relevance: does the generated answer actually address the question?\n"
            "- context_recall: does the retrieved context contain the information needed to produce the gold answer?\n\n"
            f"Question: {result.question}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Generated Answer: {result.generated_answer}\n"
            f"Gold Answer: {result.gold_answer}"
        )
        raw = self._call_llm(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc

        def _norm(key: str) -> float:
            raw_val = data.get(key, 0)
            return max(0, min(5, int(raw_val))) / 5.0

        return EvalScore(
            question=result.question,
            faithfulness=_norm("faithfulness"),
            answer_relevance=_norm("answer_relevance"),
            context_recall=_norm("context_recall"),
            generated_answer=result.generated_answer,
            gold_answer=result.gold_answer,
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
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

    @staticmethod
    def save(scores: list[EvalScore], path: str | Path = "results.json") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([dataclasses.asdict(s) for s in scores], fh, ensure_ascii=False, indent=2)
