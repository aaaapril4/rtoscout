"""LLM interaction and RTO scoring logic (Ollama, local)."""
import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RTO_SCORING_SYSTEM_PROMPT,
    SCORE_MAX,
    SCORE_MIN,
)
from ..schemas.models import RTOScoreResult


_SCORING_OUTPUT_SYSTEM_SUFFIX = (
    'Use ONLY the provided excerpts. Output ONLY valid JSON with '
    'keys "score" (integer 0-10) and "rationale" (string). No other text.'
)


class Analyzer:
    """RTO scoring from retrieved context (Ollama)."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or OLLAMA_MODEL
        self._base_url = base_url or OLLAMA_BASE_URL
        self._llm: Optional[BaseChatModel] = None

    def _get_llm(self) -> BaseChatModel:
        if self._llm is not None:
            return self._llm
        from langchain_ollama import ChatOllama
        self._llm = ChatOllama(
            model=self.model,
            temperature=0,
            base_url=self._base_url,
        )
        return self._llm

    def score(
        self,
        ticker: str,
        context: str,
    ) -> RTOScoreResult:
        """Score the company from retrieved filing context and return rationale."""
        if not (context or "").strip():
            return RTOScoreResult(
                score=0,
                rationale="No context found in the provided 10-K excerpts.",
            )

        user_content = f"""Company: {ticker}

Excerpts from 10-K (RTO/workplace related):

{context}

Provide the JSON score and rationale."""

        system_content = f"{RTO_SCORING_SYSTEM_PROMPT.rstrip()}\n\n{_SCORING_OUTPUT_SYSTEM_SUFFIX}"
        response = self._get_llm().invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        return self._parse_response(text)

    def _parse_response(self, text: str) -> RTOScoreResult:
        text = text.strip()
        start = text.find("{")
        if start >= 0:
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    obj = json.loads(text[start:end])
                    score = float(obj.get("score", 0))
                    score = max(SCORE_MIN, min(SCORE_MAX, score))
                    return RTOScoreResult(
                        score=score,
                        rationale=obj.get("rationale", "No rationale provided."),
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        score_m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        rationale_m = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if score_m:
            score = max(SCORE_MIN, min(SCORE_MAX, float(score_m.group(1))))
            rationale = rationale_m.group(1).replace("\\n", "\n").replace('\\"', '"') if rationale_m else "No rationale provided."
            return RTOScoreResult(score=score, rationale=rationale)
        raise ValueError(f"Could not parse score from LLM response: {text[:500]}")

