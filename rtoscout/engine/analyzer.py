"""LLM interaction and RTO scoring logic (Ollama, local)."""
import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL, SCORE_MAX, SCORE_MIN
from ..schemas.models import RTOScoreResult


SYSTEM_PROMPT = """You are an analyst scoring companies' Return-to-Office (RTO) flexibility based on filing excerpts.

Score from 0 to 10 (higher = more remote/flexible, lower = stricter in-office requirement):
- 0-2: Strict mandatory return-to-office with explicit in-person requirements.
- 3-4: Office-forward; clear attendance expectations and stronger in-person norms.
- 5-6: Balanced hybrid; recurring in-office expectation for many roles.
- 7-8: Flexible hybrid; office presence encouraged but mostly optional.
- 9-10: Remote-first / highly flexible; no meaningful office mandate.

Calibration guidance:
- If language is explicit and directive (for example: "required", "must", "expected to be in office"), score lower.
- If language is optional/employee-choice focused (for example: "flexible", "may work remotely"), score higher.
- If evidence is mixed, prefer middle values (4-7) and explain the tradeoff.
- If excerpts contain little or no workplace policy language, return a conservative mid score (typically 4-6), and explain uncertainty.

Use ONLY the provided excerpts. Output ONLY valid JSON with keys "score" (integer 0-10) and "rationale" (string). No other text."""


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

    def score_rto(
        self,
        ticker: str,
        context: str,
    ) -> RTOScoreResult:
        """Score the company from RTO context and return rationale."""
        if not (context or "").strip():
            return RTOScoreResult(
                score=0,
                rationale="No RTO-related context found in the provided 10-K excerpts.",
            )

        user_content = f"""Company: {ticker}

Excerpts from 10-K (RTO/workplace related):

{context}

Provide the JSON score and rationale."""

        response = self._get_llm().invoke([
            SystemMessage(content=SYSTEM_PROMPT),
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

