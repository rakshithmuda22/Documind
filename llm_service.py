"""
LLM Service for DocuMind.

Handles all Groq API interactions: question answering with RAG context,
follow-up detection, and robust JSON parsing. Uses AsyncGroq for
non-blocking FastAPI compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError

from prompts import get_followup_detection_prompt, get_qa_prompts

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MODEL_NAME: str = "llama-3.1-8b-instant"
MAX_RETRIES: int = 2
RETRY_BACKOFF_SECONDS: float = 1.5
DEFAULT_MAX_TOKENS: int = 1500
FOLLOWUP_MAX_TOKENS: int = 256

# Default fallback when QA JSON parsing fails
_QA_FALLBACK: Dict[str, Any] = {
    "answer": "I encountered an issue generating a response. Please try again.",
    "confidence": 0.0,
    "sources_used": [],
    "follow_up_suggestions": [],
}

_FOLLOWUP_FALLBACK: Dict[str, Any] = {
    "is_followup": False,
    "standalone_question": "",
}


class LLMService:
    """Async Groq LLM client for RAG question answering.

    Provides retry logic with exponential backoff, streaming support,
    and robust JSON extraction from LLM responses.
    """

    def __init__(self) -> None:
        """Initialise the async Groq client.

        Raises:
            RuntimeError: On first LLM call if GROQ_API_KEY is not set.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY not set — LLM calls will fail until configured. "
                "Copy .env.example to .env and add your free key from https://console.groq.com"
            )
            self.client = None
        else:
            self.client = AsyncGroq(api_key=api_key)
        logger.info("LLMService initialised — model: %s", MODEL_NAME)

    # ──────────────────────────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────────────────────────

    async def answer_question(
        self,
        question: str,
        context_chunks: List[Dict[str, str]],
        chat_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Generate a RAG-grounded answer with confidence and sources.

        Args:
            question: The user's question.
            context_chunks: Retrieved document chunks with metadata.
            chat_history: Recent chat messages for conversation context.

        Returns:
            Dict with keys: answer, confidence, sources_used,
            follow_up_suggestions.
        """
        system_prompt, user_prompt = get_qa_prompts(
            question, context_chunks, chat_history,
        )

        raw = await self._call_llm(
            system_prompt,
            user_prompt,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        parsed = self._parse_json_response(raw, None)

        # Fallback chain: full JSON → regex field extraction → plain text
        if parsed is None:
            logger.info(
                "Full JSON parse failed — trying regex field extraction"
            )
            parsed = self._extract_fields_from_raw(raw)

        if parsed is None:
            # Truly plain-text response — strip any JSON-like artefacts
            logger.info(
                "Regex extraction also failed — using cleaned plain text"
            )
            clean = raw.strip()
            # If the whole response looks like a JSON blob, discard it
            if clean.startswith("{") and clean.endswith("}"):
                clean = _QA_FALLBACK["answer"]
            result = {
                "answer": clean,
                "confidence": 0.5,
                "sources_used": list(range(len(context_chunks))),
                "follow_up_suggestions": [],
            }
        else:
            # Validate and normalise fields
            result = {
                "answer": str(parsed.get("answer", raw.strip())),
                "confidence": self._clamp_float(
                    parsed.get("confidence", 0.5), 0.0, 1.0,
                ),
                "sources_used": parsed.get("sources_used", []),
                "follow_up_suggestions": parsed.get(
                    "follow_up_suggestions", [],
                )[:3],
            }

        return result

    async def detect_follow_up(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Detect if a question is a follow-up and rewrite as standalone.

        Args:
            question: The user's current question.
            chat_history: Recent chat messages.

        Returns:
            Dict with is_followup (bool) and standalone_question (str).
        """
        system_prompt, user_prompt = get_followup_detection_prompt(
            question, chat_history,
        )

        raw = await self._call_llm(
            system_prompt,
            user_prompt,
            max_tokens=FOLLOWUP_MAX_TOKENS,
        )

        parsed = self._parse_json_response(raw, _FOLLOWUP_FALLBACK)

        return {
            "is_followup": bool(parsed.get("is_followup", False)),
            "standalone_question": str(
                parsed.get("standalone_question", question)
            ),
        }

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Call Groq with streaming, buffering the full response.

        Uses exponential backoff retry for transient errors and rate limits.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            max_tokens: Maximum tokens to generate.

        Returns:
            Complete response text.

        Raises:
            RuntimeError: After exhausting all retries or if API key missing.
        """
        if self.client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Add your free API key from https://console.groq.com to the .env file."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                stream = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    stream=True,
                )

                # Buffer the streamed response
                chunks = []
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta is not None:
                        chunks.append(delta)

                return "".join(chunks)

            except RateLimitError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    import asyncio
                    wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    logger.warning(
                        "Rate limited (attempt %d/%d). Waiting %.1fs...",
                        attempt + 1, MAX_RETRIES + 1, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        "AI service is busy due to rate limiting. "
                        "Please wait a few seconds and try again."
                    ) from exc

            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    import asyncio
                    wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, MAX_RETRIES + 1, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("LLM call failed after %d attempts: %s", MAX_RETRIES + 1, exc)
                    raise RuntimeError(
                        "Failed to get a response from the AI service. "
                        "Please try again."
                    ) from exc

        raise RuntimeError("Unexpected retry loop exit") from last_error

    @staticmethod
    def _parse_json_response(raw: str, fallback: Any) -> Any:
        """Robustly extract JSON from an LLM response string.

        Tries multiple strategies: direct parse, code fences,
        brace/bracket extraction.

        Args:
            raw: Raw LLM response string.
            fallback: Value returned when parsing fails.

        Returns:
            Parsed Python object or fallback.
        """
        if not raw:
            return fallback

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from code fences
        try:
            if "```json" in raw:
                s = raw.index("```json") + 7
                e = raw.index("```", s)
                return json.loads(raw[s:e].strip())
            if "```" in raw:
                s = raw.index("```") + 3
                e = raw.index("```", s)
                return json.loads(raw[s:e].strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2.5: Fix common LLM JSON malformations
        try:
            cleaned = raw.strip()
            # Remove trailing commas before } or ]
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            # Strip text before first { and after last }
            fb = cleaned.find("{")
            lb = cleaned.rfind("}")
            if fb != -1 and lb > fb:
                cleaned = cleaned[fb : lb + 1]
                return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3: Find outermost braces
        try:
            s, e = raw.find("{"), raw.rfind("}") + 1
            if s != -1 and e > s:
                return json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass

        # Strategy 4: Find outermost brackets
        try:
            s, e = raw.find("["), raw.rfind("]") + 1
            if s != -1 and e > s:
                return json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass

        logger.warning("JSON extraction failed; returning fallback.")
        return fallback

    @staticmethod
    def _clamp_float(val: Any, lo: float, hi: float) -> float:
        """Coerce a value to a float clamped between lo and hi.

        Args:
            val: Input value.
            lo: Minimum value.
            hi: Maximum value.

        Returns:
            Clamped float, or lo if conversion fails.
        """
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return lo

    @staticmethod
    def _extract_fields_from_raw(raw: str) -> Optional[Dict[str, Any]]:
        """Last-resort regex extraction of known QA fields from raw text.

        When ``_parse_json_response`` fails on all strategies, this method
        pulls individual field values using targeted regular expressions so
        we never show raw JSON to the user.

        Args:
            raw: Raw LLM response string.

        Returns:
            Dict with answer/confidence/sources_used/follow_up_suggestions,
            or ``None`` if the answer field cannot be extracted.
        """
        # Must be able to extract the answer field at minimum
        ans_match = re.search(
            r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]',
            raw,
            re.DOTALL,
        )
        if not ans_match:
            return None

        answer = (
            ans_match.group(1)
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", " ")
        )

        # Confidence
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        # Sources used
        src_match = re.search(r'"sources_used"\s*:\s*\[([\d,\s]*)\]', raw)
        sources: List[int] = []
        if src_match and src_match.group(1).strip():
            sources = [
                int(x.strip())
                for x in src_match.group(1).split(",")
                if x.strip().isdigit()
            ]

        # Follow-up suggestions
        sug_match = re.search(
            r'"follow_up_suggestions"\s*:\s*\[(.*?)\]',
            raw,
            re.DOTALL,
        )
        suggestions: List[str] = []
        if sug_match:
            suggestions = re.findall(
                r'"((?:[^"\\]|\\.)*)"', sug_match.group(1),
            )

        return {
            "answer": answer,
            "confidence": confidence,
            "sources_used": sources,
            "follow_up_suggestions": suggestions[:3],
        }
