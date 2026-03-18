"""
Prompt templates for DocuMind RAG pipeline.

All LLM interactions use structured prompts that enforce JSON-only output,
source citation, and grounded answers. Each function returns a
(system_prompt, user_prompt) tuple.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MAX_CHAT_HISTORY_MESSAGES: int = 6
MAX_CONTEXT_CHARS: int = 8000


# ──────────────────────────────────────────────────────────────────────
# RAG question-answering prompts
# ──────────────────────────────────────────────────────────────────────

_QA_SYSTEM = """You are DocuMind, an expert document analysis assistant.
You answer questions ONLY using the provided document context.

STRICT RULES:
1. ONLY use information from the provided context chunks to answer.
2. ALWAYS cite your sources using [Page X, filename] format inline in your answer.
3. If the context does not contain enough information to answer the question,
   say: "I don't have enough information in the provided documents to answer
   this question." and set confidence to 0.1 or lower.
4. NEVER hallucinate or make up information not present in the context.
5. Be concise but complete. Use markdown formatting for clarity.
6. When multiple sources support your answer, cite all of them.
7. When the question asks for specific numbers, statistics, or facts, ONLY
   provide them if they appear VERBATIM in the context. Do not estimate or
   infer numerical values that are not explicitly stated.
8. If you can only partially answer the question, state what you know from
   the context and explicitly note what is NOT covered in the provided
   documents.

You MUST respond with valid JSON only. No markdown fences, no preamble.
JSON schema:
{
  "answer": "<your answer in markdown, with [Page X, filename] citations>",
  "confidence": <float 0.0-1.0 based on how well the context answers the question>,
  "sources_used": [<indices of context chunks you actually used, e.g. 0, 2, 4>],
  "follow_up_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}

Confidence guide:
- 0.9-1.0: Context directly and fully answers the question
- 0.7-0.8: Context mostly answers the question with minor gaps
- 0.4-0.6: Context partially answers — clearly state what parts are NOT covered
- 0.1-0.3: Context is tangentially related
- 0.0: Context is completely irrelevant"""


def get_qa_prompts(
    question: str,
    context_chunks: List[Dict[str, str]],
    chat_history: List[Dict[str, str]],
) -> Tuple[str, str]:
    """Build the QA system and user prompts.

    Args:
        question: The user's current question.
        context_chunks: List of dicts with keys 'text', 'source_filename',
            'page_number', 'chunk_index'.
        chat_history: Recent chat messages as dicts with 'role' and 'content'.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Format context
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        context_parts.append(
            f"[Source {i} — Page {chunk['page_number']}, "
            f"{chunk['source_filename']}]\n{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    # Truncate context if too long
    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

    # Format chat history (last N messages)
    history_str = ""
    recent = chat_history[-MAX_CHAT_HISTORY_MESSAGES:]
    if recent:
        history_lines = []
        for msg in recent:
            role = msg["role"].capitalize()
            history_lines.append(f"{role}: {msg['content']}")
        history_str = (
            "CONVERSATION HISTORY:\n"
            + "\n".join(history_lines)
            + "\n\n"
        )

    user_prompt = (
        f"{history_str}"
        f"DOCUMENT CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond with valid JSON only."
    )

    return _QA_SYSTEM, user_prompt


# ──────────────────────────────────────────────────────────────────────
# Follow-up detection prompt
# ──────────────────────────────────────────────────────────────────────

_FOLLOWUP_SYSTEM = """You are a question classifier. Your job is to determine
if a user's question is a follow-up to a previous conversation, and if so,
rewrite it as a standalone question that includes all necessary context.

You MUST respond with valid JSON only. No markdown fences, no preamble.
JSON schema:
{
  "is_followup": <true or false>,
  "standalone_question": "<the original or rewritten question>"
}

Examples:
- Original: "What about section 3?" after discussing a document
  → {"is_followup": true, "standalone_question": "What does section 3 of the document discuss?"}
- Original: "Tell me more about that"
  → {"is_followup": true, "standalone_question": "<rewrite based on context>"}
- Original: "What is the main argument of this paper?"
  → {"is_followup": false, "standalone_question": "What is the main argument of this paper?"}"""


def get_followup_detection_prompt(
    question: str,
    chat_history: List[Dict[str, str]],
) -> Tuple[str, str]:
    """Build prompt to detect and rewrite follow-up questions.

    Args:
        question: The user's current question.
        chat_history: Recent chat messages as dicts with 'role' and 'content'.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    recent = chat_history[-MAX_CHAT_HISTORY_MESSAGES:]
    history_lines = []
    for msg in recent:
        role = msg["role"].capitalize()
        history_lines.append(f"{role}: {msg['content']}")
    history_str = "\n".join(history_lines)

    user_prompt = (
        f"CONVERSATION HISTORY:\n{history_str}\n\n"
        f"CURRENT QUESTION: {question}\n\n"
        f"Is this a follow-up question? If yes, rewrite it as standalone. "
        f"Respond with valid JSON only."
    )

    return _FOLLOWUP_SYSTEM, user_prompt
