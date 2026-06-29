"""Summarization agent for the multi-agent LangGraph (Issue #7).

Public surface
--------------

- :class:`SummarizationAgent` — two methods: :meth:`summarize_single`
  for one chunk, and :meth:`summarize_map_reduce` for a list of
  chunks.

Map-reduce strategy
-------------------

For a list of N chunks, :meth:`summarize_map_reduce` groups them into
batches of ``max_chunks_per_group`` (default 5), calls the LLM once
per group to produce a per-group summary, then calls the LLM one
final time to reduce the per-group summaries into a single
coherent answer. The intermediate summaries are not exposed — the
caller only sees the final reduced string.

Why group first (not summarize-each-then-concat)
-----------------------------------------------

Concatenating N per-chunk summaries produces a verbose, repetitive
output ("chunk 1 says X. chunk 2 says Y. chunk 3 says Z..."). The
group-then-reduce pattern produces a coherent narrative because the
reduce step can de-duplicate and re-organize. This is a small
map-reduce (the LLM is the reducer); it's NOT a true MR pattern with
a separate combiner, but it's the right shape for LLM-driven
summarization of small N.

Empty input
-----------

:meth:`summarize_map_reduce` with an empty list returns the empty
string (no LLM call) and :meth:`summarize_single` with an empty
string returns the empty string. Tests assert both paths don't crash
and don't burn an LLM call on empty input.
"""

from __future__ import annotations

from typing import Any, Dict, List

from telecom_rag import llm
from telecom_rag.llm import ProviderCallError
from telecom_rag.observability.cost import Agent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

__all__ = ["SummarizationAgent", "ProviderCallError"]  # noqa: F401


_RETRYING = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ProviderCallError),
    reraise=True,
)


_SYSTEM_PROMPT = (
    "You are a summarization agent for telecom-domain documents. "
    "Produce a clear, coherent summary that preserves the key technical "
    "facts and procedures. Do not invent details not present in the input."
)


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Render chunks as a numbered text block.

    Mirrors :func:`telecom_rag.agents.validation_agent._format_chunks`
    — duplicated here rather than imported so the two agents can
    evolve independently (a validation-specific format change should
    not force a summarization change).
    """
    blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown.pdf") or "unknown.pdf"
        page_number = chunk.get("page_number", 0) or 0
        text = chunk.get("text", "") or ""
        blocks.append(f"[{idx}] {source_file} p.{page_number}\n{text}")
    return "\n\n".join(blocks)


class SummarizationAgent:
    """Map-reduce document summarizer.

    Stateless; safe to construct once at module scope.
    """

    @_RETRYING
    def summarize_single(self, text: str) -> str:
        """Summarize a single chunk of text.

        Returns the empty string for an empty input without making
        an LLM call (tests assert this to keep the cold-path cost
        at zero).
        """
        if not text or not text.strip():
            return ""
        prompt = (
            "Summarize the following telecom-document excerpt:\n\n"
            f"{text}"
        )
        result = llm.chat_with_fallback(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.2,  # slight creativity for narrative flow
            agent=Agent.SUMMARIZATION,
        )
        return result.text

    def summarize_map_reduce(
        self,
        chunks: List[Dict[str, Any]],
        *,
        max_chunks_per_group: int = 5,
    ) -> str:
        """Map-reduce summarization across a list of chunks.

        Splits ``chunks`` into groups of ``max_chunks_per_group``,
        summarizes each group, then reduces the per-group summaries
        into a final answer. Returns the empty string for an empty
        input without making an LLM call.
        """
        if not chunks:
            return ""

        # ---- Map phase: per-group summaries ---------------------------
        groups: List[List[Dict[str, Any]]] = [
            chunks[i : i + max_chunks_per_group]
            for i in range(0, len(chunks), max_chunks_per_group)
        ]
        group_summaries: List[str] = []
        for group in groups:
            context = _format_chunks(group)
            prompt = (
                "Summarize the following group of telecom-document "
                "excerpts into a single coherent paragraph:\n\n"
                f"{context}"
            )
            result = llm.chat_with_fallback(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.2,
                agent=Agent.SUMMARIZATION,
            )
            group_summaries.append(result.text)

        # ---- Reduce phase: combine into final answer -----------------
        # If we only had one group, its summary IS the final answer
        # (avoids a redundant reduce LLM call).
        if len(group_summaries) == 1:
            return group_summaries[0]

        joined = "\n\n---\n\n".join(group_summaries)
        prompt = (
            "Combine the following per-group summaries into a single "
            "coherent, de-duplicated summary. Preserve the key technical "
            "facts and procedures:\n\n"
            f"{joined}"
        )
        result = llm.chat_with_fallback(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.2,
            agent=Agent.SUMMARIZATION,
        )
        return result.text
