"""Validation agent for the multi-agent LangGraph (Issue #7).

Public surface
--------------

- :class:`ValidationAgent` — three independent grading methods plus a
  combined :meth:`grade_all` that returns the three scores plus the
  computed overall score (mean).
- :func:`telecom_rag.graphs.router.parse_json_text` — re-used here
  for JSON extraction; tests stub the LLM call the same way.

Grading dimensions
------------------

1. :meth:`grade_relevance` — are the retrieved chunks relevant to
   the user's query? Score 0.0 (totally irrelevant) to 1.0 (perfect).
2. :meth:`grade_faithfulness` — is the generated answer supported by
   the retrieved chunks (no hallucinated facts)? Same 0.0–1.0 range.
3. :meth:`grade_completeness` — does the answer fully address the
   user's question? Same range.

The three dimensions are graded independently so the graph's
conditional edge after :func:`validate_node` can compute
``overall_score = mean(relevance, faithfulness, completeness)`` and
route to ``refuse_node`` if ``overall_score < 0.6`` (README §7.3).

Tenacity retry
--------------

The README §11.1 retry spec calls for ``@retry`` on all LLM-touching
methods. We apply it to each of the three public grading methods so
transient OpenAI 429s retry transparently. The retry config matches
``telecom_rag.llm._retrying`` (``stop_after_attempt(3)``,
``wait_exponential(multiplier=1, min=1, max=10)``,
``retry_if_exception_type(ProviderCallError)``). Note the per-method
retry is INSIDE the LLM fallback layer's retry; the outer
``chat_with_fallback`` already retries per-provider.

JSON-from-LLM contract
----------------------

Same as the Router: the system prompt instructs the model to emit a
JSON object with ``{"score": <float>, "reason": <str>}``. We reuse
``parse_json_text`` from ``telecom_rag.graphs.router`` (duplicated
as a vendored copy would drift; the import keeps the parsing logic
in one place). The Pydantic ``RelevanceGrade`` / etc. models
validate the parsed dict; on validation failure we raise
:class:`ValueError`, which the graph's node-level try/except catches.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from telecom_rag import llm
from telecom_rag.graphs.router import parse_json_text
from telecom_rag.llm import ProviderCallError
from telecom_rag.observability.cost import Agent
from telecom_rag.schemas import (
    CompletenessGrade,
    FaithfulnessGrade,
    RelevanceGrade,
)

# Re-imported here so callers can do
# ``from telecom_rag.agents.validation_agent import ProviderCallError``
# without reaching into the LLM module. ``noqa: F401`` keeps ruff
# happy if the import would otherwise be flagged as unused at module
# scope.
__all__ = [
    "ValidationAgent",
    "ProviderCallError",  # noqa: F401
]


# Retry config matches ``telecom_rag.llm._retrying`` per README §11.1.
# Applied as a decorator on each public grading method.
_RETRYING = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ProviderCallError),
    reraise=True,
)


_BASE_SYSTEM_PROMPT = (
    "You are a grading agent for a telecom-domain RAG system. "
    "Grade the provided input on a scale of 0.0 to 1.0 and emit a "
    "JSON object with two keys:\n"
    '  "score": a float in [0.0, 1.0]\n'
    '  "reason": a short string explaining the score\n\n'
    "Respond with ONLY the JSON object, no prose, no markdown fences."
)


_RELEVANCE_PROMPT = (
    "Grade the RELEVANCE of the retrieved context to the user's query.\n\n"
    "Query: {query}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Return your grade."
)


_FAITHFULNESS_PROMPT = (
    "Grade the FAITHFULNESS of the answer to the retrieved context. "
    "A faithful answer contains ONLY facts supported by the context; "
    "any hallucinated or external facts should lower the score.\n\n"
    "Retrieved context:\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Return your grade."
)


_COMPLETENESS_PROMPT = (
    "Grade the COMPLETENESS of the answer to the user's query. "
    "A complete answer addresses every part of the query.\n\n"
    "Query: {query}\n\n"
    "Answer: {answer}\n\n"
    "Return your grade."
)


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Render the retrieved chunks as a numbered text block.

    Mirrors :func:`RetrievalAgent._build_context_block` (kept private
    here to avoid coupling the validation agent to the retrieval
    agent's internals — tests assert the formatted output is what the
    LLM sees).
    """
    blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown.pdf") or "unknown.pdf"
        page_number = chunk.get("page_number", 0) or 0
        text = chunk.get("text", "") or ""
        blocks.append(f"[{idx}] {source_file} p.{page_number}\n{text}")
    return "\n\n".join(blocks)


def _parse_grade(parsed: Dict[str, Any], model_cls: type) -> Any:
    """Validate ``parsed`` against one of the Pydantic grade models.

    Raises :class:`ValueError` on validation failure so the graph's
    generic except clause catches it uniformly.
    """
    try:
        return model_cls(**parsed)
    except ValidationError as exc:
        raise ValueError(
            f"LLM response did not validate as {model_cls.__name__}: "
            f"{parsed!r} (original error: {exc})"
        ) from exc


class ValidationAgent:
    """Three-dimension answer grader.

    Stateless; safe to construct once at module scope.
    """

    @_RETRYING
    def grade_relevance(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> RelevanceGrade:
        """Grade the relevance of ``chunks`` to ``query``."""
        context = _format_chunks(chunks)
        prompt = _RELEVANCE_PROMPT.format(query=query, context=context)
        # Goes through the module attribute so test seams flow through.
        # Issue #19: attribute cost to the VALIDATION agent.
        result = llm.chat_with_fallback(
            prompt=prompt,
            system=_BASE_SYSTEM_PROMPT,
            temperature=0.0,
            agent=Agent.VALIDATION,
        )
        parsed = parse_json_text(result.text)
        return _parse_grade(parsed, RelevanceGrade)

    @_RETRYING
    def grade_faithfulness(
        self, answer: str, chunks: List[Dict[str, Any]]
    ) -> FaithfulnessGrade:
        """Grade the faithfulness of ``answer`` to ``chunks``."""
        context = _format_chunks(chunks)
        prompt = _FAITHFULNESS_PROMPT.format(answer=answer, context=context)
        # Issue #19: attribute cost to the VALIDATION agent.
        result = llm.chat_with_fallback(
            prompt=prompt,
            system=_BASE_SYSTEM_PROMPT,
            temperature=0.0,
            agent=Agent.VALIDATION,
        )
        parsed = parse_json_text(result.text)
        return _parse_grade(parsed, FaithfulnessGrade)

    @_RETRYING
    def grade_completeness(
        self, query: str, answer: str
    ) -> CompletenessGrade:
        """Grade the completeness of ``answer`` w.r.t. ``query``."""
        prompt = _COMPLETENESS_PROMPT.format(query=query, answer=answer)
        # Issue #19: attribute cost to the VALIDATION agent.
        result = llm.chat_with_fallback(
            prompt=prompt,
            system=_BASE_SYSTEM_PROMPT,
            temperature=0.0,
            agent=Agent.VALIDATION,
        )
        parsed = parse_json_text(result.text)
        return _parse_grade(parsed, CompletenessGrade)

    def grade_all(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        answer: str,
    ) -> Dict[str, Any]:
        """Grade all three dimensions and return a dict with overall_score.

        Returns a dict shaped like::

            {
                "relevance": float,
                "faithfulness": float,
                "completeness": float,
                "overall_score": float,  # mean of the three
                "reasons": {<dim>: <str>},
            }

        Each of the three grading methods has its own retry decorator;
        :meth:`grade_all` does NOT add an outer retry (that would
        compound retries and could trigger API rate limits). A single
        failure raises and the graph node handles it.
        """
        rel = self.grade_relevance(query, chunks)
        faith = self.grade_faithfulness(answer, chunks)
        comp = self.grade_completeness(query, answer)
        overall = (rel.score + faith.score + comp.score) / 3.0
        return {
            "relevance": rel.score,
            "faithfulness": faith.score,
            "completeness": comp.score,
            "overall_score": overall,
            "reasons": {
                "relevance": rel.reason,
                "faithfulness": faith.reason,
                "completeness": comp.reason,
            },
        }
