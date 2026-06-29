"""Router for the multi-agent LangGraph (Issue #7).

Public surface
--------------

- :class:`Router` — single class with one public method, :meth:`Router.decide`.
- :func:`parse_json_text` — internal JSON-extraction helper exposed at
  module level so tests can unit-test it directly without spinning up
  a Router instance.

Why a class (not a free function)
---------------------------------

The Router holds no state, but a class lets us swap it for a stub
deterministically in tests via ``monkeypatch.setattr(router, "Router",
StubRouter)`` (a free function would have to be patched at every
import site). The class is stateless so construction is cheap.

JSON-from-LLM contract
----------------------

We instruct the LLM (via the system prompt) to emit a single JSON
object with the keys ``{"route": <str>, "confidence": <float>,
"reason": <str>}``. ``parse_json_text`` extracts the JSON even if the
model wraps it in ```json ... ``` fences or surrounds it with prose.
We deliberately do NOT use the OpenAI ``response_format=json_object``
feature here because it would require touching the LLM seam and the
README does not require it — keep the change surface minimal.

If parsing fails, :meth:`Router.decide` raises :class:`ValueError`.
The graph wraps the node in try/except per README §11.2 and routes
to ``refuse_node`` after ``retry_count`` exhaustion. Tests assert
that a malformed LLM response raises (rather than silently defaulting
to "qa"), so a model regression is visible at test time, not in
production.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from pydantic import ValidationError

from telecom_rag import llm
from telecom_rag.observability.cost import Agent
from telecom_rag.schemas import RouterDecision


# System prompt instructs the model to emit a single JSON object
# with the three keys. We pin the schema in the prompt rather than
# relying on the model's training because we want the test to know
# exactly what shape ``parse_json_text`` will see.
_SYSTEM_PROMPT = (
    "You are a router for a telecom-domain knowledge assistant. "
    "Given a user's query, classify it into exactly one of these "
    "four routes and return a JSON object with three keys:\n"
    '  "route": one of "qa", "summarize", "validate_only", "refuse"\n'
    '  "confidence": a float in [0.0, 1.0]\n'
    '  "reason": a short string explaining the classification\n\n'
    "Definitions:\n"
    '  - "qa": the user wants a factual answer grounded in telecom docs.\n'
    '  - "summarize": the user wants a summary of one or more documents.\n'
    '  - "validate_only": the user wants to know whether a specific '
    "technical claim is supported by the corpus.\n"
    '  - "refuse": the query is outside the telecom domain '
    "(weather, cooking, general chitchat, etc.).\n\n"
    "Respond with ONLY the JSON object, no prose, no markdown fences."
)


# Match the first {...} block in the text, non-greedy. ``re.DOTALL``
# lets ``.`` span newlines so multi-line JSON objects are captured.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_json_text(text: str) -> Dict[str, Any]:
    """Extract a single JSON object from ``text`` and return it as a dict.

    Handles three common cases the LLM produces:
    1. Pure JSON: ``{"route": "qa", ...}``
    2. Fenced JSON: ```json\\n{"route": "qa", ...}\\n```
    3. JSON with surrounding prose: ``Sure, here you go: {...} hope this helps.``

    Raises :class:`ValueError` if no JSON object is found OR if the
    extracted text is not valid JSON. We deliberately do NOT swallow
    :class:`json.JSONDecodeError` and return an empty dict — silent
    failure here would cause the router to default to "qa" and hide
    a model regression from tests.
    """
    if not text:
        raise ValueError("LLM returned empty text; cannot parse JSON")

    # Strip ```json / ``` fences if present.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (with optional language tag).
        cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", cleaned)
        # Drop the closing fence.
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        raise ValueError(
            f"No JSON object found in LLM response: {text!r}"
        )

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response contains malformed JSON: {candidate!r} "
            f"(original error: {exc})"
        ) from exc


class Router:
    """Classify a user query into one of the four graph routes.

    Stateless; safe to construct once at module scope or per-call.
    """

    def decide(self, query: str) -> RouterDecision:
        """Return a :class:`RouterDecision` for ``query``.

        Calls :func:`telecom_rag.llm.chat_with_fallback` with a JSON
        system prompt, parses the response, and validates it as a
        :class:`RouterDecision`.

        Raises :class:`ValueError` on malformed LLM output. The
        LangGraph node wraps this in try/except per README §11.2.
        """
        # Goes through the module attribute (not a local binding) so
        # test seams (``monkeypatch.setattr(llm, "chat_with_fallback",
        # stub)``) flow through without ``importlib.reload``. Same
        # pattern as ``RetrievalAgent.run``.
        # Issue #19: attribute cost to the ROUTER agent.
        result = llm.chat_with_fallback(
            prompt=query,
            system=_SYSTEM_PROMPT,
            temperature=0.0,  # routing is deterministic; we want stable test outcomes
            agent=Agent.ROUTER,
        )
        parsed = parse_json_text(result.text)
        try:
            return RouterDecision(**parsed)
        except ValidationError as exc:
            # Re-raise as ValueError so the graph's generic except
            # clause in route_node catches it uniformly with the
            # "LLM returned malformed JSON" path.
            raise ValueError(
                f"LLM response did not validate as RouterDecision: {parsed!r} "
                f"(original error: {exc})"
            ) from exc
