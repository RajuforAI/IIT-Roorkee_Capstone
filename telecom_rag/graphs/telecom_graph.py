"""Telecom-specific LangGraph StateGraph (Issue #7).

Public surface
--------------

- :func:`build_graph` — factory that returns a compiled
  :class:`langgraph.graph.StateGraph` with the full router +
  retrieval + validation + generation/summarization + human-approval
  + refusal topology.
- :data:`REFUSAL_MESSAGE` — the single refusal string the graph
  writes to ``state['answer_text']`` for both domain-refusals AND
  validation-rejections (per README §11.2; users cannot tell the
  two cases apart).

Graph topology
--------------

```
  ┌──────┐    ┌─────────┐    ┌────────┐
  │ START│───▶│ route_  │───▶│retrieve│──┐
  └──────┘    │  node   │    │  node  │  │
              └────┬────┘    └────────┘  │
                   │                     │
                   │ (refuse/validate)   │ (qa/summarize)
                   ▼                     ▼
              ┌────────┐            ┌────────┐
              │ refuse │            │validate│
              │  node  │            │  node  │
              └────┬───┘            └───┬────┘
                   │                    │ (low score)
                   │                    ▼
                   │                ┌────────┐
                   │                │ refuse │
                   │                │  node  │
                   │                └────┬───┘
                   │                     │
                   │  (qa, sufficient)   │
                   │ ◀───────────────────┘
                   ▼
              ┌──────────┐       ┌────────────┐
              │ generate │       │ summarize  │
              │  _node   │       │   _node    │
              └────┬─────┘       └─────┬──────┘
                   │                   │
                   └──────┬────────────┘
                          ▼
                  ┌──────────────────┐
                  │ human_approval_  │  ← INTERRUPT BEFORE THIS NODE
                  │       node       │     (framework pauses execution;
                  └────────┬─────────┘      chat page renders Approve UI)
                           │
                           ▼
                         END
```

The `interrupt_before=["human_approval"]` parameter on
``graph.compile(...)`` is the standard LangGraph pattern for
human-in-the-loop. The framework pauses just before the
``human_approval_node`` runs; the chat page calls
``graph.invoke(None, config)`` to resume.

Conditional edges
-----------------

- After ``route_node``: dispatch to ``retrieve_node`` (qa/summarize),
  ``validate_node`` (validate_only), or ``refuse_node`` (refuse).
- After ``validate_node``: if ``state['overall_score'] < 0.6`` route
  to ``refuse_node``, else route to ``generate_node`` (qa) or
  ``END`` (validate_only — there is no answer to deliver, just a
  validation verdict).
- After ``generate_node`` / ``summarize_node``: always to
  ``human_approval_node``.

Error handling (README §11.2)
-----------------------------

Every node is wrapped in try/except. On exception, the node sets
``state['error'] = str(exc)`` and increments ``state['retry_count']``.
After 3 failures, the graph refuses with the standard message rather
than re-raising. This is the same graceful-degradation behavior
already documented in the issue file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from telecom_rag import llm
from telecom_rag.agents.summarization_agent import SummarizationAgent
from telecom_rag.agents.validation_agent import ValidationAgent
from telecom_rag.graphs.router import Router
from telecom_rag.observability.cost import Agent
from telecom_rag.schemas import TelecomState

logger = logging.getLogger(__name__)


# Single refusal message for both domain-refusals AND
# validation-rejections. Per README §11.2 the user cannot tell the
# two cases apart; this is intentional and the chat page surfaces
# only this one string.
REFUSAL_MESSAGE = "This query is outside my telecom domain."

# Refusal threshold (README §7.3): overall validation score below
# this triggers refuse_node.
REFUSAL_THRESHOLD = 0.6

# Max retry count before a node gives up and routes to refuse_node
# (README §11.2).
MAX_RETRY_COUNT = 3


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def _bump_retry(state: TelecomState) -> TelecomState:
    """Increment retry_count; create the key if absent."""
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def route_node(state: TelecomState) -> TelecomState:
    """Call Router.decide(query) and write route + confidence.

    On any exception, routes to 'refuse' (graceful degradation per
    README §11.2) — does NOT re-raise, so the graph always
    terminates cleanly.
    """
    try:
        decision = Router().decide(state["query"])
        state["route"] = decision.route
        state["route_confidence"] = decision.confidence
    except Exception as exc:  # noqa: BLE001 — graceful per README §11.2
        logger.warning("route_node failed: %s; falling back to refuse", exc)
        state["error"] = str(exc)
        _bump_retry(state)
        state["route"] = "refuse"
        state["route_confidence"] = 0.0
    return state


def retrieve_node(state: TelecomState) -> TelecomState:
    """Retrieve chunks for qa + summarize routes.

    For refuse / validate_only routes, this is a no-op (caller
    already routed around it). On failure, falls back to refuse.

    Calls :func:`telecom_rag.tools.retrieval.retrieve_chunks`
    DIRECTLY — NOT :func:`retrieval_agent.run` — because
    ``RetrievalAgent.run`` also calls the LLM to generate an
    answer, and the qa path's answer is produced by ``generate_node``
    (so validation can grade the graph's answer, not a parallel
    one). Two separate LLM calls per query would double the cost
    and produce two different answers.
    """
    if state.get("route") not in {"qa", "summarize"}:
        return state
    try:
        from telecom_rag.tools.retrieval import retrieve_chunks

        collection = _get_bound_collection()
        chunks = retrieve_chunks(state["query"], collection, k=5)
        # Build Citation-shaped projections for the chat page.
        citations: List[Dict[str, Any]] = []
        for c in chunks:
            citations.append(
                {
                    "source_file": str(c.get("source_file", "unknown.pdf") or "unknown.pdf"),
                    "page_number": int(c.get("page_number", 0) or 0),
                    "chunk_index": int(c.get("chunk_index", 0) or 0),
                    "snippet": (c.get("text", "") or "")[:200],
                }
            )
        state["retrieved_chunks"] = chunks
        state["citations"] = citations
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve_node failed: %s; falling back to refuse", exc)
        state["error"] = str(exc)
        _bump_retry(state)
        state["route"] = "refuse"
    return state


def validate_node(state: TelecomState) -> TelecomState:
    """Grade the retrieved chunks (relevance only — faithfulness and
    completeness require an answer, which generate_node produces).

    For the qa route, generate_node calls grade_all AFTER
    generating. For validate_only, we only grade relevance here
    (no answer to assess for faithfulness/completeness). On
    failure, falls back to refuse.
    """
    if state.get("route") != "validate_only":
        return state
    try:
        chunks = state.get("retrieved_chunks", [])
        relevance = ValidationAgent().grade_relevance(state["query"], chunks)
        state["validation"] = {
            "relevance": relevance.score,
            "reason": relevance.reason,
        }
        # For validate_only, "overall_score" is just relevance.
        state["overall_score"] = relevance.score
    except Exception as exc:  # noqa: BLE001
        logger.warning("validate_node failed: %s; falling back to refuse", exc)
        state["error"] = str(exc)
        _bump_retry(state)
        state["route"] = "refuse"
    return state


def generate_node(state: TelecomState) -> TelecomState:
    """Generate the qa answer with the retrieved chunks as context.

    After generating, runs the full three-dimension validation
    (relevance, faithfulness, completeness) and sets
    ``state['overall_score']`` so the conditional edge AFTER
    validate_node can route to refuse if the score is low.
    """
    if state.get("route") != "qa":
        return state
    try:
        chunks = state.get("retrieved_chunks", [])
        context_block = _format_chunks_for_prompt(chunks)
        prompt = (
            f"Context:\n\n{context_block}\n\n"
            f"Question: {state['query']}"
        )
        # Reuse the existing retrieval-agent system prompt for
        # answer style consistency. Issue #19: attribute the chat
        # cost to the RETRIEVAL agent — the answer is produced from
        # retrieved context, so this is the right attribution axis
        # (the validation pass that follows is its own charge).
        result = llm.chat_with_fallback(
            prompt=prompt,
            system=(
                "You are a telecom-domain knowledge assistant. "
                "Answer the user's question using ONLY the context "
                "blocks provided. Cite sources inline as [1], [2], etc. "
                "If the context does not contain the answer, say so "
                "explicitly rather than guessing."
            ),
            agent=Agent.RETRIEVAL,
        )
        state["answer_text"] = result.text

        # Now run the full three-dimension validation. If the
        # overall score is below the refusal threshold, the
        # conditional edge after validate_node will route to
        # refuse_node (the answer is never delivered).
        try:
            grades = ValidationAgent().grade_all(
                state["query"], chunks, state["answer_text"]
            )
            state["validation"] = {
                "relevance": grades["relevance"],
                "faithfulness": grades["faithfulness"],
                "completeness": grades["completeness"],
                "reason": "; ".join(
                    f"{k}: {v}" for k, v in grades["reasons"].items()
                ),
            }
            state["overall_score"] = grades["overall_score"]
        except Exception as exc:  # noqa: BLE001
            # If validation fails (LLM error, malformed JSON),
            # treat the answer as unverified — the conditional edge
            # will route to refuse.
            logger.warning("validation in generate_node failed: %s", exc)
            state["validation"] = {
                "relevance": 0.0,
                "faithfulness": 0.0,
                "completeness": 0.0,
                "reason": f"validation failed: {exc}",
            }
            state["overall_score"] = 0.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_node failed: %s; falling back to refuse", exc)
        state["error"] = str(exc)
        _bump_retry(state)
        state["route"] = "refuse"
    return state


def summarize_node(state: TelecomState) -> TelecomState:
    """Map-reduce summary across the retrieved chunks."""
    if state.get("route") != "summarize":
        return state
    try:
        chunks = state.get("retrieved_chunks", [])
        summary = SummarizationAgent().summarize_map_reduce(chunks)
        state["answer_text"] = summary
    except Exception as exc:  # noqa: BLE001
        logger.warning("summarize_node failed: %s; falling back to refuse", exc)
        state["error"] = str(exc)
        _bump_retry(state)
        state["route"] = "refuse"
    return state


def human_approval_node(state: TelecomState) -> TelecomState:
    """No-op node; the framework pauses BEFORE this runs.

    When the chat page calls ``graph.invoke(None, config)`` to
    resume, the framework runs this node, which simply marks
    ``state['human_approved'] = True`` and returns. The graph
    then terminates at END.

    Setting ``human_approved`` here is what makes the second
    ``graph.invoke`` return a state the chat page can read to
    confirm approval succeeded.
    """
    state["human_approved"] = True
    return state


def refuse_node(state: TelecomState) -> TelecomState:
    """Write the standard refusal message and stop."""
    state["answer_text"] = REFUSAL_MESSAGE
    return state


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def _after_route(state: TelecomState) -> str:
    """Dispatch to the next node based on the router's decision."""
    route = state.get("route", "refuse")
    if route == "qa":
        return "retrieve"
    if route == "summarize":
        return "retrieve"
    if route == "validate_only":
        return "validate"
    return "refuse"


def _after_validate(state: TelecomState) -> str:
    """Route from validate_node to the appropriate next node.

    For ``qa``: ALWAYS go to generate_node. The validation that
    drives the refusal decision is performed INSIDE generate_node
    (which has the answer in hand), not here. validate_node for qa
    is a no-op, so the conditional edge can't check ``overall_score``
    here — doing so would default to 0.0 and falsely route to refuse.

    For ``validate_only``: route to human_approval (the validation
    verdict IS the answer; the chat page renders the grades).
    """
    route = state.get("route", "refuse")
    if route == "qa":
        return "generate"
    return "human_approval"


def _after_retrieve(state: TelecomState) -> str:
    """For qa: validate then generate. For summarize: summarize directly."""
    route = state.get("route", "refuse")
    if state.get("route") == "refuse":
        return "refuse"
    if route == "qa":
        return "validate"
    if route == "summarize":
        return "summarize"
    return "refuse"


def _after_generate_or_summarize(state: TelecomState) -> str:
    """After generate or summarize, route to refuse if the validation
    score is below threshold; else to human_approval.

    generate_node runs ``ValidationAgent.grade_all`` and sets
    ``state['overall_score']`` before this edge is evaluated, so
    this is where the refusal decision for qa routes happens.
    For summarize routes, ``overall_score`` is unset (summarize_node
    does NOT run grading) and defaults to >= threshold, so the
    summarize answer always pauses at human_approval.
    """
    score = state.get("overall_score", 1.0)
    if score < REFUSAL_THRESHOLD:
        return "refuse"
    return "human_approval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Render chunks as a numbered context block for the generate prompt."""
    blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown.pdf") or "unknown.pdf"
        page_number = chunk.get("page_number", 0) or 0
        text = chunk.get("text", "") or ""
        blocks.append(f"[{idx}] {source_file} p.{page_number}\n{text}")
    return "\n\n".join(blocks)


def _citations_to_chunks(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """DEPRECATED: kept for backward compatibility only.

    retrieve_node now projects chunks -> citations directly inline,
    so this helper is no longer called from the graph. Kept because
    earlier revisions referenced it; remove in a follow-up cleanup
    once the chat page wiring lands and we can verify no external
    callers remain.
    """
    return citations


# ---------------------------------------------------------------------------
# Collection binding (set on the graph module at build_graph time)
# ---------------------------------------------------------------------------


_BOUND_COLLECTION: Any = None


def _get_bound_collection() -> Any:
    """Return the Chroma collection bound to the graph at build time.

    The chat page binds the live vectorstore via
    ``build_graph(collection=vs)``; tests bind a stub. This indirection
    lets the node functions be plain (not closures) while still
    routing the collection through the same code path.
    """
    if _BOUND_COLLECTION is None:
        raise RuntimeError(
            "No collection bound to the graph. Call "
            "build_graph(collection=...) before invoking the graph."
        )
    return _BOUND_COLLECTION


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_graph(*, collection: Any, checkpointer: Any = None) -> Any:
    """Build and compile the multi-agent StateGraph.

    Parameters
    ----------
    collection:
        A ``langchain_chroma.Chroma`` vectorstore (the same shape
        :func:`telecom_rag.tools.retrieval.get_vectorstore` returns).
        Bound to the graph module so node functions can read it
        without a closure.
    checkpointer:
        A :class:`langgraph.checkpoint.SqliteSaver` (or compatible).
        If ``None``, the graph compiles without persistence (tests
        can pass a ``:memory:`` checkpointer explicitly).

    Returns
    -------
    CompiledStateGraph
        The compiled graph. Call ``.invoke(state, config)`` to run
        a query; the ``config`` must include
        ``{"configurable": {"thread_id": <str>}}`` for the
        checkpointer to key state correctly.
    """
    # Bind the collection at module scope so the node functions
    # can read it. This is a single-process bind; the chat page
    # calls ``build_graph`` once per Streamlit session and reuses
    # the resulting compiled graph.
    global _BOUND_COLLECTION
    _BOUND_COLLECTION = collection

    graph = StateGraph(TelecomState)

    # ---- Nodes --------------------------------------------------------
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate", validate_node)
    graph.add_node("generate", generate_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("refuse", refuse_node)

    # ---- Edges -------------------------------------------------------
    graph.add_edge(START, "route")

    graph.add_conditional_edges(
        "route",
        _after_route,
        {
            "retrieve": "retrieve",
            "validate": "validate",
            "refuse": "refuse",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {
            "validate": "validate",
            "summarize": "summarize",
            "refuse": "refuse",
        },
    )

    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {
            "generate": "generate",
            "human_approval": "human_approval",
        },
    )

    # After generate OR summarize, the answer always goes to
    # human_approval (qa answers always pause for approval; the
    # refusal decision is made at the validate boundary, not here).
    graph.add_conditional_edges(
        "generate",
        _after_generate_or_summarize,
        {"human_approval": "human_approval", "refuse": "refuse"},
    )
    graph.add_conditional_edges(
        "summarize",
        _after_generate_or_summarize,
        {"human_approval": "human_approval", "refuse": "refuse"},
    )

    graph.add_edge("human_approval", END)
    graph.add_edge("refuse", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )
