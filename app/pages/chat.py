"""Chat page (Issue #7 AC8).

Reads from the persisted Chroma collection via the LangGraph
multi-agent StateGraph (Issue #7) and renders the answer text plus
a numbered citation list. Supports the human-approval gate: when the
graph pauses at ``interrupt_before=["human_approval"]``, the chat
page renders an "Awaiting approval" UI with an Approve button. On
click, the chat page calls ``graph.invoke(None, config)`` to resume.

Thread model
------------

``st.session_state.thread_id`` holds the conversation thread_id for
the lifetime of the Streamlit session. The graph checkpointer keys
state off this thread_id; two tabs in the same session share the
same thread_id (per Issue #7 concurrency invariant 4).

Graph lifecycle
---------------

The compiled graph is held in ``st.session_state.graph`` (cached via
``@st.cache_resource``). On ingest from the upload page, the chat
page's vectorstore handle may need to be re-bound; in MVP we rebuild
the graph when the chat page re-mounts.

Approval flow
-------------

1. User asks a question -> chat page calls ``graph.invoke(state, config)``
2. Graph pauses at ``human_approval``; chat page reads
   ``graph.get_state(config)`` to render the candidate answer + grades.
3. Chat page renders an "Approve" button.
4. On click, ``graph.invoke(None, config)`` resumes; graph terminates.
5. Chat page reads final state to render the approved answer.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from telecom_rag.graphs.telecom_graph import build_graph
from telecom_rag.llm import LLMAvailabilityError
from telecom_rag.memory.checkpointer import get_checkpointer
from telecom_rag.tools.retrieval import format_sources, get_vectorstore


@st.cache_resource
def _get_graph():
    """Build the compiled graph once per Streamlit session.

    Holds an in-memory SqliteSaver for the session. Production uses
    ``"./checkpoints.db"`` so conversation state persists across
    browser refreshes; for MVP the in-memory checkpointer is fine
    (Issue #7 wave 1 scope).

    ``@st.cache_resource`` (not ``@st.cache_data``) is required
    because the compiled graph object is not pickleable.  The
    checkpointer's SQLite connection must outlive the function body
    (LangGraph reads it on every ``invoke``), so we own it here
    rather than scoping it to a ``with`` block — the in-memory
    connection is reclaimed when the Streamlit process exits.
    """
    vs = get_vectorstore()
    cp = get_checkpointer(":memory:")
    graph = build_graph(collection=vs, checkpointer=cp)
    return graph


def _render_citations(citations: List[Dict[str, Any]]) -> None:
    """Project state-shaped citation dicts into the format
    :func:`format_sources` expects and render them."""
    chunk_dicts = [
        {
            "source_file": c.get("source_file", ""),
            "page_number": c.get("page_number", 0),
            "chunk_index": c.get("chunk_index", 0),
            "text": c.get("snippet", ""),
            "doc_category": "",
            "score": None,
        }
        for c in citations
    ]
    sources_md = format_sources(chunk_dicts)
    if sources_md:
        st.markdown(sources_md)


def _render_validation_panel(validation: Dict[str, Any]) -> None:
    """Render the validation grades as a small caption under the answer.

    Three dimensions (relevance, faithfulness, completeness) plus the
    overall score. Helps the power-user (Network Architect persona)
    audit the answer before approving it (README §2 persona 3).
    """
    if not validation:
        return
    rel = validation.get("relevance", 0.0)
    faith = validation.get("faithfulness", 0.0)
    comp = validation.get("completeness", 0.0)
    overall = (rel + faith + comp) / 3.0
    st.caption(
        f"Validation — relevance: {rel:.2f} · "
        f"faithfulness: {faith:.2f} · "
        f"completeness: {comp:.2f} · "
        f"overall: {overall:.2f}"
    )


# ---- Session-scoped chat history ---------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # Each new Streamlit session starts a fresh thread. Re-running the
    # chat page (e.g. on rerender) preserves the thread_id so the
    # conversation stays coherent.
    import uuid

    st.session_state.thread_id = f"user_{uuid.uuid4().hex[:8]}"

if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = None  # None or the pending state dict


# Render prior turns so the conversation persists across Streamlit reruns.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ---------------------------------------------------------

prompt = st.chat_input("Ask about telecom docs...")
if prompt:
    # Echo the user turn.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        graph = _get_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        graph.invoke({"query": prompt, "thread_id": st.session_state.thread_id}, config)
        snapshot = graph.get_state(config)
        state_values = snapshot.values
        # Pause at human_approval -> snapshot.next is ("human_approval",)
        # If the graph already terminated (refuse / validate_only),
        # snapshot.next is ().
        if snapshot.next and "human_approval" in snapshot.next:
            # Render candidate answer + grades + Approve button.
            with st.chat_message("assistant"):
                st.markdown(state_values.get("answer_text", ""))
                _render_validation_panel(state_values.get("validation", {}))
                _render_citations(state_values.get("citations", []))
            st.session_state.awaiting_approval = state_values
        else:
            # Graph terminated without pause (refuse / validate_only).
            answer_text = state_values.get("answer_text", "")
            with st.chat_message("assistant"):
                st.markdown(answer_text)
                _render_citations(state_values.get("citations", []))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text}
            )
    except LLMAvailabilityError:
        st.error("LLM service unavailable. Check your API keys in the .env file.")


# ---- Approval gate ------------------------------------------------------

if st.session_state.awaiting_approval is not None:
    pending = st.session_state.awaiting_approval
    if st.button("Approve", key=f"approve_{len(st.session_state.messages)}"):
        try:
            graph = _get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            graph.invoke(None, config)
            final_snapshot = graph.get_state(config)
            answer_text = final_snapshot.values.get("answer_text", "")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text}
            )
            st.session_state.awaiting_approval = None
            st.rerun()
        except LLMAvailabilityError:
            st.error("LLM service unavailable during approval resume.")