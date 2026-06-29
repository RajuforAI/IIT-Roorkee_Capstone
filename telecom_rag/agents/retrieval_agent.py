"""Retrieval agent for the Streamlit chat page (Issue #5 AC3).

Public surface
--------------

- :class:`Citation` — one chunk projected for the chat UI; ``snippet``
  is the first 200 characters of the chunk text for hover-preview.
- :class:`AnswerWithCitations` — the structured result returned by
  :func:`run`; ``answer_text`` is what the chat page renders, and
  ``citations`` are the numbered references the page turns into a
  citation list under the answer.
- :func:`run` — retrieve top-k chunks for the query, build a
  context-grounded prompt, call :func:`chat_with_fallback`, and
  project the retrieved chunks to :class:`Citation` objects.

Empty-hits sentinel
-------------------

When retrieval returns no chunks, :func:`run` returns
``AnswerWithCitations`` with ``answer_text="No relevant documents
found in the indexed corpus."``, ``citations=[]``, and empty
``provider`` / ``model`` strings. The Streamlit chat page relies on
this exact sentinel to render a friendly "no results" message instead
of crashing on an empty citation list.

LLM-failure propagation
-----------------------

If both configured LLM providers fail, the LLM layer's
:func:`chat_with_fallback` raises :class:`LLMAvailabilityError`.
:func:`run` does NOT catch it — it propagates so the chat page (and
the AC7 test) can render an explicit error. Silently swallowing the
failure would leave the user staring at an empty answer with no
indication of what went wrong.

Implementation note: :func:`run` calls the LLM through the
``telecom_rag.llm`` module attribute (``llm.chat_with_fallback``)
rather than a local ``from ... import chat_with_fallback`` binding.
A local ``from`` import would cache the function in this module's
namespace at import time, making source-module test stubs
(``monkeypatch.setattr(telecom_rag.llm, "chat_with_fallback", ...)``)
invisible to the agent without ``importlib.reload``. Going through
the module attribute keeps the agent looking up the function on the
source module at call time, so test seams flow through directly.

Projection order
----------------

:func:`run` projects the retrieved chunk dicts to :class:`Citation`
objects in the SAME ORDER they were returned by
:func:`telecom_rag.tools.retrieval.retrieve_chunks`. The
context-grounded prompt numbers them ``[1]``, ``[2]``, ... in that
order, and the system prompt tells the model to cite sources inline
using that convention. Therefore ``answer_text`` ``[1]`` corresponds
to ``citations[0]``, ``[2]`` to ``citations[1]``, etc. The chat page
relies on this positional mapping to render the answer and its
citations together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

# Re-exported so callers can do:
#     from telecom_rag.agents.retrieval_agent import LLMAvailabilityError
# without reaching into the LLM module. The chat page handles this
# exception explicitly (renders an error message instead of an empty
# answer). noqa: F401 keeps ruff happy if the import would otherwise
# be flagged as unused at module scope.
from telecom_rag.llm import LLMAvailabilityError  # noqa: F401
# Imported as a module (not a function) so the agent looks up
# ``chat_with_fallback`` on the source module at call time. This keeps
# source-module test stubs (e.g. ``monkeypatch.setattr(telecom_rag.llm,
# "chat_with_fallback", ...)``) visible to the agent without needing
# ``importlib.reload(agent)``. See the module docstring.
from telecom_rag import llm
from telecom_rag.observability.cost import Agent

from telecom_rag.tools.retrieval import retrieve_chunks

# The sentinel string the chat page matches on. Defined as a module
# constant so the test and the agent can't drift apart silently.
EMPTY_HITS_SENTINEL = "No relevant documents found in the indexed corpus."

# Maximum snippet length stored on each Citation. The data contract
# pins this at 200 chars for hover preview in the chat UI.
_SNIPPET_MAX_CHARS = 200

# System prompt instructing the model to answer using ONLY the
# provided context and to cite sources inline as [1], [2], etc.
# The rendering (turning [1] into a clickable citation) is the chat
# page's job; the model is told the citation convention only.
_SYSTEM_PROMPT = (
    "You are a telecom-domain knowledge assistant. Answer the user's "
    "question using ONLY the context blocks provided below. Each block "
    "is numbered in square brackets, e.g. [1], [2]. When you use "
    "information from a block, cite the block number inline in your "
    "answer (e.g. 'per [1]' or 'see [2]'). If the context does not "
    "contain the answer, say so explicitly rather than guessing."
)


@dataclass(frozen=True)
class Citation:
    """One retrieved chunk projected for the chat UI.

    ``snippet`` is the first 200 characters of the chunk text, used
    for hover preview in the chat page. The chat page maps
    ``citations[i]`` to the inline marker ``[i+1]`` in ``answer_text``.
    """

    source_file: str
    page_number: int
    chunk_index: int
    snippet: str  # first 200 chars of chunk text, for hover preview


@dataclass(frozen=True)
class AnswerWithCitations:
    """Structured result returned by :func:`run`.

    When retrieval returns no hits, ``answer_text`` is the
    :data:`EMPTY_HITS_SENTINEL`, ``citations`` is ``[]``, and both
    ``provider`` and ``model`` are empty strings. Otherwise
    ``answer_text`` is the LLM's response (which may itself contain
    inline citation markers like ``[1]``, ``[2]``) and ``citations``
    is the corresponding list of :class:`Citation` objects in
    retrieval order — so ``[1]`` in the answer text maps to
    ``citations[0]``.
    """

    answer_text: str
    citations: List[Citation]
    provider: str  # "openai" or "gemini"
    model: str


def run(query: str, collection, *, k: int = 5) -> AnswerWithCitations:
    """Retrieve top-k chunks for ``query`` and answer with citations.

    Parameters
    ----------
    query:
        The user's natural-language question.
    collection:
        A ``langchain_chroma.Chroma`` vectorstore — the same object
        :func:`telecom_rag.tools.retrieval.get_vectorstore` returns.
        Not a raw ``chromadb.api.models.Collection``; we go through
        the LangChain wrapper so the same embedding-fallback layer
        the ingestion pipeline uses handles the query embedding.
    k:
        Number of chunks to retrieve (default 5).

    Returns
    -------
    AnswerWithCitations
        See module docstring for the empty-hits sentinel and the
        projection-order contract.

    Raises
    ------
    LLMAvailabilityError
        Propagated from :func:`chat_with_fallback` when both
        configured LLM providers fail. NOT caught here.
    """
    chunks: List[dict] = retrieve_chunks(query, collection, k=k)

    if not chunks:
        return AnswerWithCitations(
            answer_text=EMPTY_HITS_SENTINEL,
            citations=[],
            provider="",
            model="",
        )

    context_block = _build_context_block(chunks)
    prompt = (
        f"Context:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    # llm.chat_with_fallback may raise LLMAvailabilityError if every
    # configured provider fails; we intentionally do NOT catch it
    # so the chat page can render the failure explicitly. The call
    # goes through the ``llm`` module attribute (not a local binding)
    # so source-module test stubs flow through directly — see the
    # module docstring.
    chat_result: Any = llm.chat_with_fallback(
        prompt=prompt, system=_SYSTEM_PROMPT, agent=Agent.RETRIEVAL
    )

    citations = [_chunk_to_citation(c) for c in chunks]
    return AnswerWithCitations(
        answer_text=chat_result.text,
        citations=citations,
        provider=chat_result.provider,
        model=chat_result.model,
    )


def _build_context_block(chunks: List[dict]) -> str:
    """Render the chunks as a numbered context block for the prompt.

    Format::

        [1] <source_file> p.<page_number>
        <chunk text>

        [2] <source_file> p.<page_number>
        <chunk text>

    Numbering is 1-based and matches the order of ``chunks`` so that
    the inline markers the model emits (``[1]``, ``[2]``, ...) align
    positionally with ``citations[i]`` in the final
    ``AnswerWithCitations``.
    """
    blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown.pdf") or "unknown.pdf"
        page_number = chunk.get("page_number", 0) or 0
        text = chunk.get("text", "") or ""
        blocks.append(
            f"[{idx}] {source_file} p.{page_number}\n{text}"
        )
    return "\n\n".join(blocks)


def _chunk_to_citation(chunk: dict) -> Citation:
    """Project a retrieval chunk dict to a :class:`Citation`.

    ``snippet`` is the first 200 characters of ``chunk["text"]`` per
    the data contract; we coerce types defensively so a malformed
    metadata value in the index doesn't crash the chat page.
    """
    text = chunk.get("text", "") or ""
    snippet = text[:_SNIPPET_MAX_CHARS]
    return Citation(
        source_file=str(chunk.get("source_file", "unknown.pdf") or "unknown.pdf"),
        page_number=int(chunk.get("page_number", 0) or 0),
        chunk_index=int(chunk.get("chunk_index", 0) or 0),
        snippet=snippet,
    )