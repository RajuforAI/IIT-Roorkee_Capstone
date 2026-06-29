"""Pydantic / dataclass schemas for the Telecom RAG system.

Application logic is intentionally omitted in this skeleton.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class DocCategory(str, Enum):
    """Valid document categories for the telecom corpus."""

    SOP = "sop"
    MANUAL = "manual"
    SPEC = "spec"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL = "general"


class TelecomMetadata(BaseModel):
    """Metadata attached to every chunk stored in ChromaDB."""

    source_file: str
    doc_category: DocCategory
    title: str
    page_number: int
    chunk_index: int
    total_chunks: int
    equipment_type: Optional[str] = None
    protocol: Optional[str] = None
    vendor: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    created_at: str


class RelevanceGrade(BaseModel):
    """Graded relevance of a retrieved chunk to the user's query."""

    score: float
    reason: str


class FaithfulnessGrade(BaseModel):
    """Graded faithfulness of a generated answer to the retrieved context."""

    score: float
    reason: str


class CompletenessGrade(BaseModel):
    """Graded completeness of a generated answer to the user's question."""

    score: float
    reason: str


class RouterDecision(BaseModel):
    """Router output: which graph branch should handle the query."""

    route: str  # one of: 'qa', 'summarize', 'validate_only', 'refuse'
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Issue #7 — LangGraph state
# ---------------------------------------------------------------------------


class TelecomState(TypedDict, total=False):
    """State flowing through the LangGraph StateGraph (Issue #7).

    All fields are optional (``total=False``); nodes populate only what
    they need. The chat page seeds only ``query`` + ``thread_id``; all
    other fields are written by nodes as the graph runs.

    The TypedDict is intentionally permissive: a node that doesn't run
    for a given route leaves its fields unset. Tests assert on the
    *presence* of fields, not their absence.
    """

    # ---- Set by caller -------------------------------------------------
    query: str
    thread_id: str

    # ---- Populated by route_node --------------------------------------
    route: str  # 'qa' | 'summarize' | 'validate_only' | 'refuse'
    route_confidence: float

    # ---- Populated by retrieve_node (qa + summarize routes) -----------
    retrieved_chunks: List[Dict[str, Any]]  # shape RetrievalAgent.run consumes
    citations: List[Dict[str, Any]]  # {source_file, page_number, chunk_index, snippet}

    # ---- Populated by validate_node (qa route only) -------------------
    validation: Dict[str, Any]  # {relevance, faithfulness, completeness, reason}
    overall_score: float  # mean of the three; refusal threshold 0.6 (README §7.3)

    # ---- Populated by generate_node (qa) or summarize_node ------------
    answer_text: str

    # ---- Populated by human_approval_node (no-op; framework writes) ---
    human_approved: bool

    # ---- Graph-level error tracking (README §11.2) --------------------
    error: Optional[str]
    retry_count: int
