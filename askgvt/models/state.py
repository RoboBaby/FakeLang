"""LangGraph state schema for AskGVT."""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from askgvt.models.evidence import (
    RetrievalHit,
    FusedEvidence,
    AnswerMetadata,
)


class AskGVTState(TypedDict):
    """Complete state schema for the LangGraph agent."""
    # Input
    user_query: str
    normalized_query: Optional[str]

    # Classification (aligned with CSV taxonomy)
    category: Optional[str]
    capability: Optional[str]
    intent: Optional[str]
    evidence_required: Optional[bool]
    difficulty: Optional[str]

    question_type: Optional[str]
    needs_visual_evidence: Optional[bool]
    needs_freshness: Optional[bool]
    time_horizon: Optional[str]
    strict_askgvt_only: Optional[bool]
    explicit_video_ids: List[str]

    # Retrieval planning & results
    retrieval_plan: Optional[Dict[str, Any]]
    narrative_hits: List[RetrievalHit]
    transcript_hits: List[RetrievalHit]
    image_hits: List[RetrievalHit]
    fused_evidence: Optional[FusedEvidence]

    # Reasoning
    background_answer: Optional[str]
    final_answer: Optional[str]
    answer_metadata: Optional[AnswerMetadata]

    # Control flow
    retry_count: int

    # Deep Research fields
    requires_deep_research: Optional[bool]
    plan_steps: Optional[List[str]]
    past_steps: Optional[List[Dict[str, str]]]
    visit_count: int
