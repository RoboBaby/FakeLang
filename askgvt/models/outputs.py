"""Pydantic models for LLM structured outputs."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ClassificationOutput(BaseModel):
    """Structured output for query classification."""
    category: str = Field(
        description="Domain category: cooking, fitness, fashion, tech, DIY, etc."
    )
    capability: Literal[
        "temporal_reasoning", "frame_detection", "pose_detection",
        "motion_analysis", "narrative_segmentation", "audio_visual_alignment",
        "creator_enterprise_analytics"
    ] = Field(description="Required visual understanding capability")
    intent: Literal[
        "trend", "find", "count", "compare", "explain",
        "sequence", "verify", "correlate", "detect_change"
    ] = Field(description="User's intent")
    evidence_required: bool = Field(
        description="Whether answer must be grounded in video evidence"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Complexity of the question"
    )
    question_type: Literal[
        "trend", "how_to", "comparison", "analytics",
        "factoid", "content_search", "moderation", "other"
    ] = Field(description="High-level question category")
    needs_visual_evidence: bool = Field(
        description="Whether answer requires AskGVT corpus vs general knowledge"
    )
    needs_freshness: bool = Field(
        description="Whether recency/trending matters"
    )
    time_horizon: Literal[
        "this_week", "last_month", "last_year", "all_time", "unspecified"
    ] = Field(description="Time constraint for the query")
    strict_askgvt_only: bool = Field(
        description="Must answer only from video corpus, no external knowledge"
    )
    explicit_video_ids: List[str] = Field(
        default_factory=list,
        description="Specific video IDs mentioned in query"
    )
    requires_deep_research: bool = Field(
        default=False,
        description="Whether this query needs multi-step investigation (complex comparisons, multi-part questions, analytical queries)"
    )


class QueryVariant(BaseModel):
    """A query variant for a specific index."""
    target: Literal["narrative", "transcript", "image"]
    query: str
    weight: float = 1.0


class RetrievalFilters(BaseModel):
    """Filters to apply during retrieval."""
    category: Optional[str] = None
    time_horizon: Optional[str] = None
    creator_ids: List[str] = Field(default_factory=list)
    video_ids: List[str] = Field(default_factory=list)
    language: str = "en"


class RetrievalPlanOutput(BaseModel):
    """Structured output for retrieval planning."""
    primary_sources: List[Literal["narrative", "transcript", "image"]]
    secondary_sources: List[Literal["narrative", "transcript", "image"]]
    n_results_per_source: int = 40
    filters: RetrievalFilters
    query_variants: List[QueryVariant]
    aggregation_strategy: str = "group_by_video_then_segment"
    max_total_hits: int = 200


class CriticOutput(BaseModel):
    """Structured output for answer criticism."""
    is_satisfactory: bool
    reasons: List[str]
    should_retry_retrieval: bool
    suggested_retrieval_adjustments: Dict[str, Any] = Field(default_factory=dict)
