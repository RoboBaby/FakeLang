"""Evidence and retrieval data structures."""

from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict


class RetrievalHit(TypedDict):
    """Single retrieval result from any index."""
    source: Literal["narrative", "transcript", "image"]
    video_id: str
    start_s: float
    end_s: float
    score: float
    text: str
    metadata: Dict[str, Any]


class EvidenceClip(TypedDict):
    """A clip selected as evidence for the answer."""
    video_id: str
    start_s: float
    end_s: float
    reason: str
    score: float
    metadata: Dict[str, Any]


class FusedEvidence(TypedDict):
    """Aggregated evidence from all sources."""
    num_videos: int
    num_segments: int
    categories: List[str]
    time_horizon: str
    key_patterns: List[str]
    disagreements: List[str]
    limitations: List[str]
    segments: List[Dict[str, Any]]


class AnswerMetadata(TypedDict):
    """Metadata about how the answer was generated."""
    used_sources: List[Literal["narrative", "transcript", "image", "llm_background"]]
    num_videos_considered: int
    evidence_clips: List[EvidenceClip]
    coverage: Literal["high", "medium", "low", "none"]
    limitations: List[str]
    critic: Optional[Dict[str, Any]]
