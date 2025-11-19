"""
AskGVT Retrieval Agent - Alpha Version (Full Architecture)
A video-first search engine using LangGraph and Qdrant.

This implements the complete retrieval pipeline with:
- classify_query: Structured query classification
- plan_retrieval: Multi-source retrieval planning
- search_narrative/transcript/image: RAG tools
- fuse_evidence: Evidence aggregation
- background_answer: LLM-only knowledge
- generate_answer: Hybrid answer generation
- answer_critic: Answer quality check with retry loop
"""

import os
import json
from typing import List, Optional, Literal, Dict, Any, cast
from datetime import datetime
from math import floor

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langgraph.graph import StateGraph, END


# =============================================================================
# Part 1: Complete Type Definitions
# =============================================================================

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


# =============================================================================
# Part 2: Structured Output Models for LLM Tools
# =============================================================================

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


# =============================================================================
# Part 3: Mock Data Setup
# =============================================================================

# Comprehensive mock data with all three index types
MOCK_DATA = {
    "narrative": [
        # Video A - Cooking
        {
            "id": 1,
            "text": "Chef aggressively smashes wagyu beef patty onto hot cast iron skillet, creating sizzle and smoke",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 10.0,
            "end_s": 20.0,
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        {
            "id": 2,
            "text": "Chef seasons patty from height letting salt crystals fall evenly across the surface",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 20.0,
            "end_s": 30.0,
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        # Video B - Tech
        {
            "id": 3,
            "text": "Reviewer struggles to fit massive graphics card into case, pushing hard against PCIe slot",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 20.0,
            "end_s": 30.0,
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        {
            "id": 4,
            "text": "Reviewer carefully removes GPU from anti-static bag and inspects the triple fan cooler",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 5.0,
            "end_s": 15.0,
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        # Video C - Fashion
        {
            "id": 5,
            "text": "Creator does slow 360 spin to show skirt flow and movement of fabric",
            "video_id": "video_C",
            "video_title": "Summer Haul 2025",
            "creator_id": "fashion_influencer",
            "category": "fashion",
            "start_s": 50.0,
            "end_s": 60.0,
            "views": 890000,
            "likes": 67000,
            "upload_date": "2025-03-10"
        },
        # Video D - Fitness
        {
            "id": 6,
            "text": "Trainer demonstrates proper squat form with barbell, keeping back straight and knees tracking over toes",
            "video_id": "video_D",
            "video_title": "Perfect Squat Form",
            "creator_id": "fitness_coach",
            "category": "fitness",
            "start_s": 15.0,
            "end_s": 25.0,
            "views": 3200000,
            "likes": 245000,
            "upload_date": "2025-02-28"
        },
    ],
    "transcript": [
        # Video A - Cooking
        {
            "id": 101,
            "text": "Always season from a height, this gives you even distribution across the entire patty",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 12.0,
            "end_s": 16.0,
            "speaker": "Chef",
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        {
            "id": 102,
            "text": "The key to a good smash burger is getting that cast iron absolutely screaming hot",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 5.0,
            "end_s": 9.0,
            "speaker": "Chef",
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        # Video B - Tech
        {
            "id": 103,
            "text": "This thing is absolutely massive, I'm genuinely worried it won't fit in my case",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 25.0,
            "end_s": 29.0,
            "speaker": "Reviewer",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        {
            "id": 104,
            "text": "128 gigabytes of VRAM, that's just insane for a consumer card",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 8.0,
            "end_s": 12.0,
            "speaker": "Reviewer",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        # Video C - Fashion
        {
            "id": 105,
            "text": "It feels a bit cheaper than I expected for a Zara piece at this price point",
            "video_id": "video_C",
            "video_title": "Summer Haul 2025",
            "creator_id": "fashion_influencer",
            "category": "fashion",
            "start_s": 55.0,
            "end_s": 59.0,
            "speaker": "Creator",
            "views": 890000,
            "likes": 67000,
            "upload_date": "2025-03-10"
        },
        # Video D - Fitness
        {
            "id": 106,
            "text": "Keep your core tight and your chest up, don't let those knees cave inward",
            "video_id": "video_D",
            "video_title": "Perfect Squat Form",
            "creator_id": "fitness_coach",
            "category": "fitness",
            "start_s": 18.0,
            "end_s": 22.0,
            "speaker": "Trainer",
            "views": 3200000,
            "likes": 245000,
            "upload_date": "2025-02-28"
        },
    ],
    "image": [
        # Video A - Cooking
        {
            "id": 201,
            "text": "Close-up of premium Wagyu beef label on black packaging with gold lettering",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 10.0,
            "end_s": 10.0,
            "visual_objects": ["Wagyu label", "beef packaging"],
            "ocr_text": "A5 Wagyu Premium",
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        # Video B - Tech
        {
            "id": 202,
            "text": "RTX 5090 retail box showing 128GB VRAM specification in large white text on green accent",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 5.0,
            "end_s": 5.0,
            "visual_objects": ["RTX 5090 box", "NVIDIA logo"],
            "ocr_text": "128GB VRAM",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        {
            "id": 203,
            "text": "Massive triple-fan GPU barely fitting into mid-tower case, cables strained",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 25.0,
            "end_s": 25.0,
            "visual_objects": ["graphics card", "PC case", "cables"],
            "ocr_text": "",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        # Video C - Fashion
        {
            "id": 204,
            "text": "Zara brand tag visible on cream-colored flowing midi skirt",
            "video_id": "video_C",
            "video_title": "Summer Haul 2025",
            "creator_id": "fashion_influencer",
            "category": "fashion",
            "start_s": 50.0,
            "end_s": 50.0,
            "visual_objects": ["Zara tag", "skirt", "clothing label"],
            "ocr_text": "ZARA",
            "views": 890000,
            "likes": 67000,
            "upload_date": "2025-03-10"
        },
        # Video D - Fitness
        {
            "id": 205,
            "text": "Side view of trainer in squat position with Olympic barbell, gym environment with mirrors",
            "video_id": "video_D",
            "video_title": "Perfect Squat Form",
            "creator_id": "fitness_coach",
            "category": "fitness",
            "start_s": 20.0,
            "end_s": 20.0,
            "visual_objects": ["barbell", "squat rack", "gym mirrors"],
            "ocr_text": "",
            "views": 3200000,
            "likes": 245000,
            "upload_date": "2025-02-28"
        },
    ]
}


# =============================================================================
# Part 4: Qdrant Setup
# =============================================================================

def setup_qdrant(client: QdrantClient, embeddings: OpenAIEmbeddings):
    """Set up Qdrant collections and ingest mock data."""

    collections = ["narrative", "transcript", "image"]

    for collection_name in collections:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    # Ingest data
    for collection_name in collections:
        data = MOCK_DATA[collection_name]
        if not data:
            continue

        texts = [d["text"] for d in data]
        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=d["id"],
                vector=vector,
                payload={k: v for k, v in d.items() if k != "id"}
            )
            for d, vector in zip(data, vectors)
        ]

        client.upsert(collection_name=collection_name, points=points)

    return client


# =============================================================================
# Part 5: Search Functions (RAG Tools)
# =============================================================================

def search_index(
    client: QdrantClient,
    embeddings: OpenAIEmbeddings,
    collection_name: str,
    query: str,
    top_k: int = 40,
    filters: Optional[Dict[str, Any]] = None
) -> List[RetrievalHit]:
    """Generic search function for any collection."""

    query_vector = embeddings.embed_query(query)

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    hits: List[RetrievalHit] = []
    for result in results:
        payload = result.payload or {}
        hit: RetrievalHit = {
            "source": cast(Literal["narrative", "transcript", "image"], collection_name),
            "video_id": payload.get("video_id", "unknown"),
            "start_s": payload.get("start_s", 0.0),
            "end_s": payload.get("end_s", 0.0),
            "score": result.score,
            "text": payload.get("text", ""),
            "metadata": {k: v for k, v in payload.items() if k not in ["text", "video_id", "start_s", "end_s"]}
        }
        hits.append(hit)

    return hits


# =============================================================================
# Part 6: Node Functions
# =============================================================================

def normalize_question(state: AskGVTState) -> dict:
    """Normalize the user query."""
    state["normalized_query"] = state["user_query"].strip()
    return {"normalized_query": state["normalized_query"]}


def create_classify_query_node(llm: ChatOpenAI):
    """Create the query classification node."""

    structured_llm = llm.with_structured_output(ClassificationOutput)

    def classify_query_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Query Classifier for AskGVT, a visual search system over millions of social videos.

You receive ONE user question. You must return a JSON object with fields:

- category: One of the known categories:
  DIY, PC building, appliances, art/crafts, auto detailing, beauty, coffee,
  cooking, creator_analytics, dance, education, enterprise brand/compliance,
  fashion, fitness, gaming, gardening, home cleaning, music gear, parenting,
  pets, photography, small business packaging, sports, stationery/desk setups,
  thrifting, travel, tutorial, yoga, tech, or "other".

- capability: One of:
  temporal_reasoning, frame_detection, pose_detection, motion_analysis,
  narrative_segmentation, audio_visual_alignment, creator_enterprise_analytics.

- intent: One of:
  trend, find, count, compare, explain, sequence, verify, correlate, detect_change.

- evidence_required: Boolean.
  true when the user clearly expects answers based on real video evidence,
  video analytics, or specific clips.

- difficulty: "easy" | "medium" | "hard".
  Hard if the question requires multiple steps.

- question_type: High-level intent bucket: trend, how_to, comparison, analytics,
  factoid, content_search, moderation, other.

- needs_visual_evidence: Boolean indicating if the answer should primarily come from
  the AskGVT video corpus rather than generic LLM knowledge.

- needs_freshness: Boolean. True when the user cares about "this week", "recent", etc.

- time_horizon: this_week | last_month | last_year | all_time | unspecified.

- strict_askgvt_only: true if the question explicitly constrains to this video corpus.

- explicit_video_ids: array of video IDs if mentioned in the query.

If you are unsure, still choose the closest option."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state.get("normalized_query", state["user_query"])}
        ]

        result = structured_llm.invoke(messages)

        return {
            "category": result.category,
            "capability": result.capability,
            "intent": result.intent,
            "evidence_required": result.evidence_required,
            "difficulty": result.difficulty,
            "question_type": result.question_type,
            "needs_visual_evidence": result.needs_visual_evidence,
            "needs_freshness": result.needs_freshness,
            "time_horizon": result.time_horizon,
            "strict_askgvt_only": result.strict_askgvt_only,
            "explicit_video_ids": result.explicit_video_ids
        }

    return classify_query_node


def create_plan_retrieval_node(llm: ChatOpenAI):
    """Create the retrieval planning node."""

    structured_llm = llm.with_structured_output(RetrievalPlanOutput)

    def plan_retrieval_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Retrieval Planner for AskGVT.

You know AskGVT has 3 semantic indexes:
- narrative: time-aligned descriptions of actions, movement, and behaviour (10s windows).
- transcript: ASR + creator speech, product names, subjective statements.
- image: keyframe visual captions (logos, packaging, aesthetics, scene style).

Your job: given the user query and classification, choose:
- which sources to call (primary vs secondary),
- how many results to retrieve,
- which filters to apply,
- specific query strings adapted to each source.

Guidelines:
- How-to / "what do people do" → primary: NARRATIVE; secondary: TRANSCRIPT and IMAGE.
- Product / brand / ratings → primary: TRANSCRIPT; secondary: NARRATIVE and IMAGE.
- Visual aesthetics / layouts → primary: IMAGE; secondary: NARRATIVE.
- Trends → apply time filters and sort by engagement.
- Analytics → favour sources that carry the relevant signal.

The queries should be tailored to each target:
- narrative: describe actions, movements, and outcomes.
- transcript: emphasise keywords and phrases in speech.
- image: describe visual appearance, logos, and aesthetics.

Return the retrieval plan JSON."""

        classification_context = f"""
User Query: {state.get("normalized_query", state["user_query"])}

Classification:
- Category: {state.get("category", "unknown")}
- Capability: {state.get("capability", "unknown")}
- Intent: {state.get("intent", "unknown")}
- Needs Visual Evidence: {state.get("needs_visual_evidence", True)}
- Needs Freshness: {state.get("needs_freshness", False)}
- Time Horizon: {state.get("time_horizon", "unspecified")}
- Strict AskGVT Only: {state.get("strict_askgvt_only", False)}
- Explicit Video IDs: {state.get("explicit_video_ids", [])}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": classification_context}
        ]

        result = structured_llm.invoke(messages)

        return {
            "retrieval_plan": {
                "primary_sources": result.primary_sources,
                "secondary_sources": result.secondary_sources,
                "n_results_per_source": result.n_results_per_source,
                "filters": result.filters.model_dump(),
                "query_variants": [qv.model_dump() for qv in result.query_variants],
                "aggregation_strategy": result.aggregation_strategy,
                "max_total_hits": result.max_total_hits
            }
        }

    return plan_retrieval_node


def create_search_nodes(client: QdrantClient, embeddings: OpenAIEmbeddings):
    """Create search nodes for all three indexes."""

    def search_narrative_node(state: AskGVTState) -> dict:
        plan = state.get("retrieval_plan", {})
        queries = [q for q in plan.get("query_variants", []) if q["target"] == "narrative"]

        hits: List[RetrievalHit] = []
        for q in queries:
            results = search_index(
                client, embeddings, "narrative",
                q["query"],
                top_k=plan.get("n_results_per_source", 40),
                filters=plan.get("filters", {})
            )
            hits.extend(results)

        return {"narrative_hits": hits}

    def search_transcript_node(state: AskGVTState) -> dict:
        plan = state.get("retrieval_plan", {})
        queries = [q for q in plan.get("query_variants", []) if q["target"] == "transcript"]

        hits: List[RetrievalHit] = []
        for q in queries:
            results = search_index(
                client, embeddings, "transcript",
                q["query"],
                top_k=plan.get("n_results_per_source", 40),
                filters=plan.get("filters", {})
            )
            hits.extend(results)

        return {"transcript_hits": hits}

    def search_image_node(state: AskGVTState) -> dict:
        plan = state.get("retrieval_plan", {})
        queries = [q for q in plan.get("query_variants", []) if q["target"] == "image"]

        hits: List[RetrievalHit] = []
        for q in queries:
            results = search_index(
                client, embeddings, "image",
                q["query"],
                top_k=plan.get("n_results_per_source", 40),
                filters=plan.get("filters", {})
            )
            hits.extend(results)

        return {"image_hits": hits}

    return search_narrative_node, search_transcript_node, search_image_node


def create_fuse_evidence_node(llm: ChatOpenAI):
    """Create the evidence fusion node."""

    def fuse_evidence_node(state: AskGVTState) -> dict:
        narrative_hits = state.get("narrative_hits", [])
        transcript_hits = state.get("transcript_hits", [])
        image_hits = state.get("image_hits", [])

        # Group by (video_id, 10s window)
        segments: Dict[str, Dict[str, Any]] = {}

        for hit in narrative_hits + transcript_hits + image_hits:
            video_id = hit["video_id"]
            window_start = floor(hit["start_s"] / 10) * 10
            key = f"{video_id}_{window_start}"

            if key not in segments:
                segments[key] = {
                    "video_id": video_id,
                    "start_s": window_start,
                    "end_s": window_start + 10,
                    "narrative_snippets": [],
                    "transcript_snippets": [],
                    "image_snippets": [],
                    "scores": [],
                    "metadata": hit["metadata"]
                }

            segments[key]["scores"].append(hit["score"])

            if hit["source"] == "narrative":
                segments[key]["narrative_snippets"].append(hit["text"])
            elif hit["source"] == "transcript":
                segments[key]["transcript_snippets"].append(hit["text"])
            elif hit["source"] == "image":
                segments[key]["image_snippets"].append(hit["text"])

        # Calculate combined scores
        for seg in segments.values():
            seg["combined_score"] = sum(seg["scores"]) / len(seg["scores"]) if seg["scores"] else 0
            del seg["scores"]

        # Convert to list and sort by score
        segment_list = sorted(segments.values(), key=lambda x: x["combined_score"], reverse=True)

        # Get unique videos and categories
        unique_videos = set(seg["video_id"] for seg in segment_list)
        categories = list(set(
            seg["metadata"].get("category", "unknown")
            for seg in segment_list
            if seg.get("metadata")
        ))

        # Use LLM to analyze patterns if we have enough data
        if segment_list:
            segment_summaries = []
            for seg in segment_list[:10]:  # Top 10 segments
                summary = f"Video {seg['video_id']} ({seg['start_s']}-{seg['end_s']}s): "
                if seg["narrative_snippets"]:
                    summary += f"Actions: {'; '.join(seg['narrative_snippets'][:2])}. "
                if seg["transcript_snippets"]:
                    summary += f"Said: {'; '.join(seg['transcript_snippets'][:2])}. "
                if seg["image_snippets"]:
                    summary += f"Visual: {'; '.join(seg['image_snippets'][:2])}."
                segment_summaries.append(summary)

            analysis_prompt = f"""Analyze these video segments and identify:
1. Key patterns (what do creators commonly do?)
2. Any disagreements or variations
3. Limitations of this evidence

Segments:
{chr(10).join(segment_summaries)}

Return a brief analysis with key_patterns, disagreements, and limitations as bullet points."""

            response = llm.invoke([{"role": "user", "content": analysis_prompt}])
            analysis_text = response.content

            # Parse simple patterns from response
            key_patterns = []
            disagreements = []
            limitations = []

            current_section = None
            for line in analysis_text.split('\n'):
                line = line.strip()
                if 'pattern' in line.lower():
                    current_section = 'patterns'
                elif 'disagreement' in line.lower() or 'variation' in line.lower():
                    current_section = 'disagreements'
                elif 'limitation' in line.lower():
                    current_section = 'limitations'
                elif line.startswith('-') or line.startswith('*'):
                    item = line.lstrip('-* ').strip()
                    if current_section == 'patterns':
                        key_patterns.append(item)
                    elif current_section == 'disagreements':
                        disagreements.append(item)
                    elif current_section == 'limitations':
                        limitations.append(item)
        else:
            key_patterns = []
            disagreements = []
            limitations = ["No evidence found in the corpus"]

        fused: FusedEvidence = {
            "num_videos": len(unique_videos),
            "num_segments": len(segment_list),
            "categories": categories,
            "time_horizon": state.get("time_horizon", "unspecified"),
            "key_patterns": key_patterns[:5],
            "disagreements": disagreements[:3],
            "limitations": limitations[:3] if limitations else ["Limited sample size"],
            "segments": [
                {
                    "video_id": seg["video_id"],
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "summary": "; ".join(
                        seg["narrative_snippets"][:1] +
                        seg["transcript_snippets"][:1] +
                        seg["image_snippets"][:1]
                    ),
                    "score": seg["combined_score"],
                    "metadata": seg["metadata"]
                }
                for seg in segment_list[:20]
            ]
        }

        return {"fused_evidence": fused}

    return fuse_evidence_node


def create_background_answer_node(llm: ChatOpenAI):
    """Create the background answer node."""

    def background_answer_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Background Explainer for AskGVT.

Task:
- Answer from your general knowledge.
- DO NOT claim anything about specific AskGVT videos or creators.
- Focus on definitions, theory, general best practices, and likely reasons.

Return a concise but clear answer in natural language."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_query"]}
        ]

        response = llm.invoke(messages)
        return {"background_answer": response.content}

    return background_answer_node


def create_generate_answer_node(llm: ChatOpenAI):
    """Create the final answer generation node."""

    def generate_answer_node(state: AskGVTState) -> dict:
        system_prompt = """You are AskGVT, the visual search assistant that answers questions
by analysing millions of social videos.

You are given:
1. The user's question.
2. (Optional) A background answer from general knowledge.
3. (Optional) Fused video evidence summarising what actually happens in
   the AskGVT video corpus: patterns, segments, and limitations.

Your task:
- If fused evidence is available and coverage is not "none":
  - Ground your answer primarily in that evidence.
  - Use background knowledge only for definitions or hypotheses.
- If fused evidence is weak or missing:
  - Use background answer, but clearly flag that you have limited video evidence.

When citing evidence:
- Refer to patterns, counts, and example clips.
- Choose 3-10 representative segments as evidence_clips.

Rules:
- Prefer honesty over hallucination.
- If patterns disagree across videos, explain the variation.
- Keep the answer user-friendly and evidence-based."""

        # Compile context
        fused = state.get("fused_evidence")
        background = state.get("background_answer", "")

        evidence_context = ""
        if fused and fused.get("num_segments", 0) > 0:
            evidence_context = f"""
Fused Evidence Summary:
- Videos analyzed: {fused['num_videos']}
- Segments found: {fused['num_segments']}
- Categories: {', '.join(fused['categories'])}

Key Patterns:
{chr(10).join('- ' + p for p in fused['key_patterns'])}

Disagreements/Variations:
{chr(10).join('- ' + d for d in fused['disagreements']) if fused['disagreements'] else '- None noted'}

Limitations:
{chr(10).join('- ' + l for l in fused['limitations'])}

Top Segments:
"""
            for seg in fused.get("segments", [])[:10]:
                evidence_context += f"\n[{seg['video_id']} @ {seg['start_s']}-{seg['end_s']}s]: {seg['summary']}"

        user_content = f"""User Query: {state["user_query"]}

Background Knowledge Answer:
{background if background else "Not available"}

{evidence_context if evidence_context else "No video evidence available."}

Please provide a comprehensive answer grounded in the evidence above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = llm.invoke(messages)
        answer_text = response.content

        # Build evidence clips from fused segments
        evidence_clips: List[EvidenceClip] = []
        if fused:
            for seg in fused.get("segments", [])[:5]:
                clip: EvidenceClip = {
                    "video_id": seg["video_id"],
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "reason": seg.get("summary", "Relevant to query")[:100],
                    "score": seg.get("score", 0.5),
                    "metadata": seg.get("metadata", {})
                }
                evidence_clips.append(clip)

        # Determine sources used and coverage
        used_sources: List[Literal["narrative", "transcript", "image", "llm_background"]] = []
        if state.get("narrative_hits"):
            used_sources.append("narrative")
        if state.get("transcript_hits"):
            used_sources.append("transcript")
        if state.get("image_hits"):
            used_sources.append("image")
        if background:
            used_sources.append("llm_background")

        num_videos = fused.get("num_videos", 0) if fused else 0
        if num_videos >= 3:
            coverage: Literal["high", "medium", "low", "none"] = "high"
        elif num_videos >= 1:
            coverage = "medium"
        elif evidence_clips:
            coverage = "low"
        else:
            coverage = "none"

        answer_metadata: AnswerMetadata = {
            "used_sources": used_sources,
            "num_videos_considered": num_videos,
            "evidence_clips": evidence_clips,
            "coverage": coverage,
            "limitations": fused.get("limitations", []) if fused else ["No evidence retrieved"],
            "critic": None
        }

        return {
            "final_answer": answer_text,
            "answer_metadata": answer_metadata
        }

    return generate_answer_node


def create_answer_critic_node(llm: ChatOpenAI):
    """Create the answer critic node."""

    structured_llm = llm.with_structured_output(CriticOutput)

    def answer_critic_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Answer Critic for AskGVT.

Your job:
- Check if the answer is:
  - grounded in the evidence (no obvious hallucinations about videos),
  - addressing the full question,
  - honest about limitations.

Return your assessment."""

        critic_context = f"""
User Query: {state["user_query"]}

Final Answer:
{state.get("final_answer", "")}

Answer Metadata:
- Coverage: {state.get("answer_metadata", {}).get("coverage", "unknown")}
- Videos Considered: {state.get("answer_metadata", {}).get("num_videos_considered", 0)}
- Limitations: {state.get("answer_metadata", {}).get("limitations", [])}

Evidence Summary:
{json.dumps(state.get("fused_evidence", {}), indent=2, default=str)[:1000]}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critic_context}
        ]

        result = structured_llm.invoke(messages)

        # Update answer_metadata with critic result
        current_metadata = state.get("answer_metadata", {})
        current_metadata["critic"] = {
            "is_satisfactory": result.is_satisfactory,
            "reasons": result.reasons,
            "should_retry_retrieval": result.should_retry_retrieval,
            "suggested_retrieval_adjustments": result.suggested_retrieval_adjustments
        }

        return {"answer_metadata": current_metadata}

    return answer_critic_node


# =============================================================================
# Part 7: Graph Construction
# =============================================================================

def build_askgvt_graph(client: QdrantClient, embeddings: OpenAIEmbeddings, llm: ChatOpenAI):
    """Build the complete LangGraph for AskGVT."""

    # Create all nodes
    classify_query_node = create_classify_query_node(llm)
    plan_retrieval_node = create_plan_retrieval_node(llm)
    search_narrative_node, search_transcript_node, search_image_node = create_search_nodes(client, embeddings)
    fuse_evidence_node = create_fuse_evidence_node(llm)
    background_answer_node = create_background_answer_node(llm)
    generate_answer_node = create_generate_answer_node(llm)
    answer_critic_node = create_answer_critic_node(llm)

    # Build graph
    builder = StateGraph(AskGVTState)

    # Add nodes
    builder.add_node("normalize_question", normalize_question)
    builder.add_node("classify_query", classify_query_node)
    builder.add_node("plan_retrieval", plan_retrieval_node)
    builder.add_node("search_narrative", search_narrative_node)
    builder.add_node("search_transcript", search_transcript_node)
    builder.add_node("search_image", search_image_node)
    builder.add_node("fuse_evidence", fuse_evidence_node)
    builder.add_node("background_answer", background_answer_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("answer_critic", answer_critic_node)

    # Set entry point
    builder.set_entry_point("normalize_question")

    # Add edges
    builder.add_edge("normalize_question", "classify_query")

    # Conditional: needs corpus or background only
    def needs_corpus(state: AskGVTState) -> str:
        if not state.get("needs_visual_evidence") and not state.get("strict_askgvt_only"):
            return "background_only"
        return "askgvt"

    builder.add_conditional_edges(
        "classify_query",
        needs_corpus,
        {
            "background_only": "background_answer",
            "askgvt": "plan_retrieval"
        }
    )

    # AskGVT branch: plan -> parallel searches -> fuse
    builder.add_edge("plan_retrieval", "search_narrative")
    builder.add_edge("plan_retrieval", "search_transcript")
    builder.add_edge("plan_retrieval", "search_image")

    # All searches lead to fusion (LangGraph handles parallel execution)
    builder.add_edge("search_narrative", "fuse_evidence")
    builder.add_edge("search_transcript", "fuse_evidence")
    builder.add_edge("search_image", "fuse_evidence")

    # Fusion -> background -> generate -> critic
    builder.add_edge("fuse_evidence", "background_answer")
    builder.add_edge("background_answer", "generate_answer")
    builder.add_edge("generate_answer", "answer_critic")

    # Critic decision: done or retry
    def critic_decision(state: AskGVTState) -> str:
        critic = (state.get("answer_metadata") or {}).get("critic") or {}
        retry_count = state.get("retry_count", 0)

        if critic.get("is_satisfactory", True):
            return "done"
        if critic.get("should_retry_retrieval", False) and retry_count < 1:
            return "retry"
        return "done"

    builder.add_conditional_edges(
        "answer_critic",
        critic_decision,
        {"done": END, "retry": "plan_retrieval"}
    )

    return builder.compile()


# =============================================================================
# Part 8: Main Execution
# =============================================================================

def main():
    """Main function to run the AskGVT agent."""

    print("=" * 70)
    print("AskGVT Retrieval Agent - Full Architecture")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not set.")
        print("Please set it: export OPENAI_API_KEY='your-key-here'")
        return None

    # Initialize components
    print("\nInitializing components...")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    client = QdrantClient(location=":memory:")

    # Setup Qdrant
    print("Setting up Qdrant collections and ingesting mock data...")
    setup_qdrant(client, embeddings)

    # Build graph
    print("Building LangGraph workflow...")
    graph = build_askgvt_graph(client, embeddings, llm)

    # Test query
    test_query = "What happens when the reviewer tries to install the graphics card?"

    print("\n" + "=" * 70)
    print("Test Query:")
    print(f'  "{test_query}"')
    print("=" * 70)

    # Initialize state
    initial_state: AskGVTState = {
        "user_query": test_query,
        "normalized_query": None,
        "category": None,
        "capability": None,
        "intent": None,
        "evidence_required": None,
        "difficulty": None,
        "question_type": None,
        "needs_visual_evidence": None,
        "needs_freshness": None,
        "time_horizon": None,
        "strict_askgvt_only": None,
        "explicit_video_ids": [],
        "retrieval_plan": None,
        "narrative_hits": [],
        "transcript_hits": [],
        "image_hits": [],
        "fused_evidence": None,
        "background_answer": None,
        "final_answer": None,
        "answer_metadata": None,
        "retry_count": 0
    }

    # Run graph
    print("\nRunning agent pipeline...")
    result = graph.invoke(initial_state)

    # Display results
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"Category: {result.get('category')}")
    print(f"Capability: {result.get('capability')}")
    print(f"Intent: {result.get('intent')}")
    print(f"Question Type: {result.get('question_type')}")
    print(f"Needs Visual Evidence: {result.get('needs_visual_evidence')}")
    print(f"Difficulty: {result.get('difficulty')}")

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    print(f"Narrative hits: {len(result.get('narrative_hits', []))}")
    print(f"Transcript hits: {len(result.get('transcript_hits', []))}")
    print(f"Image hits: {len(result.get('image_hits', []))}")

    fused = result.get("fused_evidence")
    if fused:
        print(f"\nFused Evidence:")
        print(f"  Videos: {fused.get('num_videos')}")
        print(f"  Segments: {fused.get('num_segments')}")
        print(f"  Categories: {fused.get('categories')}")
        if fused.get("key_patterns"):
            print(f"  Key Patterns:")
            for p in fused["key_patterns"][:3]:
                print(f"    - {p}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(f"\n{result.get('final_answer', 'No answer generated')}")

    metadata = result.get("answer_metadata", {})
    if metadata:
        print("\n" + "=" * 70)
        print("ANSWER METADATA")
        print("=" * 70)
        print(f"Coverage: {metadata.get('coverage')}")
        print(f"Sources Used: {metadata.get('used_sources')}")
        print(f"Videos Considered: {metadata.get('num_videos_considered')}")

        clips = metadata.get("evidence_clips", [])
        if clips:
            print(f"\nEvidence Clips ({len(clips)}):")
            for clip in clips[:5]:
                print(f"  [{clip['video_id']} @ {clip['start_s']}-{clip['end_s']}s]")
                print(f"    Reason: {clip['reason'][:80]}...")

        critic = metadata.get("critic", {})
        if critic:
            print(f"\nCritic Assessment:")
            print(f"  Satisfactory: {critic.get('is_satisfactory')}")
            if critic.get("reasons"):
                for reason in critic["reasons"][:3]:
                    print(f"    - {reason}")

    print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    main()
