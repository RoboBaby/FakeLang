"""Query classification node."""

from langchain_openai import ChatOpenAI

from askgvt.models import AskGVTState, ClassificationOutput


def normalize_question(state: AskGVTState) -> dict:
    """Normalize the user query."""
    return {"normalized_query": state["user_query"].strip()}


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

- requires_deep_research: Boolean indicating if the question needs multi-step investigation.
  Set to true when:
  - The question has multiple distinct parts (e.g., "How do I X and what are the benefits?")
  - It's a comparison across multiple items (e.g., "Compare techniques between A and B")
  - It requires analytical reasoning (e.g., "Analyze trends in..." or "Why do creators...")
  - The intent is "compare", "correlate", or involves multiple "count" operations
  - Difficulty is "hard" and evidence_required is true
  Set to false for simple how-to, factoid, or single-focus content searches.

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
            "explicit_video_ids": result.explicit_video_ids,
            "requires_deep_research": result.requires_deep_research
        }

    return classify_query_node
