"""Retrieval planning node."""

from langchain_openai import ChatOpenAI

from askgvt.models import AskGVTState, RetrievalPlanOutput


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
