"""LangGraph construction for AskGVT."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

from askgvt.models import AskGVTState
from askgvt.nodes import (
    normalize_question,
    create_classify_query_node,
    create_plan_retrieval_node,
    create_search_nodes,
    create_fuse_evidence_node,
    create_background_answer_node,
    create_generate_answer_node,
    create_answer_critic_node,
    create_deep_planner_node,
    create_step_executor_node,
    create_re_planner_node,
    should_continue_research,
)


def build_askgvt_graph(
    client: QdrantClient,
    embeddings: Embeddings,
    llm: BaseChatModel
):
    """Build the complete LangGraph for AskGVT.

    Args:
        client: Qdrant client instance
        embeddings: OpenAI embeddings model
        llm: OpenAI chat model

    Returns:
        Compiled LangGraph
    """

    # Create all nodes
    classify_query_node = create_classify_query_node(llm)
    plan_retrieval_node = create_plan_retrieval_node(llm)
    search_narrative_node, search_transcript_node, search_image_node = create_search_nodes(
        client, embeddings
    )
    fuse_evidence_node = create_fuse_evidence_node(llm)
    background_answer_node = create_background_answer_node(llm)
    generate_answer_node = create_generate_answer_node(llm)
    answer_critic_node = create_answer_critic_node(llm)

    # Deep Research nodes
    deep_planner_node = create_deep_planner_node(llm)
    step_executor_node = create_step_executor_node(client, embeddings)
    re_planner_node = create_re_planner_node(llm)

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

    # Deep Research nodes
    builder.add_node("deep_planner", deep_planner_node)
    builder.add_node("step_executor", step_executor_node)
    builder.add_node("re_planner", re_planner_node)

    # Set entry point
    builder.set_entry_point("normalize_question")

    # Add edges
    builder.add_edge("normalize_question", "classify_query")

    # Conditional: route based on query complexity
    def routing_decision(state: AskGVTState) -> str:
        # Check if visual evidence is needed at all
        if not state.get("needs_visual_evidence") and not state.get("strict_askgvt_only"):
            return "background_only"

        # Check if deep research is needed for complex queries
        if state.get("requires_deep_research"):
            return "deep_research"

        # Standard retrieval path
        return "simple_retrieval"

    builder.add_conditional_edges(
        "classify_query",
        routing_decision,
        {
            "background_only": "background_answer",
            "deep_research": "deep_planner",
            "simple_retrieval": "plan_retrieval"
        }
    )

    # AskGVT branch: plan -> sequential searches -> fuse
    # Note: Running sequentially for simplicity. Can optimize with Send() for parallelism
    builder.add_edge("plan_retrieval", "search_narrative")
    builder.add_edge("search_narrative", "search_transcript")
    builder.add_edge("search_transcript", "search_image")
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

    # Deep Research loop edges
    builder.add_edge("deep_planner", "step_executor")

    # Deep Research routing after step execution
    def deep_research_routing(state: AskGVTState) -> str:
        # Check visit count limit
        if state.get("visit_count", 0) >= 5:
            return "synthesize"

        # Check if we have a final answer already
        if state.get("final_answer"):
            return "done"

        # Check if there are more steps to execute
        if state.get("plan_steps"):
            return "execute"

        # No steps left - need replanning
        return "replan"

    builder.add_conditional_edges(
        "step_executor",
        deep_research_routing,
        {
            "execute": "step_executor",
            "replan": "re_planner",
            "synthesize": "fuse_evidence",
            "done": END
        }
    )

    # Re-planner routing
    def re_planner_routing(state: AskGVTState) -> str:
        # If we have a final answer, we're done
        if state.get("final_answer"):
            return "done"

        # Check visit count limit
        if state.get("visit_count", 0) >= 5:
            return "synthesize"

        # If there are new steps, continue executing
        if state.get("plan_steps"):
            return "execute"

        # Fallback - synthesize what we have
        return "synthesize"

    builder.add_conditional_edges(
        "re_planner",
        re_planner_routing,
        {
            "execute": "step_executor",
            "synthesize": "fuse_evidence",
            "done": END
        }
    )

    return builder.compile()
