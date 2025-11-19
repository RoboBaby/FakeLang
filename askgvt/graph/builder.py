"""LangGraph construction for AskGVT."""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
)


def build_askgvt_graph(
    client: QdrantClient,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI
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
