"""Main entry point for AskGVT."""

import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient

from askgvt.models import AskGVTState
from askgvt.search import setup_qdrant
from askgvt.graph import build_askgvt_graph


def create_agent(
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-ada-002"
):
    """Create and return the AskGVT agent.

    Args:
        openai_api_key: OpenAI API key (uses env var if not provided)
        model: LLM model to use
        embedding_model: Embedding model to use

    Returns:
        Tuple of (compiled graph, qdrant client, embeddings, llm)
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    llm = ChatOpenAI(model=model, temperature=0)
    client = QdrantClient(location=":memory:")

    setup_qdrant(client, embeddings)
    graph = build_askgvt_graph(client, embeddings, llm)

    return graph, client, embeddings, llm


def create_initial_state(user_query: str) -> AskGVTState:
    """Create initial state for a query.

    Args:
        user_query: The user's question

    Returns:
        Initial AskGVTState
    """
    return {
        "user_query": user_query,
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


def print_results(result: AskGVTState):
    """Print formatted results."""

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

    # Create agent
    print("\nInitializing components...")
    graph, client, embeddings, llm = create_agent()

    # Test query
    test_query = "What happens when the reviewer tries to install the graphics card?"

    print("\n" + "=" * 70)
    print("Test Query:")
    print(f'  "{test_query}"')
    print("=" * 70)

    # Run graph
    print("\nRunning agent pipeline...")
    initial_state = create_initial_state(test_query)
    result = graph.invoke(initial_state)

    # Display results
    print_results(result)

    return result


if __name__ == "__main__":
    main()
