"""Main entry point for AskGVT."""

import os
import hashlib
import numpy as np
from typing import Optional, List

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from askgvt.models import AskGVTState
from askgvt.search import setup_qdrant
from askgvt.graph import build_askgvt_graph


class MockEmbeddings(Embeddings):
    """Mock embeddings that work offline using deterministic hashing.

    This is suitable for demos and testing when network access is restricted.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """Generate deterministic embedding from text hash."""
        # Create deterministic seed from text
        text_hash = hashlib.sha256(text.lower().encode()).digest()
        seed = int.from_bytes(text_hash[:4], 'big')

        # Generate deterministic embedding
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


def create_agent(
    anthropic_api_key: Optional[str] = None,
    model: str = "claude-3-5-haiku-20241022",
    embedding_model: str = "mock"
):
    """Create and return the AskGVT agent.

    Args:
        anthropic_api_key: Anthropic API key (uses env var if not provided)
        model: LLM model to use
        embedding_model: Embedding model to use (mock for offline demo)

    Returns:
        Tuple of (compiled graph, qdrant client, embeddings, llm)
    """
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Use mock embeddings for offline demo (no network access needed)
    # Dimension 1536 to match OpenAI ada-002 format used in Qdrant setup
    embeddings = MockEmbeddings(dimension=1536)
    llm = ChatAnthropic(model=model, temperature=0)
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
    print("AskGVT Retrieval Agent - Full Architecture (Claude + Local Embeddings)")
    print("=" * 70)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY not set.")
        print("Please set it: export ANTHROPIC_API_KEY='your-key-here'")
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
    config = {"recursion_limit": 50}

    # Stream to see which nodes execute
    result = None
    for step in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, output in step.items():
            print(f"  -> {node_name}")
            result = output

    if result is None:
        result = initial_state

    # Display results
    print_results(result)

    return result


if __name__ == "__main__":
    main()
