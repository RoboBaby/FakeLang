"""Search nodes for the three indexes."""

from typing import List

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from askgvt.models import AskGVTState, RetrievalHit
from askgvt.search.qdrant import search_index


def create_search_nodes(client: QdrantClient, embeddings: OpenAIEmbeddings):
    """Create search nodes for all three indexes."""

    def search_narrative_node(state: AskGVTState) -> dict:
        """Search the narrative index."""
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
        """Search the transcript index."""
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
        """Search the image index."""
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
