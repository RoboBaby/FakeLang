"""Qdrant setup and search functions."""

from typing import List, Dict, Any, Optional, Literal, cast

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from askgvt.models import RetrievalHit
from askgvt.data.mock_data import MOCK_DATA


def setup_qdrant(client: QdrantClient, embeddings: OpenAIEmbeddings) -> QdrantClient:
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
            "metadata": {
                k: v for k, v in payload.items()
                if k not in ["text", "video_id", "start_s", "end_s"]
            }
        }
        hits.append(hit)

    return hits
