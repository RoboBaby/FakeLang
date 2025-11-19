"""
AskGVT Retrieval Agent - Alpha Version
A video-first search engine using LangGraph and Qdrant.
"""

import os
from typing import List, Optional, Literal, Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langgraph.graph import StateGraph, END


# =============================================================================
# Part 1: The Data Structures (Strict Type Definitions)
# =============================================================================

class VideoChunkMetadata(BaseModel):
    """Schema for Qdrant payload structure."""
    video_id: str
    video_title: str
    creator_id: str
    upload_date: str  # ISO 8601 format YYYY-MM-DD
    start_time: float  # Seconds
    end_time: float    # Seconds
    chunk_type: Literal["visual_frame", "narrative_action", "transcript"]

    # Specific fields based on type (Optional but populated based on type)
    visual_objects: Optional[List[str]] = None  # e.g., ["Air Fryer", "Nike Logo"]
    visual_text_ocr: Optional[str] = None       # Text read from screen
    action_verb: Optional[str] = None           # e.g., "chopping", "running"
    speaker_name: Optional[str] = None          # For transcripts


class VideoCitation(BaseModel):
    """Citation model - the 'receipt' for claims."""
    video_id: str
    timestamp: float
    reason: str  # Why was this clip selected?
    thumbnail_url: str  # Mock URL


class FinalResponse(BaseModel):
    """Final output structure with answer and citations."""
    answer_text: str
    citations: List[VideoCitation]
    confidence_score: float


# =============================================================================
# Part 2: The LangGraph State Schema
# =============================================================================

class AgentState(TypedDict):
    """Stateful schema for the LangGraph agent."""
    # Input
    user_query: str
    current_date: str  # Important for "trending this week" queries

    # Router Outputs
    intent: Literal["frame_lookup", "narrative_sequence", "verbal_fact", "comprehensive_trend"]
    time_filter: Optional[str]  # e.g., "2023-01-01" derived from user query

    # Retrieval State (Separated by source for distinct processing)
    visual_hits: List[Document]
    narrative_hits: List[Document]
    transcript_hits: List[Document]

    # Grading State
    is_relevant: bool

    # Final Output
    final_answer: FinalResponse


# =============================================================================
# Part 3: Router Pydantic Models for Structured Output
# =============================================================================

class RouterOutput(BaseModel):
    """Structured output for the router."""
    intent: Literal["frame_lookup", "narrative_sequence", "verbal_fact", "comprehensive_trend"] = Field(
        description="The classified intent of the user query"
    )
    time_filter: Optional[str] = Field(
        default=None,
        description="Date filter extracted from query in ISO 8601 format (YYYY-MM-DD), if any"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was selected"
    )


# =============================================================================
# Part 4: Mock Data Setup
# =============================================================================

# Mock data representing the 3 videos
MOCK_VIDEO_DATA = [
    # Video A (Cooking): "Gordon Ramsay Style Burger"
    {
        "id": 1,
        "text": "Close up of 'Wagyu' beef label on premium meat packaging",
        "metadata": VideoChunkMetadata(
            video_id="video_A",
            video_title="Gordon Ramsay Style Burger",
            creator_id="chef_gordon_fan",
            upload_date="2025-01-15",
            start_time=10.0,
            end_time=10.0,
            chunk_type="visual_frame",
            visual_objects=["Wagyu beef label", "meat packaging"],
            visual_text_ocr="Wagyu"
        ).model_dump()
    },
    {
        "id": 2,
        "text": "Chef aggressively smashes patty onto cast iron skillet with force",
        "metadata": VideoChunkMetadata(
            video_id="video_A",
            video_title="Gordon Ramsay Style Burger",
            creator_id="chef_gordon_fan",
            upload_date="2025-01-15",
            start_time=10.0,
            end_time=15.0,
            chunk_type="narrative_action",
            action_verb="smashing"
        ).model_dump()
    },
    {
        "id": 3,
        "text": "Always season from a height for even distribution",
        "metadata": VideoChunkMetadata(
            video_id="video_A",
            video_title="Gordon Ramsay Style Burger",
            creator_id="chef_gordon_fan",
            upload_date="2025-01-15",
            start_time=12.0,
            end_time=12.0,
            chunk_type="transcript",
            speaker_name="Chef"
        ).model_dump()
    },

    # Video B (Tech): "RTX 5090 Unboxing"
    {
        "id": 4,
        "text": "Box clearly shows '128GB VRAM' text on RTX 5090 graphics card packaging",
        "metadata": VideoChunkMetadata(
            video_id="video_B",
            video_title="RTX 5090 Unboxing",
            creator_id="tech_reviewer_99",
            upload_date="2025-02-20",
            start_time=5.0,
            end_time=5.0,
            chunk_type="visual_frame",
            visual_objects=["RTX 5090 box", "graphics card"],
            visual_text_ocr="128GB VRAM"
        ).model_dump()
    },
    {
        "id": 5,
        "text": "Reviewer struggles to fit card into case, pushes hard against the slot",
        "metadata": VideoChunkMetadata(
            video_id="video_B",
            video_title="RTX 5090 Unboxing",
            creator_id="tech_reviewer_99",
            upload_date="2025-02-20",
            start_time=20.0,
            end_time=30.0,
            chunk_type="narrative_action",
            action_verb="struggling"
        ).model_dump()
    },
    {
        "id": 6,
        "text": "This thing is absolutely massive, barely fits in my case",
        "metadata": VideoChunkMetadata(
            video_id="video_B",
            video_title="RTX 5090 Unboxing",
            creator_id="tech_reviewer_99",
            upload_date="2025-02-20",
            start_time=25.0,
            end_time=25.0,
            chunk_type="transcript",
            speaker_name="Reviewer"
        ).model_dump()
    },

    # Video C (Fashion): "Summer Haul 2025"
    {
        "id": 7,
        "text": "Zara logo on clothing tag visible in frame",
        "metadata": VideoChunkMetadata(
            video_id="video_C",
            video_title="Summer Haul 2025",
            creator_id="fashion_influencer",
            upload_date="2025-03-10",
            start_time=50.0,
            end_time=50.0,
            chunk_type="visual_frame",
            visual_objects=["Zara logo", "clothing tag"],
            visual_text_ocr="Zara"
        ).model_dump()
    },
    {
        "id": 8,
        "text": "Creator does a 360 spin to show the skirt flow and movement",
        "metadata": VideoChunkMetadata(
            video_id="video_C",
            video_title="Summer Haul 2025",
            creator_id="fashion_influencer",
            upload_date="2025-03-10",
            start_time=50.0,
            end_time=60.0,
            chunk_type="narrative_action",
            action_verb="spinning"
        ).model_dump()
    },
    {
        "id": 9,
        "text": "It feels a bit cheaper than I expected for the price",
        "metadata": VideoChunkMetadata(
            video_id="video_C",
            video_title="Summer Haul 2025",
            creator_id="fashion_influencer",
            upload_date="2025-03-10",
            start_time=55.0,
            end_time=55.0,
            chunk_type="transcript",
            speaker_name="Creator"
        ).model_dump()
    },
]


# =============================================================================
# Qdrant Setup and Data Ingestion
# =============================================================================

def setup_qdrant_collections(client: QdrantClient, embeddings: OpenAIEmbeddings):
    """Set up Qdrant collections and ingest mock data."""

    # Create collections for each chunk type
    collections = ["visual_frames", "narrative_actions", "transcripts"]

    for collection_name in collections:
        # Delete if exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        # Create collection with OpenAI embedding dimension (1536 for text-embedding-ada-002)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    # Separate data by chunk type
    visual_data = [d for d in MOCK_VIDEO_DATA if d["metadata"]["chunk_type"] == "visual_frame"]
    narrative_data = [d for d in MOCK_VIDEO_DATA if d["metadata"]["chunk_type"] == "narrative_action"]
    transcript_data = [d for d in MOCK_VIDEO_DATA if d["metadata"]["chunk_type"] == "transcript"]

    # Ingest data into respective collections
    def ingest_to_collection(collection_name: str, data: list):
        if not data:
            return

        texts = [d["text"] for d in data]
        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=d["id"],
                vector=vector,
                payload={"text": d["text"], **d["metadata"]}
            )
            for d, vector in zip(data, vectors)
        ]

        client.upsert(collection_name=collection_name, points=points)

    ingest_to_collection("visual_frames", visual_data)
    ingest_to_collection("narrative_actions", narrative_data)
    ingest_to_collection("transcripts", transcript_data)

    return client


# =============================================================================
# Node Functions
# =============================================================================

def create_router_node(llm: ChatOpenAI):
    """Create the router node that classifies user intent."""

    # Create structured LLM for routing
    structured_llm = llm.with_structured_output(RouterOutput)

    def router_node(state: AgentState) -> dict:
        """Route the query to appropriate retrieval strategy."""

        system_prompt = """You are a query router for AskGVT, a video-first search engine.

Classify the user's query into one of these intents:

1. frame_lookup: For queries about visual elements
   - Keywords: "Logo", "Text on screen", "Color", "Visible product", "Brand", "label", "shows"

2. narrative_sequence: For queries about actions and sequences
   - Keywords: "What happens", "happens when", "Steps to", "Movement", "Action", "Pose", "Gesture", "tries to", "install", "fit"

3. verbal_fact: For queries about spoken content
   - Keywords: "Did they mention", "Quote", "Say", "said", "mentioned"

4. comprehensive_trend: For broad trend queries
   - Keywords: "Trending", "Most popular", "Common", "typical"

Also extract any date/time filters from the query if present.

Current date for reference: {current_date}
"""

        messages = [
            {"role": "system", "content": system_prompt.format(current_date=state.get("current_date", "2025-01-01"))},
            {"role": "user", "content": state["user_query"]}
        ]

        result = structured_llm.invoke(messages)

        return {
            "intent": result.intent,
            "time_filter": result.time_filter
        }

    return router_node


def create_retriever_functions(client: QdrantClient, embeddings: OpenAIEmbeddings):
    """Create retriever functions for each collection."""

    def retrieve_from_collection(collection_name: str, query: str, top_k: int = 3) -> List[Document]:
        """Generic retrieval function for a collection."""
        query_vector = embeddings.embed_query(query)

        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        documents = []
        for result in results:
            doc = Document(
                page_content=result.payload.get("text", ""),
                metadata={k: v for k, v in result.payload.items() if k != "text"}
            )
            doc.metadata["score"] = result.score
            documents.append(doc)

        return documents

    def visual_retriever(state: AgentState) -> dict:
        """Retrieve from visual frames collection."""
        docs = retrieve_from_collection("visual_frames", state["user_query"])
        return {"visual_hits": docs}

    def narrative_retriever(state: AgentState) -> dict:
        """Retrieve from narrative actions collection."""
        docs = retrieve_from_collection("narrative_actions", state["user_query"])
        return {"narrative_hits": docs}

    def transcript_retriever(state: AgentState) -> dict:
        """Retrieve from transcripts collection."""
        docs = retrieve_from_collection("transcripts", state["user_query"])
        return {"transcript_hits": docs}

    def all_retriever(state: AgentState) -> dict:
        """Retrieve from all collections for comprehensive queries."""
        visual = retrieve_from_collection("visual_frames", state["user_query"])
        narrative = retrieve_from_collection("narrative_actions", state["user_query"])
        transcript = retrieve_from_collection("transcripts", state["user_query"])

        return {
            "visual_hits": visual,
            "narrative_hits": narrative,
            "transcript_hits": transcript
        }

    return visual_retriever, narrative_retriever, transcript_retriever, all_retriever


def create_generator_node(llm: ChatOpenAI):
    """Create the generator node that produces the final response."""

    def generator_node(state: AgentState) -> dict:
        """Generate final answer with citations."""

        system_prompt = """You are AskGVT. You are NOT a generic AI.

Source Knowledge Rule: You must prioritize the retrieved video content over your internal training.
- Internal Knowledge: 'Normally, you put thermal paste on the CPU.'
- Source Knowledge (Video B): 'User put thermal paste on the socket pins.'
- Your Output: 'In this video, the user incorrectly applies paste to the socket pins.'

Evidence Rule: You cannot make a claim without a receipt. If you say the user is struggling, you must reference the narrative chunk with its timestamp.

Synthesis: Combining visual cues (Visual Frame) with actions (Narrative) is your superpower. If the visual shows a brand, and the narrative shows it breaking, mention both.

Based on the retrieved content, provide a helpful answer to the user's query. Always cite specific timestamps and video sources."""

        # Compile retrieved content
        context_parts = []
        all_docs = []

        visual_hits = state.get("visual_hits", [])
        narrative_hits = state.get("narrative_hits", [])
        transcript_hits = state.get("transcript_hits", [])

        if visual_hits:
            context_parts.append("=== Visual Frame Content ===")
            for doc in visual_hits:
                context_parts.append(f"[{doc.metadata.get('video_title')} @ {doc.metadata.get('start_time')}s]: {doc.page_content}")
                all_docs.append(doc)

        if narrative_hits:
            context_parts.append("\n=== Narrative Action Content ===")
            for doc in narrative_hits:
                context_parts.append(f"[{doc.metadata.get('video_title')} @ {doc.metadata.get('start_time')}-{doc.metadata.get('end_time')}s]: {doc.page_content}")
                all_docs.append(doc)

        if transcript_hits:
            context_parts.append("\n=== Transcript Content ===")
            for doc in transcript_hits:
                speaker = doc.metadata.get('speaker_name', 'Unknown')
                context_parts.append(f"[{doc.metadata.get('video_title')} @ {doc.metadata.get('start_time')}s - {speaker}]: {doc.page_content}")
                all_docs.append(doc)

        context = "\n".join(context_parts)

        if not context.strip():
            context = "No relevant content found in the video database."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""User Query: {state["user_query"]}

Retrieved Video Content:
{context}

Provide a helpful answer based on the retrieved content. Be specific about what happens in the videos and cite timestamps."""}
        ]

        response = llm.invoke(messages)
        answer_text = response.content

        # Generate citations from retrieved documents
        citations = []
        seen_citations = set()

        for doc in all_docs:
            video_id = doc.metadata.get("video_id", "unknown")
            timestamp = doc.metadata.get("start_time", 0)
            citation_key = f"{video_id}_{timestamp}"

            if citation_key not in seen_citations:
                seen_citations.add(citation_key)

                # Generate reason based on chunk type
                chunk_type = doc.metadata.get("chunk_type", "unknown")
                if chunk_type == "visual_frame":
                    reason = f"Visual evidence: {doc.page_content[:50]}..."
                elif chunk_type == "narrative_action":
                    reason = f"Shows action: {doc.page_content[:50]}..."
                else:
                    reason = f"Verbal content: {doc.page_content[:50]}..."

                citation = VideoCitation(
                    video_id=video_id,
                    timestamp=timestamp,
                    reason=reason,
                    thumbnail_url=f"http://askgvt.img/{video_id}/{timestamp}.jpg"
                )
                citations.append(citation)

        # Calculate confidence based on retrieval scores
        if all_docs:
            avg_score = sum(doc.metadata.get("score", 0.5) for doc in all_docs) / len(all_docs)
            confidence = min(avg_score, 1.0)
        else:
            confidence = 0.0

        final_response = FinalResponse(
            answer_text=answer_text,
            citations=citations,
            confidence_score=round(confidence, 2)
        )

        return {"final_answer": final_response, "is_relevant": len(citations) > 0}

    return generator_node


# =============================================================================
# Graph Construction
# =============================================================================

def build_askgvt_graph(client: QdrantClient, embeddings: OpenAIEmbeddings, llm: ChatOpenAI):
    """Build the complete LangGraph for AskGVT."""

    # Create nodes
    router_node = create_router_node(llm)
    visual_retriever, narrative_retriever, transcript_retriever, all_retriever = create_retriever_functions(client, embeddings)
    generator_node = create_generator_node(llm)

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("visual_retriever", visual_retriever)
    workflow.add_node("narrative_retriever", narrative_retriever)
    workflow.add_node("transcript_retriever", transcript_retriever)
    workflow.add_node("all_retriever", all_retriever)
    workflow.add_node("generator", generator_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Define conditional routing
    def route_by_intent(state: AgentState) -> str:
        """Route to appropriate retriever based on intent."""
        intent = state.get("intent", "comprehensive_trend")

        routing_map = {
            "frame_lookup": "visual_retriever",
            "narrative_sequence": "narrative_retriever",
            "verbal_fact": "transcript_retriever",
            "comprehensive_trend": "all_retriever"
        }

        return routing_map.get(intent, "all_retriever")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "visual_retriever": "visual_retriever",
            "narrative_retriever": "narrative_retriever",
            "transcript_retriever": "transcript_retriever",
            "all_retriever": "all_retriever"
        }
    )

    # Connect retrievers to generator
    workflow.add_edge("visual_retriever", "generator")
    workflow.add_edge("narrative_retriever", "generator")
    workflow.add_edge("transcript_retriever", "generator")
    workflow.add_edge("all_retriever", "generator")

    # Connect generator to end
    workflow.add_edge("generator", END)

    # Compile graph
    return workflow.compile()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the AskGVT agent."""

    print("=" * 60)
    print("AskGVT Retrieval Agent - Alpha Version")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not set. Please set it to run the agent.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize components
    print("\nInitializing components...")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    client = QdrantClient(location=":memory:")

    # Setup Qdrant with mock data
    print("Setting up Qdrant collections and ingesting mock data...")
    setup_qdrant_collections(client, embeddings)

    # Build the graph
    print("Building LangGraph workflow...")
    graph = build_askgvt_graph(client, embeddings, llm)

    # Test query
    test_query = "What happens when the reviewer tries to install the graphics card?"

    print("\n" + "=" * 60)
    print("Test Query:")
    print(f"  \"{test_query}\"")
    print("=" * 60)

    # Initialize state
    initial_state: AgentState = {
        "user_query": test_query,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "intent": "comprehensive_trend",  # Will be overwritten by router
        "time_filter": None,
        "visual_hits": [],
        "narrative_hits": [],
        "transcript_hits": [],
        "is_relevant": False,
        "final_answer": FinalResponse(
            answer_text="",
            citations=[],
            confidence_score=0.0
        )
    }

    # Run the graph
    print("\nRunning agent...")
    result = graph.invoke(initial_state)

    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    print(f"\nDetected Intent: {result['intent']}")

    if result.get("time_filter"):
        print(f"Time Filter: {result['time_filter']}")

    print(f"\nRetrieved Documents:")
    print(f"  - Visual hits: {len(result.get('visual_hits', []))}")
    print(f"  - Narrative hits: {len(result.get('narrative_hits', []))}")
    print(f"  - Transcript hits: {len(result.get('transcript_hits', []))}")

    final_answer = result["final_answer"]

    print(f"\n{'=' * 60}")
    print("Final Answer:")
    print("=" * 60)
    print(f"\n{final_answer.answer_text}")

    print(f"\n{'=' * 60}")
    print("Citations:")
    print("=" * 60)

    for i, citation in enumerate(final_answer.citations, 1):
        print(f"\n[{i}] Video: {citation.video_id}")
        print(f"    Timestamp: {citation.timestamp}s")
        print(f"    Reason: {citation.reason}")
        print(f"    Thumbnail: {citation.thumbnail_url}")

    print(f"\nConfidence Score: {final_answer.confidence_score}")
    print(f"\n{'=' * 60}")

    # Return the result for programmatic use
    return result


if __name__ == "__main__":
    main()
