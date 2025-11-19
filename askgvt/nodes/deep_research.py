"""Deep Research nodes for complex multi-step queries.

Implements Plan-Execute-Replan pattern for thorough investigation.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from askgvt.models import AskGVTState


# Pydantic models for structured outputs

class PlanStep(BaseModel):
    """A single step in the research plan."""
    step: str = Field(description="Description of what to investigate")
    tool: Literal["narrative", "transcript", "image"] = Field(
        description="Which search tool to use for this step"
    )
    query: str = Field(description="Specific search query for this step")


class PlanOutput(BaseModel):
    """Output from the Planner node."""
    steps: List[PlanStep] = Field(
        description="List of investigation steps to execute"
    )
    reasoning: str = Field(
        description="Brief explanation of the research strategy"
    )


class ExecutorOutput(BaseModel):
    """Output from executing a single step."""
    step_description: str
    tool_used: str
    results_summary: str
    key_findings: List[str]
    num_hits: int


class ReplanOutput(BaseModel):
    """Output from the Re-Planner node."""
    action: Literal["continue", "respond"] = Field(
        description="Whether to continue research or generate final response"
    )
    new_steps: List[PlanStep] = Field(
        default_factory=list,
        description="Additional steps if continuing"
    )
    response: str = Field(
        default="",
        description="Final synthesized response if action is 'respond'"
    )
    reasoning: str = Field(
        description="Explanation of the decision"
    )


def create_deep_planner_node(llm: BaseChatModel):
    """Create the Deep Research Planner node.

    Acts as a "Principal Investigator" that decomposes complex queries
    into granular, executable steps.
    """

    structured_llm = llm.with_structured_output(PlanOutput)

    def deep_planner_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Principal Investigator for AskGVT Deep Research.

Your job is to decompose complex user questions into a sequence of focused investigation steps.
You do NOT answer the question - you only create the research plan.

Available search tools:
- narrative: Search time-aligned descriptions of actions, movements, behaviors (10s windows)
- transcript: Search ASR + creator speech, product names, subjective statements
- image: Search keyframe visual captions (logos, packaging, aesthetics, scene style)

Guidelines for creating effective plans:
1. Break multi-part questions into separate investigation steps
2. For comparisons, create parallel steps for each item being compared
3. For "how and why" questions, start with descriptive steps, then analytical
4. Order steps from broad to specific
5. Typically 3-6 steps for complex queries
6. Each step should target ONE specific piece of information

Example: "Compare meal prep techniques between Italian and Japanese cooking"
Steps:
1. Search narrative for "Italian meal prep techniques"
2. Search transcript for "Italian cooking tips preparation"
3. Search narrative for "Japanese meal prep techniques"
4. Search transcript for "Japanese cooking tips preparation"
5. Search image for "Italian vs Japanese kitchen setup"

Return a focused research plan with clear, executable steps."""

        # Include any past steps if this is a re-planning iteration
        past_context = ""
        if state.get("past_steps"):
            past_context = "\n\nPrevious investigation results:\n"
            for ps in state["past_steps"]:
                past_context += f"- {ps['step']}: {ps['result']}\n"

        user_context = f"""User Query: {state.get("normalized_query", state["user_query"])}

Classification:
- Category: {state.get("category", "unknown")}
- Intent: {state.get("intent", "unknown")}
- Difficulty: {state.get("difficulty", "unknown")}
- Question Type: {state.get("question_type", "unknown")}
{past_context}

Create a research plan to thoroughly investigate this query."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context}
        ]

        result = structured_llm.invoke(messages)

        # Convert steps to list of dicts for state
        plan_steps = [
            f"[{step.tool}] {step.query}"
            for step in result.steps
        ]

        return {
            "plan_steps": plan_steps,
            "retrieval_plan": {
                "strategy": "deep_research",
                "reasoning": result.reasoning,
                "steps": [s.model_dump() for s in result.steps]
            }
        }

    return deep_planner_node


def create_step_executor_node(client: QdrantClient, embeddings: Embeddings):
    """Create the Step Executor node.

    Acts as a "Field Reporter" that executes one step at a time,
    routing to appropriate search tools.
    """

    def step_executor_node(state: AskGVTState) -> dict:
        plan_steps = state.get("plan_steps", [])
        past_steps = state.get("past_steps", []) or []

        if not plan_steps:
            # No more steps to execute
            return {}

        # Get next step
        current_step = plan_steps[0]
        remaining_steps = plan_steps[1:]

        # Parse step format: "[tool] query"
        if current_step.startswith("["):
            tool_end = current_step.index("]")
            tool = current_step[1:tool_end].lower()
            query = current_step[tool_end+1:].strip()
        else:
            tool = "narrative"
            query = current_step

        # Map tool to collection
        collection_map = {
            "narrative": "askgvt_narrative",
            "transcript": "askgvt_transcript",
            "image": "askgvt_image"
        }
        collection = collection_map.get(tool, "askgvt_narrative")

        # Execute search
        query_embedding = embeddings.embed_query(query)

        try:
            results = client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=10
            )

            # Extract hits
            hits = []
            for r in results:
                hit = {
                    "video_id": r.payload.get("video_id", ""),
                    "video_title": r.payload.get("video_title", ""),
                    "text": r.payload.get("text", ""),
                    "start_s": r.payload.get("start_s", 0),
                    "end_s": r.payload.get("end_s", 0),
                    "score": r.score,
                    "category": r.payload.get("category", "")
                }
                hits.append(hit)

            # Summarize findings
            if hits:
                key_findings = [
                    f"{h['video_title']}: {h['text'][:100]}..."
                    for h in hits[:3]
                ]
                result_summary = f"Found {len(hits)} results. Top videos: {', '.join(set(h['video_title'] for h in hits[:3]))}"
            else:
                key_findings = ["No results found"]
                result_summary = "No results found for this query"

            # Update appropriate hit list
            update = {
                "plan_steps": remaining_steps,
                "past_steps": past_steps + [{
                    "step": current_step,
                    "result": result_summary,
                    "findings": key_findings,
                    "num_hits": len(hits)
                }]
            }

            # Add to appropriate hits list
            if tool == "narrative":
                existing = state.get("narrative_hits", [])
                update["narrative_hits"] = existing + hits
            elif tool == "transcript":
                existing = state.get("transcript_hits", [])
                update["transcript_hits"] = existing + hits
            elif tool == "image":
                existing = state.get("image_hits", [])
                update["image_hits"] = existing + hits

            return update

        except Exception as e:
            # Handle search errors gracefully
            return {
                "plan_steps": remaining_steps,
                "past_steps": past_steps + [{
                    "step": current_step,
                    "result": f"Error: {str(e)}",
                    "findings": [],
                    "num_hits": 0
                }]
            }

    return step_executor_node


def create_re_planner_node(llm: BaseChatModel):
    """Create the Re-Planner node.

    Acts as an "Editor-in-Chief" that assesses evidence quality
    and decides whether to continue, refine, or synthesize.
    """

    structured_llm = llm.with_structured_output(ReplanOutput)

    def re_planner_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Editor-in-Chief for AskGVT Deep Research.

Your job is to assess the investigation progress and decide next steps.

You have three options:
1. CONTINUE: More investigation needed - provide new steps
2. RESPOND: Sufficient evidence gathered - synthesize final answer

Assessment criteria:
- Have we found relevant evidence for all aspects of the question?
- Is the evidence quality sufficient (multiple sources, specific details)?
- For comparisons: do we have balanced coverage of all items?
- Are there gaps that additional searches could fill?

If responding, synthesize a comprehensive answer that:
- Directly addresses the user's question
- Cites specific videos and timestamps
- Acknowledges any limitations in coverage
- Provides actionable insights

Be efficient - don't over-research simple aspects, but be thorough on complex ones."""

        # Build context from past steps
        past_context = "Investigation Progress:\n"
        for ps in (state.get("past_steps") or []):
            past_context += f"\nStep: {ps['step']}\n"
            past_context += f"Result: {ps['result']}\n"
            if ps.get('findings'):
                past_context += f"Key Findings: {'; '.join(ps['findings'][:3])}\n"

        # Summary of evidence collected
        narrative_count = len(state.get("narrative_hits", []))
        transcript_count = len(state.get("transcript_hits", []))
        image_count = len(state.get("image_hits", []))

        evidence_summary = f"""
Evidence Collected:
- Narrative hits: {narrative_count}
- Transcript hits: {transcript_count}
- Image hits: {image_count}
- Total: {narrative_count + transcript_count + image_count}
"""

        user_context = f"""Original Query: {state.get("normalized_query", state["user_query"])}

{past_context}

{evidence_summary}

Remaining planned steps: {len(state.get("plan_steps", []))}
Visit count: {state.get("visit_count", 0)}

Decide: Should we continue investigating or synthesize a response?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context}
        ]

        result = structured_llm.invoke(messages)

        updates = {
            "visit_count": state.get("visit_count", 0) + 1
        }

        if result.action == "respond":
            # Ready to synthesize
            updates["final_answer"] = result.response
            updates["plan_steps"] = []  # Clear remaining steps
            updates["answer_metadata"] = {
                "coverage": "deep_research",
                "num_videos_considered": narrative_count + transcript_count + image_count,
                "research_iterations": state.get("visit_count", 0) + 1,
                "reasoning": result.reasoning
            }
        else:
            # Continue with new steps
            if result.new_steps:
                new_plan_steps = [
                    f"[{step.tool}] {step.query}"
                    for step in result.new_steps
                ]
                # Prepend new steps to remaining
                existing_steps = state.get("plan_steps", [])
                updates["plan_steps"] = new_plan_steps + existing_steps

        return updates

    return re_planner_node


def should_continue_research(state: AskGVTState) -> str:
    """Routing function for Deep Research loop.

    Returns:
        "execute": More steps to run
        "replan": Steps done, need to assess
        "done": Research complete or limit reached
    """
    # Check visit count limit
    if state.get("visit_count", 0) >= 5:
        return "done"

    # Check if we have a final answer
    if state.get("final_answer"):
        return "done"

    # Check if there are steps to execute
    if state.get("plan_steps"):
        return "execute"

    # No steps but no answer - need replanning
    return "replan"
