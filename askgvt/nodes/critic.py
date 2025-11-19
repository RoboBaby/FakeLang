"""Answer critic node."""

import json

from langchain_openai import ChatOpenAI

from askgvt.models import AskGVTState, CriticOutput


def create_answer_critic_node(llm: ChatOpenAI):
    """Create the answer critic node."""

    structured_llm = llm.with_structured_output(CriticOutput)

    def answer_critic_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Answer Critic for AskGVT.

Your job:
- Check if the answer is:
  - grounded in the evidence (no obvious hallucinations about videos),
  - addressing the full question,
  - honest about limitations.

Return your assessment."""

        critic_context = f"""
User Query: {state["user_query"]}

Final Answer:
{state.get("final_answer", "")}

Answer Metadata:
- Coverage: {state.get("answer_metadata", {}).get("coverage", "unknown")}
- Videos Considered: {state.get("answer_metadata", {}).get("num_videos_considered", 0)}
- Limitations: {state.get("answer_metadata", {}).get("limitations", [])}

Evidence Summary:
{json.dumps(state.get("fused_evidence", {}), indent=2, default=str)[:1000]}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critic_context}
        ]

        result = structured_llm.invoke(messages)

        # Update answer_metadata with critic result
        current_metadata = state.get("answer_metadata", {})
        current_metadata["critic"] = {
            "is_satisfactory": result.is_satisfactory,
            "reasons": result.reasons,
            "should_retry_retrieval": result.should_retry_retrieval,
            "suggested_retrieval_adjustments": result.suggested_retrieval_adjustments
        }

        return {"answer_metadata": current_metadata}

    return answer_critic_node
