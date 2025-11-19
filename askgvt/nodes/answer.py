"""Answer generation nodes."""

from typing import List, Literal

from langchain_openai import ChatOpenAI

from askgvt.models import AskGVTState, EvidenceClip, AnswerMetadata


def create_background_answer_node(llm: ChatOpenAI):
    """Create the background answer node."""

    def background_answer_node(state: AskGVTState) -> dict:
        system_prompt = """You are the Background Explainer for AskGVT.

Task:
- Answer from your general knowledge.
- DO NOT claim anything about specific AskGVT videos or creators.
- Focus on definitions, theory, general best practices, and likely reasons.

Return a concise but clear answer in natural language."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_query"]}
        ]

        response = llm.invoke(messages)
        return {"background_answer": response.content}

    return background_answer_node


def create_generate_answer_node(llm: ChatOpenAI):
    """Create the final answer generation node."""

    def generate_answer_node(state: AskGVTState) -> dict:
        system_prompt = """You are AskGVT, the visual search assistant that answers questions
by analysing millions of social videos.

You are given:
1. The user's question.
2. (Optional) A background answer from general knowledge.
3. (Optional) Fused video evidence summarising what actually happens in
   the AskGVT video corpus: patterns, segments, and limitations.

Your task:
- If fused evidence is available and coverage is not "none":
  - Ground your answer primarily in that evidence.
  - Use background knowledge only for definitions or hypotheses.
- If fused evidence is weak or missing:
  - Use background answer, but clearly flag that you have limited video evidence.

When citing evidence:
- Refer to patterns, counts, and example clips.
- Choose 3-10 representative segments as evidence_clips.

Rules:
- Prefer honesty over hallucination.
- If patterns disagree across videos, explain the variation.
- Keep the answer user-friendly and evidence-based."""

        # Compile context
        fused = state.get("fused_evidence")
        background = state.get("background_answer", "")

        evidence_context = ""
        if fused and fused.get("num_segments", 0) > 0:
            evidence_context = f"""
Fused Evidence Summary:
- Videos analyzed: {fused['num_videos']}
- Segments found: {fused['num_segments']}
- Categories: {', '.join(fused['categories'])}

Key Patterns:
{chr(10).join('- ' + p for p in fused['key_patterns'])}

Disagreements/Variations:
{chr(10).join('- ' + d for d in fused['disagreements']) if fused['disagreements'] else '- None noted'}

Limitations:
{chr(10).join('- ' + l for l in fused['limitations'])}

Top Segments:
"""
            for seg in fused.get("segments", [])[:10]:
                evidence_context += f"\n[{seg['video_id']} @ {seg['start_s']}-{seg['end_s']}s]: {seg['summary']}"

        user_content = f"""User Query: {state["user_query"]}

Background Knowledge Answer:
{background if background else "Not available"}

{evidence_context if evidence_context else "No video evidence available."}

Please provide a comprehensive answer grounded in the evidence above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = llm.invoke(messages)
        answer_text = response.content

        # Build evidence clips from fused segments
        evidence_clips: List[EvidenceClip] = []
        if fused:
            for seg in fused.get("segments", [])[:5]:
                clip: EvidenceClip = {
                    "video_id": seg["video_id"],
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "reason": seg.get("summary", "Relevant to query")[:100],
                    "score": seg.get("score", 0.5),
                    "metadata": seg.get("metadata", {})
                }
                evidence_clips.append(clip)

        # Determine sources used and coverage
        used_sources: List[Literal["narrative", "transcript", "image", "llm_background"]] = []
        if state.get("narrative_hits"):
            used_sources.append("narrative")
        if state.get("transcript_hits"):
            used_sources.append("transcript")
        if state.get("image_hits"):
            used_sources.append("image")
        if background:
            used_sources.append("llm_background")

        num_videos = fused.get("num_videos", 0) if fused else 0
        if num_videos >= 3:
            coverage: Literal["high", "medium", "low", "none"] = "high"
        elif num_videos >= 1:
            coverage = "medium"
        elif evidence_clips:
            coverage = "low"
        else:
            coverage = "none"

        answer_metadata: AnswerMetadata = {
            "used_sources": used_sources,
            "num_videos_considered": num_videos,
            "evidence_clips": evidence_clips,
            "coverage": coverage,
            "limitations": fused.get("limitations", []) if fused else ["No evidence retrieved"],
            "critic": None
        }

        return {
            "final_answer": answer_text,
            "answer_metadata": answer_metadata
        }

    return generate_answer_node
