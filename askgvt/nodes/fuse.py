"""Evidence fusion node."""

from typing import Dict, Any
from math import floor

from langchain_openai import ChatOpenAI

from askgvt.models import AskGVTState, FusedEvidence


def create_fuse_evidence_node(llm: ChatOpenAI):
    """Create the evidence fusion node."""

    def fuse_evidence_node(state: AskGVTState) -> dict:
        narrative_hits = state.get("narrative_hits", [])
        transcript_hits = state.get("transcript_hits", [])
        image_hits = state.get("image_hits", [])

        # Group by (video_id, 10s window)
        segments: Dict[str, Dict[str, Any]] = {}

        for hit in narrative_hits + transcript_hits + image_hits:
            video_id = hit["video_id"]
            window_start = floor(hit["start_s"] / 10) * 10
            key = f"{video_id}_{window_start}"

            if key not in segments:
                segments[key] = {
                    "video_id": video_id,
                    "start_s": window_start,
                    "end_s": window_start + 10,
                    "narrative_snippets": [],
                    "transcript_snippets": [],
                    "image_snippets": [],
                    "scores": [],
                    "metadata": hit["metadata"]
                }

            segments[key]["scores"].append(hit["score"])

            if hit["source"] == "narrative":
                segments[key]["narrative_snippets"].append(hit["text"])
            elif hit["source"] == "transcript":
                segments[key]["transcript_snippets"].append(hit["text"])
            elif hit["source"] == "image":
                segments[key]["image_snippets"].append(hit["text"])

        # Calculate combined scores
        for seg in segments.values():
            seg["combined_score"] = sum(seg["scores"]) / len(seg["scores"]) if seg["scores"] else 0
            del seg["scores"]

        # Convert to list and sort by score
        segment_list = sorted(segments.values(), key=lambda x: x["combined_score"], reverse=True)

        # Get unique videos and categories
        unique_videos = set(seg["video_id"] for seg in segment_list)
        categories = list(set(
            seg["metadata"].get("category", "unknown")
            for seg in segment_list
            if seg.get("metadata")
        ))

        # Use LLM to analyze patterns if we have enough data
        if segment_list:
            segment_summaries = []
            for seg in segment_list[:10]:
                summary = f"Video {seg['video_id']} ({seg['start_s']}-{seg['end_s']}s): "
                if seg["narrative_snippets"]:
                    summary += f"Actions: {'; '.join(seg['narrative_snippets'][:2])}. "
                if seg["transcript_snippets"]:
                    summary += f"Said: {'; '.join(seg['transcript_snippets'][:2])}. "
                if seg["image_snippets"]:
                    summary += f"Visual: {'; '.join(seg['image_snippets'][:2])}."
                segment_summaries.append(summary)

            analysis_prompt = f"""Analyze these video segments and identify:
1. Key patterns (what do creators commonly do?)
2. Any disagreements or variations
3. Limitations of this evidence

Segments:
{chr(10).join(segment_summaries)}

Return a brief analysis with key_patterns, disagreements, and limitations as bullet points."""

            response = llm.invoke([{"role": "user", "content": analysis_prompt}])
            analysis_text = response.content

            # Parse simple patterns from response
            key_patterns = []
            disagreements = []
            limitations = []

            current_section = None
            for line in analysis_text.split('\n'):
                line = line.strip()
                if 'pattern' in line.lower():
                    current_section = 'patterns'
                elif 'disagreement' in line.lower() or 'variation' in line.lower():
                    current_section = 'disagreements'
                elif 'limitation' in line.lower():
                    current_section = 'limitations'
                elif line.startswith('-') or line.startswith('*'):
                    item = line.lstrip('-* ').strip()
                    if current_section == 'patterns':
                        key_patterns.append(item)
                    elif current_section == 'disagreements':
                        disagreements.append(item)
                    elif current_section == 'limitations':
                        limitations.append(item)
        else:
            key_patterns = []
            disagreements = []
            limitations = ["No evidence found in the corpus"]

        fused: FusedEvidence = {
            "num_videos": len(unique_videos),
            "num_segments": len(segment_list),
            "categories": categories,
            "time_horizon": state.get("time_horizon", "unspecified"),
            "key_patterns": key_patterns[:5],
            "disagreements": disagreements[:3],
            "limitations": limitations[:3] if limitations else ["Limited sample size"],
            "segments": [
                {
                    "video_id": seg["video_id"],
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "summary": "; ".join(
                        seg["narrative_snippets"][:1] +
                        seg["transcript_snippets"][:1] +
                        seg["image_snippets"][:1]
                    ),
                    "score": seg["combined_score"],
                    "metadata": seg["metadata"]
                }
                for seg in segment_list[:20]
            ]
        }

        return {"fused_evidence": fused}

    return fuse_evidence_node
