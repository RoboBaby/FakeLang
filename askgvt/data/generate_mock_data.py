"""Generate mock data using Claude API for realistic video content examples."""

import os
import json
import random
import argparse
from typing import List, Dict, Any
from anthropic import Anthropic


def generate_video_batch(
    client: Anthropic,
    category: str,
    num_videos: int = 3,
    entries_per_video: int = 6
) -> List[Dict[str, Any]]:
    """Generate a batch of video entries for a category using Claude.

    Args:
        client: Anthropic client
        category: Video category (e.g., 'cooking', 'tech', 'fitness')
        num_videos: Number of videos to generate
        entries_per_video: Number of entries per video (split across narrative/transcript/image)

    Returns:
        List of mock data entries
    """

    prompt = f"""Generate realistic mock data for {num_videos} YouTube-style videos in the "{category}" category.

For each video, create:
- 2 narrative entries (visual descriptions of what's happening)
- 2 transcript entries (spoken words/dialogue)
- 2 image entries (descriptions of key frames/thumbnails)

Return valid JSON array with this exact structure:
[
  {{
    "type": "narrative|transcript|image",
    "text": "detailed description (50-100 words for narrative/transcript, 30-50 for image)",
    "video_id": "video_XX",
    "video_title": "Catchy YouTube Title",
    "creator_id": "creator_username",
    "category": "{category}",
    "start_s": 10.0,
    "end_s": 20.0,
    "views": 500000,
    "likes": 35000,
    "upload_date": "2025-01-15"
  }}
]

Requirements:
- Make content realistic and educational
- Use varied time ranges (0-300s typical video)
- Realistic view/like counts (100K-5M views, 5-10% like ratio)
- Upload dates in 2025
- Unique creator IDs per video
- Sequential video IDs starting from the provided offset
- Content should be searchable and answer common questions about {category}

Generate {num_videos * entries_per_video} total entries across {num_videos} videos.
Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse the response
    response_text = response.content[0].text.strip()

    # Try to extract JSON from response
    if response_text.startswith('['):
        json_str = response_text
    else:
        # Find JSON array in response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
        else:
            raise ValueError(f"Could not find JSON in response: {response_text[:200]}")

    entries = json.loads(json_str)
    return entries


def generate_questions_batch(
    client: Anthropic,
    category: str,
    video_titles: List[str],
    num_questions: int = 10
) -> List[Dict[str, str]]:
    """Generate test questions for a category using Claude.

    Args:
        client: Anthropic client
        category: Video category
        video_titles: List of video titles in this category
        num_questions: Number of questions to generate

    Returns:
        List of question dictionaries
    """

    titles_str = "\n".join(f"- {t}" for t in video_titles)

    prompt = f"""Generate {num_questions} realistic user search questions for the "{category}" category.

These questions should be the type of queries users would ask when searching for video content.
The questions should be answerable by videos with these titles:
{titles_str}

Return valid JSON array:
[
  {{
    "question": "How do I...",
    "difficulty": "easy|medium|hard",
    "expected_video_title": "matching video title from above"
  }}
]

Requirements:
- Mix of how-to, what-is, and comparison questions
- Varied difficulty levels
- Questions should naturally match the video content
- Make questions specific enough to have clear answers

Return ONLY the JSON array."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    if response_text.startswith('['):
        json_str = response_text
    else:
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
        else:
            raise ValueError(f"Could not find JSON in response")

    return json.loads(json_str)


def generate_full_dataset(
    api_key: str,
    categories: List[str],
    videos_per_category: int = 3,
    questions_per_category: int = 10,
    output_dir: str = "."
) -> None:
    """Generate complete mock dataset with videos and questions.

    Args:
        api_key: Anthropic API key
        categories: List of categories to generate
        videos_per_category: Videos per category
        questions_per_category: Test questions per category
        output_dir: Output directory for generated files
    """

    client = Anthropic(api_key=api_key)

    all_entries = {
        "narrative": [],
        "transcript": [],
        "image": []
    }
    all_questions = []
    video_id_counter = 1
    question_id_counter = 1

    print(f"Generating data for {len(categories)} categories...")
    print("=" * 60)

    for category in categories:
        print(f"\nGenerating {category}...")

        # Generate video entries
        try:
            entries = generate_video_batch(
                client,
                category,
                num_videos=videos_per_category,
                entries_per_video=6
            )

            # Track video titles for question generation
            video_titles = set()

            # Assign IDs and organize by type
            for entry in entries:
                entry_type = entry.pop("type", "narrative")

                # Update video_id with sequential numbering
                old_video_id = entry.get("video_id", "")
                if old_video_id not in video_titles:
                    video_titles.add(entry.get("video_title", ""))

                # Assign sequential ID
                entry["id"] = video_id_counter
                video_id_counter += 1

                # Add to appropriate list
                if entry_type in all_entries:
                    all_entries[entry_type].append(entry)
                else:
                    all_entries["narrative"].append(entry)

            print(f"  Generated {len(entries)} video entries")

            # Generate questions for this category
            if video_titles:
                questions = generate_questions_batch(
                    client,
                    category,
                    list(video_titles),
                    num_questions=questions_per_category
                )

                for q in questions:
                    q["id"] = question_id_counter
                    q["category"] = category
                    question_id_counter += 1
                    all_questions.append(q)

                print(f"  Generated {len(questions)} test questions")

        except Exception as e:
            print(f"  Error generating {category}: {e}")
            continue

    # Write mock_data.py
    print("\n" + "=" * 60)
    print("Writing output files...")

    mock_data_content = '''"""Mock data representing the video knowledge graph.

Generated by generate_mock_data.py using Claude API.
"""

from typing import Dict, List, Any

MOCK_DATA: Dict[str, List[Dict[str, Any]]] = {
    "narrative": ''' + json.dumps(all_entries["narrative"], indent=8) + ''',
    "transcript": ''' + json.dumps(all_entries["transcript"], indent=8) + ''',
    "image": ''' + json.dumps(all_entries["image"], indent=8) + '''
}
'''

    mock_data_path = os.path.join(output_dir, "mock_data_generated.py")
    with open(mock_data_path, "w") as f:
        f.write(mock_data_content)
    print(f"  Written: {mock_data_path}")

    # Write test_questions.csv
    import csv
    questions_path = os.path.join(output_dir, "test_questions_generated.csv")
    with open(questions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "category", "expected_video_title", "difficulty"])
        writer.writeheader()
        for q in all_questions:
            writer.writerow(q)
    print(f"  Written: {questions_path}")

    # Summary
    total_entries = sum(len(v) for v in all_entries.values())
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total video entries: {total_entries}")
    print(f"  - Narrative: {len(all_entries['narrative'])}")
    print(f"  - Transcript: {len(all_entries['transcript'])}")
    print(f"  - Image: {len(all_entries['image'])}")
    print(f"Total test questions: {len(all_questions)}")
    print(f"Categories: {', '.join(categories)}")


def main():
    parser = argparse.ArgumentParser(description="Generate mock data using Claude API")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "cooking", "tech", "fitness", "gaming", "music",
            "travel", "diy", "automotive", "education", "beauty",
            "pets", "photography", "fashion", "finance"
        ],
        help="Categories to generate"
    )
    parser.add_argument(
        "--videos-per-category",
        type=int,
        default=3,
        help="Number of videos per category (default: 3)"
    )
    parser.add_argument(
        "--questions-per-category",
        type=int,
        default=8,
        help="Number of test questions per category (default: 8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        print("Please set: export ANTHROPIC_API_KEY='your-key'")
        return

    generate_full_dataset(
        api_key=api_key,
        categories=args.categories,
        videos_per_category=args.videos_per_category,
        questions_per_category=args.questions_per_category,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
