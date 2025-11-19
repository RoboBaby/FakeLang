"""Evaluation script for AskGVT agent with detailed tracing and statistics."""

import os
import csv
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from askgvt.main import create_agent, create_initial_state


@dataclass
class NodeExecution:
    """Record of a single node execution."""
    node_name: str
    start_time: float
    end_time: float
    duration_ms: float
    output_keys: List[str]


@dataclass
class QuestionResult:
    """Complete result for a single question evaluation."""
    question_id: int
    question: str
    category: str
    expected_videos: str
    difficulty: str

    # Execution details
    total_time_ms: float
    node_executions: List[NodeExecution]
    node_count: int

    # Classification results
    classified_category: Optional[str] = None
    classified_capability: Optional[str] = None
    needs_visual_evidence: Optional[bool] = None
    classified_difficulty: Optional[str] = None

    # Retrieval results
    narrative_hits: int = 0
    transcript_hits: int = 0
    image_hits: int = 0
    total_hits: int = 0
    videos_found: List[str] = field(default_factory=list)

    # Answer results
    final_answer: Optional[str] = None
    answer_coverage: Optional[str] = None
    videos_in_answer: int = 0

    # Critic results
    is_satisfactory: Optional[bool] = None
    critic_reasons: List[str] = field(default_factory=list)
    retry_count: int = 0

    # Evaluation metrics
    found_expected_video: bool = False
    error: Optional[str] = None


@dataclass
class EvaluationStats:
    """Aggregate statistics for the evaluation run."""
    total_questions: int = 0
    successful: int = 0
    failed: int = 0

    # Timing stats
    total_time_ms: float = 0
    avg_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0

    # Node stats
    avg_nodes_per_query: float = 0
    node_time_breakdown: Dict[str, float] = field(default_factory=dict)

    # Retrieval stats
    avg_narrative_hits: float = 0
    avg_transcript_hits: float = 0
    avg_image_hits: float = 0
    avg_total_hits: float = 0

    # Quality stats
    satisfactory_rate: float = 0
    found_expected_rate: float = 0
    retry_rate: float = 0

    # Category breakdown
    category_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    difficulty_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_questions(csv_path: str, num_questions: int = 100, shuffle: bool = True) -> List[Dict[str, Any]]:
    """Load questions from CSV file.

    Args:
        csv_path: Path to the CSV file
        num_questions: Number of questions to load
        shuffle: Whether to shuffle the questions

    Returns:
        List of question dictionaries
    """
    questions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'id': int(row['id']),
                'question': row['question'],
                'category': row['category'],
                'expected_videos': row.get('expected_videos', row.get('expected_video_title', '')),
                'difficulty': row['difficulty']
            })

    if shuffle:
        random.shuffle(questions)

    return questions[:num_questions]


def run_single_evaluation(
    graph,
    question: Dict[str, Any],
    logger: logging.Logger
) -> QuestionResult:
    """Run evaluation for a single question with detailed tracing.

    Args:
        graph: Compiled LangGraph
        question: Question dictionary
        logger: Logger instance

    Returns:
        QuestionResult with all metrics
    """
    result = QuestionResult(
        question_id=question['id'],
        question=question['question'],
        category=question['category'],
        expected_videos=question['expected_videos'],
        difficulty=question['difficulty'],
        total_time_ms=0,
        node_executions=[],
        node_count=0
    )

    try:
        # Create initial state
        initial_state = create_initial_state(question['question'])
        config = {"recursion_limit": 50}

        # Track execution
        start_time = time.time()
        node_executions = []
        final_state = initial_state.copy()

        # Stream through graph with timing
        for step in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, output in step.items():
                node_start = time.time()

                # Update final state
                if isinstance(output, dict):
                    final_state.update(output)

                node_end = time.time()
                duration_ms = (node_end - node_start) * 1000

                execution = NodeExecution(
                    node_name=node_name,
                    start_time=node_start,
                    end_time=node_end,
                    duration_ms=duration_ms,
                    output_keys=list(output.keys()) if isinstance(output, dict) else []
                )
                node_executions.append(execution)

                logger.debug(f"  Node '{node_name}' completed in {duration_ms:.1f}ms")

        end_time = time.time()

        # Populate result
        result.total_time_ms = (end_time - start_time) * 1000
        result.node_executions = node_executions
        result.node_count = len(node_executions)

        # Classification results
        result.classified_category = final_state.get('category')
        result.classified_capability = final_state.get('capability')
        result.needs_visual_evidence = final_state.get('needs_visual_evidence')
        result.classified_difficulty = final_state.get('difficulty')

        # Retrieval results
        narrative_hits = final_state.get('narrative_hits', [])
        transcript_hits = final_state.get('transcript_hits', [])
        image_hits = final_state.get('image_hits', [])

        result.narrative_hits = len(narrative_hits)
        result.transcript_hits = len(transcript_hits)
        result.image_hits = len(image_hits)
        result.total_hits = result.narrative_hits + result.transcript_hits + result.image_hits

        # Get unique videos found
        all_hits = narrative_hits + transcript_hits + image_hits
        result.videos_found = list(set(hit.get('video_id', '') for hit in all_hits))

        # Also collect video titles for matching
        video_titles_found = list(set(hit.get('video_title', '') for hit in all_hits))

        # Answer results
        result.final_answer = final_state.get('final_answer', '')

        metadata = final_state.get('answer_metadata', {})
        if metadata:
            result.answer_coverage = metadata.get('coverage')
            result.videos_in_answer = metadata.get('num_videos_considered', 0)

            # Critic results
            critic = metadata.get('critic', {})
            if critic:
                result.is_satisfactory = critic.get('is_satisfactory')
                result.critic_reasons = critic.get('reasons', [])

        result.retry_count = final_state.get('retry_count', 0)

        # Check if expected video was found (match by video_id or video_title)
        expected = question['expected_videos'].split(',')
        result.found_expected_video = any(
            exp.strip() in result.videos_found or exp.strip() in video_titles_found
            for exp in expected
        )

    except Exception as e:
        result.error = str(e)
        logger.error(f"Error evaluating question {question['id']}: {e}")

    return result


def calculate_statistics(results: List[QuestionResult]) -> EvaluationStats:
    """Calculate aggregate statistics from evaluation results.

    Args:
        results: List of QuestionResult objects

    Returns:
        EvaluationStats with computed metrics
    """
    stats = EvaluationStats()
    stats.total_questions = len(results)

    # Filter successful results
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    stats.successful = len(successful)
    stats.failed = len(failed)

    if not successful:
        return stats

    # Timing stats
    times = [r.total_time_ms for r in successful]
    stats.total_time_ms = sum(times)
    stats.avg_time_ms = sum(times) / len(times)
    stats.min_time_ms = min(times)
    stats.max_time_ms = max(times)

    # Node stats
    node_counts = [r.node_count for r in successful]
    stats.avg_nodes_per_query = sum(node_counts) / len(node_counts)

    # Node time breakdown
    node_times: Dict[str, List[float]] = {}
    for r in successful:
        for execution in r.node_executions:
            if execution.node_name not in node_times:
                node_times[execution.node_name] = []
            node_times[execution.node_name].append(execution.duration_ms)

    stats.node_time_breakdown = {
        name: sum(times) / len(times)
        for name, times in node_times.items()
    }

    # Retrieval stats
    stats.avg_narrative_hits = sum(r.narrative_hits for r in successful) / len(successful)
    stats.avg_transcript_hits = sum(r.transcript_hits for r in successful) / len(successful)
    stats.avg_image_hits = sum(r.image_hits for r in successful) / len(successful)
    stats.avg_total_hits = sum(r.total_hits for r in successful) / len(successful)

    # Quality stats
    satisfactory = [r for r in successful if r.is_satisfactory]
    found_expected = [r for r in successful if r.found_expected_video]
    retried = [r for r in successful if r.retry_count > 0]

    stats.satisfactory_rate = len(satisfactory) / len(successful) * 100
    stats.found_expected_rate = len(found_expected) / len(successful) * 100
    stats.retry_rate = len(retried) / len(successful) * 100

    # Category breakdown
    categories: Dict[str, List[QuestionResult]] = {}
    for r in successful:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    for cat, cat_results in categories.items():
        sat = len([r for r in cat_results if r.is_satisfactory])
        found = len([r for r in cat_results if r.found_expected_video])
        stats.category_stats[cat] = {
            'count': len(cat_results),
            'satisfactory_rate': sat / len(cat_results) * 100,
            'found_expected_rate': found / len(cat_results) * 100,
            'avg_time_ms': sum(r.total_time_ms for r in cat_results) / len(cat_results)
        }

    # Difficulty breakdown
    difficulties: Dict[str, List[QuestionResult]] = {}
    for r in successful:
        if r.difficulty not in difficulties:
            difficulties[r.difficulty] = []
        difficulties[r.difficulty].append(r)

    for diff, diff_results in difficulties.items():
        sat = len([r for r in diff_results if r.is_satisfactory])
        found = len([r for r in diff_results if r.found_expected_video])
        stats.difficulty_stats[diff] = {
            'count': len(diff_results),
            'satisfactory_rate': sat / len(diff_results) * 100,
            'found_expected_rate': found / len(diff_results) * 100,
            'avg_time_ms': sum(r.total_time_ms for r in diff_results) / len(diff_results)
        }

    return stats


def generate_report(
    results: List[QuestionResult],
    stats: EvaluationStats,
    output_dir: Path
) -> None:
    """Generate evaluation report files.

    Args:
        results: List of QuestionResult objects
        stats: Computed statistics
        output_dir: Directory to write reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Write detailed JSON results
    json_path = output_dir / f'evaluation_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'results': [
                {
                    **{k: v for k, v in asdict(r).items() if k != 'node_executions'},
                    'node_executions': [asdict(n) for n in r.node_executions]
                }
                for r in results
            ],
            'stats': asdict(stats)
        }, f, indent=2, default=str)

    # Write human-readable report
    report_path = output_dir / f'evaluation_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ASKGVT EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Questions: {stats.total_questions}\n")
        f.write(f"Successful: {stats.successful}\n")
        f.write(f"Failed: {stats.failed}\n")
        f.write(f"Success Rate: {stats.successful/stats.total_questions*100:.1f}%\n\n")

        # Timing
        f.write("TIMING STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Time: {stats.total_time_ms/1000:.2f}s\n")
        f.write(f"Average Time: {stats.avg_time_ms:.0f}ms\n")
        f.write(f"Min Time: {stats.min_time_ms:.0f}ms\n")
        f.write(f"Max Time: {stats.max_time_ms:.0f}ms\n")
        f.write(f"Avg Nodes/Query: {stats.avg_nodes_per_query:.1f}\n\n")

        # Node breakdown
        f.write("NODE TIME BREAKDOWN (avg ms)\n")
        f.write("-" * 40 + "\n")
        for node, avg_time in sorted(stats.node_time_breakdown.items(), key=lambda x: -x[1]):
            f.write(f"  {node}: {avg_time:.1f}ms\n")
        f.write("\n")

        # Retrieval stats
        f.write("RETRIEVAL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Avg Narrative Hits: {stats.avg_narrative_hits:.1f}\n")
        f.write(f"Avg Transcript Hits: {stats.avg_transcript_hits:.1f}\n")
        f.write(f"Avg Image Hits: {stats.avg_image_hits:.1f}\n")
        f.write(f"Avg Total Hits: {stats.avg_total_hits:.1f}\n\n")

        # Quality stats
        f.write("QUALITY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Satisfactory Rate: {stats.satisfactory_rate:.1f}%\n")
        f.write(f"Found Expected Video: {stats.found_expected_rate:.1f}%\n")
        f.write(f"Retry Rate: {stats.retry_rate:.1f}%\n\n")

        # Category breakdown
        f.write("CATEGORY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for cat, cat_stats in sorted(stats.category_stats.items()):
            f.write(f"\n  {cat.upper()} ({cat_stats['count']} questions)\n")
            f.write(f"    Satisfactory: {cat_stats['satisfactory_rate']:.1f}%\n")
            f.write(f"    Found Expected: {cat_stats['found_expected_rate']:.1f}%\n")
            f.write(f"    Avg Time: {cat_stats['avg_time_ms']:.0f}ms\n")
        f.write("\n")

        # Difficulty breakdown
        f.write("DIFFICULTY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for diff, diff_stats in sorted(stats.difficulty_stats.items()):
            f.write(f"\n  {diff.upper()} ({diff_stats['count']} questions)\n")
            f.write(f"    Satisfactory: {diff_stats['satisfactory_rate']:.1f}%\n")
            f.write(f"    Found Expected: {diff_stats['found_expected_rate']:.1f}%\n")
            f.write(f"    Avg Time: {diff_stats['avg_time_ms']:.0f}ms\n")
        f.write("\n")

        # Individual results summary
        f.write("=" * 80 + "\n")
        f.write("INDIVIDUAL QUESTION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            status = "OK" if r.error is None else "FAILED"
            sat = "Y" if r.is_satisfactory else "N"
            found = "Y" if r.found_expected_video else "N"

            f.write(f"Q{r.question_id}: [{status}] {r.question[:60]}...\n")
            f.write(f"  Category: {r.category} | Difficulty: {r.difficulty}\n")
            f.write(f"  Time: {r.total_time_ms:.0f}ms | Nodes: {r.node_count}\n")
            f.write(f"  Hits: N={r.narrative_hits} T={r.transcript_hits} I={r.image_hits}\n")
            f.write(f"  Videos Found: {', '.join(r.videos_found[:5])}\n")
            f.write(f"  Satisfactory: {sat} | Found Expected: {found}\n")

            if r.error:
                f.write(f"  ERROR: {r.error}\n")

            if r.final_answer:
                answer_preview = r.final_answer[:200].replace('\n', ' ')
                f.write(f"  Answer: {answer_preview}...\n")

            f.write("\n")

    print(f"\nReports written to:")
    print(f"  - {json_path}")
    print(f"  - {report_path}")


def main(
    num_questions: int = 100,
    output_dir: str = "evaluation_output",
    verbose: bool = False
):
    """Run the evaluation.

    Args:
        num_questions: Number of questions to evaluate
        output_dir: Directory for output files
        verbose: Enable verbose logging
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("ASKGVT EVALUATION")
    print("=" * 80)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY not set.")
        print("Please set it: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Load questions
    csv_path = Path(__file__).parent / "data" / "test_questions_generated.csv"
    print(f"\nLoading questions from {csv_path}...")
    questions = load_questions(str(csv_path), num_questions)
    print(f"Loaded {len(questions)} questions")

    # Create agent
    print("\nInitializing agent...")
    graph, client, embeddings, llm = create_agent()
    print("Agent ready")

    # Run evaluations
    print(f"\nRunning evaluation on {len(questions)} questions...")
    print("-" * 80)

    results: List[QuestionResult] = []

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Q{question['id']}: {question['question'][:50]}...")

        result = run_single_evaluation(graph, question, logger)
        results.append(result)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            sat = "Y" if result.is_satisfactory else "N"
            found = "Y" if result.found_expected_video else "N"
            print(f"  Time: {result.total_time_ms:.0f}ms | "
                  f"Nodes: {result.node_count} | "
                  f"Hits: {result.total_hits} | "
                  f"Sat: {sat} | "
                  f"Found: {found}")

    print("\n" + "-" * 80)
    print("Evaluation complete")

    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(results)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {stats.total_questions}")
    print(f"Successful: {stats.successful} ({stats.successful/stats.total_questions*100:.1f}%)")
    print(f"Satisfactory Rate: {stats.satisfactory_rate:.1f}%")
    print(f"Found Expected Video: {stats.found_expected_rate:.1f}%")
    print(f"Average Time: {stats.avg_time_ms:.0f}ms")
    print(f"Total Time: {stats.total_time_ms/1000:.1f}s")

    # Generate reports
    print("\nGenerating reports...")
    output_path = Path(output_dir)
    generate_report(results, stats, output_path)

    return results, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AskGVT agent")
    parser.add_argument(
        "-n", "--num-questions",
        type=int,
        default=100,
        help="Number of questions to evaluate (default: 100)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="evaluation_output",
        help="Output directory for reports (default: evaluation_output)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    main(args.num_questions, args.output_dir, args.verbose)
