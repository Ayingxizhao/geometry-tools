#!/usr/bin/env python3
"""
Flexible VLM Benchmark for Visual Geometry Tasks

Supports testing different vision-language models on:
- 1D Line Length Comparison (350 images)
- 2D Shape Size Comparison (300 images)
- 2D Triangle Classification (100 images)

Usage:
    python benchmark_vlm.py --model openai --seed 42        # GPT-4o (Chat Completions API)
    python benchmark_vlm.py --model openai-gpt5 --seed 42   # GPT-5.1 (Responses API)
    python benchmark_vlm.py --model anthropic --seed 42     # Claude
"""

import argparse
import base64
import json
import os
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    image_path: str
    task_type: str  # e.g., "1d_line", "2d_circle", "2d_triangle_equilateral"
    subtask: str  # e.g., "Line_50", "arrow_100", "muller_lyer_150"
    ground_truth: bool  # YES=True, NO=False
    question: str  # The question to ask the model


@dataclass
class BenchmarkResult:
    """Result for a single sample."""

    sample: BenchmarkSample
    model_response: str
    predicted: Optional[bool]
    correct: bool
    error: Optional[str] = None


class VLMProvider(ABC):
    """Abstract base class for VLM providers."""

    @abstractmethod
    def query(self, image_path: str, question: str) -> str:
        """Query the VLM with an image and question."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass


class OpenAIProvider(VLMProvider):
    """OpenAI GPT-4 Vision provider (Chat Completions API)."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def query(self, image_path: str, question: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            max_tokens=100,
            temperature=0.0,  # Deterministic
        )
        return response.choices[0].message.content


class OpenAIResponsesProvider(VLMProvider):
    """OpenAI GPT-5.1 provider using the new Responses API."""

    def __init__(self, model: str = "gpt-5.1", api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    @property
    def name(self) -> str:
        return f"OpenAI Responses ({self.model})"

    def query(self, image_path: str, question: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_data}",
                        },
                        {
                            "type": "input_text",
                            "text": question,
                        },
                    ],
                }
            ],
            # Use none reasoning for low-latency (default for gpt-5.1)
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
        )
        return response.output_text


class AnthropicProvider(VLMProvider):
    """Anthropic Claude Vision provider."""

    def __init__(self, model: str = "claude-sonnet-4.5", api_key: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"

    def query(self, image_path: str, question: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        return response.content[0].text


def get_provider(provider_name: str, model: Optional[str] = None) -> VLMProvider:
    """Factory function to get VLM provider."""
    providers = {
        "openai": lambda: OpenAIProvider(model=model or "gpt-4o"),
        "openai-gpt5": lambda: OpenAIResponsesProvider(model=model or "gpt-5.1"),
        "anthropic": lambda: AnthropicProvider(
            model=model or "claude-sonnet-4-20250514"
        ),
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {list(providers.keys())}"
        )

    return providers[provider_name]()


# Task-specific questions
QUESTIONS = {
    # 1D Line comparison tasks - ask if TOP line is longer
    "1d_line": "Is the top line longer than the bottom line? Answer only YES or NO.",
    "1d_arrow": "Is the top line (the horizontal shaft) longer than the bottom line? Answer only YES or NO.",
    "1d_muller": "Is the top horizontal line segment longer than the bottom horizontal line segment? Ignore the arrow heads, only compare the main horizontal shafts. Answer only YES or NO.",
    # 2D Shape comparison tasks - ask if LEFT shape is larger
    "2d_circle": "Is the left circle larger than the right circle? Answer only YES or NO.",
    "2d_square": "Is the left square larger than the right square? Answer only YES or NO.",
    "2d_rectangle": "Is the left rectangle larger than the right rectangle? Answer only YES or NO.",
    # 2D Triangle classification tasks
    "2d_triangle_equilateral": "Is this an equilateral triangle (all three sides equal length)? Answer only YES or NO.",
    "2d_triangle_right": "Is this a right triangle (has a 90-degree angle)? Answer only YES or NO.",
}


def load_benchmark_data(data_dir: str) -> list[BenchmarkSample]:
    """Load all benchmark samples from the data directory."""
    samples = []
    data_path = Path(data_dir)

    # 1D Line Length Comparison (350 images)
    line_dir = data_path / "vision_1d_Compare_Length"

    # Line_50
    for img in (line_dir / "Line_50").glob("*.png"):
        gt = "_YES_" in img.name
        samples.append(
            BenchmarkSample(
                image_path=str(img),
                task_type="1d_line",
                subtask="Line_50",
                ground_truth=gt,
                question=QUESTIONS["1d_line"],
            )
        )

    # arrow_100
    for img in (line_dir / "arrow_100").glob("*.png"):
        gt = "_YES_" in img.name
        samples.append(
            BenchmarkSample(
                image_path=str(img),
                task_type="1d_arrow",
                subtask="arrow_100",
                ground_truth=gt,
                question=QUESTIONS["1d_arrow"],
            )
        )

    # Rotated_line_50
    for img in (line_dir / "Rotated_line_50").glob("*.png"):
        gt = "_YES_" in img.name
        samples.append(
            BenchmarkSample(
                image_path=str(img),
                task_type="1d_line",
                subtask="Rotated_line_50",
                ground_truth=gt,
                question=QUESTIONS["1d_line"],
            )
        )

    # muller_lyer_150 (3 subtypes in same folder)
    muller_dir = line_dir / "muller_lyer_150"
    for img in muller_dir.glob("*.png"):
        gt = "_YES_" in img.name
        # Determine subtype from filename
        if "allArrows" in img.name:
            subtype = "muller_allArrows"
        elif "clean" in img.name:
            subtype = "muller_clean"
        elif "more" in img.name:
            subtype = "muller_more"
        else:
            subtype = "muller_lyer"

        samples.append(
            BenchmarkSample(
                image_path=str(img),
                task_type="1d_muller",
                subtask=subtype,
                ground_truth=gt,
                question=QUESTIONS["1d_muller"],
            )
        )

    # 2D Shape Size Comparison (300 images)
    shape_dir = data_path / "vision_2d_Compare_Size"

    for subdir, task_type in [
        ("circle_100", "2d_circle"),
        ("square_50", "2d_square"),
        ("rectangle_50", "2d_rectangle"),
        ("rotated_square_50", "2d_square"),
        ("rotated_rectangle_50", "2d_rectangle"),
    ]:
        folder = shape_dir / subdir
        if folder.exists():
            for img in folder.glob("*.png"):
                gt = "_YES_" in img.name
                samples.append(
                    BenchmarkSample(
                        image_path=str(img),
                        task_type=task_type,
                        subtask=subdir,
                        ground_truth=gt,
                        question=QUESTIONS[task_type],
                    )
                )

    # 2D Triangle Classification (100 images)
    tri_dir = data_path / "vision_2d_Check_Triangles"

    # Equilateral triangles
    eq_dir = tri_dir / "vision_equilateral_triangles_50"
    if eq_dir.exists():
        for img in eq_dir.glob("*.png"):
            gt = "_YES_" in img.name
            samples.append(
                BenchmarkSample(
                    image_path=str(img),
                    task_type="2d_triangle_equilateral",
                    subtask="equilateral_50",
                    ground_truth=gt,
                    question=QUESTIONS["2d_triangle_equilateral"],
                )
            )

    # Right triangles
    rt_dir = tri_dir / "vision_right_triangles_50"
    if rt_dir.exists():
        for img in rt_dir.glob("*.png"):
            gt = "_YES_" in img.name
            samples.append(
                BenchmarkSample(
                    image_path=str(img),
                    task_type="2d_triangle_right",
                    subtask="right_50",
                    ground_truth=gt,
                    question=QUESTIONS["2d_triangle_right"],
                )
            )

    return samples


def parse_yes_no(response: str) -> Optional[bool]:
    """Parse YES/NO from model response."""
    response = response.strip().upper()

    # Direct match
    if response == "YES":
        return True
    if response == "NO":
        return False

    # Check if starts with YES or NO
    if response.startswith("YES"):
        return True
    if response.startswith("NO"):
        return False

    # Search for YES/NO in response
    yes_match = re.search(r"\bYES\b", response)
    no_match = re.search(r"\bNO\b", response)

    if yes_match and not no_match:
        return True
    if no_match and not yes_match:
        return False

    # Ambiguous or no clear answer
    return None


def run_benchmark(
    provider: VLMProvider,
    samples: list[BenchmarkSample],
    seed: int = 42,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run benchmark on samples."""

    # Set seed and shuffle
    random.seed(seed)
    samples = samples.copy()
    random.shuffle(samples)

    if max_samples:
        samples = samples[:max_samples]

    results = []

    for i, sample in enumerate(samples):
        if verbose:
            print(
                f"[{i + 1}/{len(samples)}] Processing {Path(sample.image_path).name}...",
                end=" ",
            )

        try:
            response = provider.query(sample.image_path, sample.question)
            predicted = parse_yes_no(response)
            correct = (
                (predicted == sample.ground_truth) if predicted is not None else False
            )

            result = BenchmarkResult(
                sample=sample,
                model_response=response,
                predicted=predicted,
                correct=correct,
            )

            if verbose:
                status = "✓" if correct else "✗"
                print(f"{status} (pred={predicted}, gt={sample.ground_truth})")

        except Exception as e:
            result = BenchmarkResult(
                sample=sample,
                model_response="",
                predicted=None,
                correct=False,
                error=str(e),
            )
            if verbose:
                print(f"ERROR: {e}")

        results.append(result)

    return results


def compute_metrics(results: list[BenchmarkResult]) -> dict:
    """Compute accuracy metrics from results."""
    metrics = {
        "overall": {"correct": 0, "total": 0, "errors": 0},
        "by_task_type": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0}),
        "by_subtask": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0}),
    }

    for result in results:
        task_type = result.sample.task_type
        subtask = result.sample.subtask

        metrics["overall"]["total"] += 1
        metrics["by_task_type"][task_type]["total"] += 1
        metrics["by_subtask"][subtask]["total"] += 1

        if result.error:
            metrics["overall"]["errors"] += 1
            metrics["by_task_type"][task_type]["errors"] += 1
            metrics["by_subtask"][subtask]["errors"] += 1
        elif result.correct:
            metrics["overall"]["correct"] += 1
            metrics["by_task_type"][task_type]["correct"] += 1
            metrics["by_subtask"][subtask]["correct"] += 1

    # Compute accuracies
    def add_accuracy(d):
        if d["total"] > 0:
            d["accuracy"] = d["correct"] / d["total"]
        else:
            d["accuracy"] = 0.0

    add_accuracy(metrics["overall"])
    for task_metrics in metrics["by_task_type"].values():
        add_accuracy(task_metrics)
    for subtask_metrics in metrics["by_subtask"].values():
        add_accuracy(subtask_metrics)

    return metrics


def print_report(metrics: dict, provider_name: str):
    """Print formatted benchmark report."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK REPORT: {provider_name}")
    print("=" * 60)

    # Overall accuracy
    overall = metrics["overall"]
    print(
        f"\nOverall Accuracy: {overall['accuracy']:.2%} ({overall['correct']}/{overall['total']})"
    )
    if overall["errors"] > 0:
        print(f"  Errors: {overall['errors']}")

    # By task type
    print("\n" + "-" * 40)
    print("Accuracy by Task Type:")
    print("-" * 40)

    task_order = [
        "1d_line",
        "1d_arrow",
        "1d_muller",
        "2d_circle",
        "2d_square",
        "2d_rectangle",
        "2d_triangle_equilateral",
        "2d_triangle_right",
    ]

    for task_type in task_order:
        if task_type in metrics["by_task_type"]:
            m = metrics["by_task_type"][task_type]
            print(
                f"  {task_type:30s}: {m['accuracy']:6.2%} ({m['correct']:3d}/{m['total']:3d})"
            )

    # By subtask
    print("\n" + "-" * 40)
    print("Accuracy by Subtask:")
    print("-" * 40)

    for subtask, m in sorted(metrics["by_subtask"].items()):
        print(
            f"  {subtask:30s}: {m['accuracy']:6.2%} ({m['correct']:3d}/{m['total']:3d})"
        )

    # Category summaries
    print("\n" + "-" * 40)
    print("Accuracy by Category:")
    print("-" * 40)

    # 1D tasks
    line_tasks = ["1d_line", "1d_arrow", "1d_muller"]
    line_correct = sum(
        metrics["by_task_type"].get(t, {}).get("correct", 0) for t in line_tasks
    )
    line_total = sum(
        metrics["by_task_type"].get(t, {}).get("total", 0) for t in line_tasks
    )
    if line_total > 0:
        print(
            f"  1D Line Comparison:             {line_correct / line_total:6.2%} ({line_correct:3d}/{line_total:3d})"
        )

    # 2D shape tasks
    shape_tasks = ["2d_circle", "2d_square", "2d_rectangle"]
    shape_correct = sum(
        metrics["by_task_type"].get(t, {}).get("correct", 0) for t in shape_tasks
    )
    shape_total = sum(
        metrics["by_task_type"].get(t, {}).get("total", 0) for t in shape_tasks
    )
    if shape_total > 0:
        print(
            f"  2D Shape Comparison:            {shape_correct / shape_total:6.2%} ({shape_correct:3d}/{shape_total:3d})"
        )

    # Triangle tasks
    tri_tasks = ["2d_triangle_equilateral", "2d_triangle_right"]
    tri_correct = sum(
        metrics["by_task_type"].get(t, {}).get("correct", 0) for t in tri_tasks
    )
    tri_total = sum(
        metrics["by_task_type"].get(t, {}).get("total", 0) for t in tri_tasks
    )
    if tri_total > 0:
        print(
            f"  2D Triangle Classification:     {tri_correct / tri_total:6.2%} ({tri_correct:3d}/{tri_total:3d})"
        )

    print("=" * 60)


def save_results(results: list[BenchmarkResult], metrics: dict, output_path: str):
    """Save detailed results to JSON."""
    output = {
        "metrics": {
            "overall": metrics["overall"],
            "by_task_type": dict(metrics["by_task_type"]),
            "by_subtask": dict(metrics["by_subtask"]),
        },
        "results": [
            {
                "image": result.sample.image_path,
                "task_type": result.sample.task_type,
                "subtask": result.sample.subtask,
                "question": result.sample.question,
                "ground_truth": result.sample.ground_truth,
                "model_response": result.model_response,
                "predicted": result.predicted,
                "correct": result.correct,
                "error": result.error,
            }
            for result in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLMs on visual geometry tasks"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai",
        choices=["openai", "openai-gpt5", "anthropic"],
        help="VLM provider to use (openai=gpt-4o, openai-gpt5=gpt-5.1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Specific model name (e.g., gpt-4o, claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Maximum number of samples to test (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="2821_finalProj_IMAGE_DATA",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress per-sample output"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading benchmark data from: {args.data_dir}")
    samples = load_benchmark_data(args.data_dir)
    print(f"Loaded {len(samples)} samples")

    # Initialize provider
    print(f"\nInitializing {args.model} provider...")
    provider = get_provider(args.model, args.model_name)
    print(f"Using: {provider.name}")

    # Run benchmark
    print(f"\nRunning benchmark with seed={args.seed}...")
    results = run_benchmark(
        provider=provider,
        samples=samples,
        seed=args.seed,
        max_samples=args.max_samples,
        verbose=not args.quiet,
    )

    # Compute and print metrics
    metrics = compute_metrics(results)
    print_report(metrics, provider.name)

    # Save results if requested
    if args.output:
        save_results(results, metrics, args.output)
    else:
        # Default output filename
        output_file = f"benchmark_results_{args.model}_{args.seed}.json"
        save_results(results, metrics, output_file)


if __name__ == "__main__":
    main()
