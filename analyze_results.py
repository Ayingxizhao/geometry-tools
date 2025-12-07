#!/usr/bin/env python3
"""
Comprehensive Analysis of Benchmark Results

Computes three key metrics:
1. Classification Accuracy - overall and per-task
2. Reflection Efficacy - how often reflection corrected vs. introduced errors (Condition A vs B only)
3. Average Rounds to Convergence - computational overhead (Condition A vs B only)
"""

import json
from pathlib import Path
from collections import defaultdict


def load_results(filepath):
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_classification_accuracy(data, name):
    """
    Compute classification accuracy.
    Accuracy = N_correct / N_total
    """
    metrics = data.get("metrics", {})
    overall = metrics.get("overall", {})
    by_task = metrics.get("by_task_type", {})
    by_subtask = metrics.get("by_subtask", {})

    print(f"\n{'='*70}")
    print(f"CLASSIFICATION ACCURACY: {name}")
    print(f"{'='*70}")

    # Overall
    correct = overall.get("correct", 0)
    total = overall.get("total", 0)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {accuracy:.4f} ({correct}/{total})")

    # By task type
    print(f"\n{'Task Type':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 62)

    task_order = [
        "1d_line", "1d_arrow", "1d_muller",
        "2d_circle", "2d_square", "2d_rectangle",
        "2d_triangle_equilateral", "2d_triangle_right"
    ]

    for task in task_order:
        if task in by_task:
            t = by_task[task]
            acc = t.get("accuracy", t.get("correct", 0) / t.get("total", 1))
            print(f"{task:<30} {acc:>10.4f} {t.get('correct', 0):>10} {t.get('total', 0):>10}")

    # By subtask
    print(f"\n{'Subtask':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 62)
    for subtask, s in sorted(by_subtask.items()):
        acc = s.get("accuracy", s.get("correct", 0) / s.get("total", 1))
        print(f"{subtask:<30} {acc:>10.4f} {s.get('correct', 0):>10} {s.get('total', 0):>10}")

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_task": by_task,
        "by_subtask": by_subtask
    }


def compute_reflection_efficacy(data, name):
    """
    Compute Reflection Efficacy for Condition A or B.

    Tracks cases where initial answer (Round 1) differs from final answer.
    Categories:
    - Corrected: Initial wrong -> Final correct (reflection helped)
    - Degraded: Initial correct -> Final wrong (reflection hurt)
    - Changed but still wrong: Initial wrong -> Final wrong (different answer)
    - Changed but still correct: Initial correct -> Final correct (different answer) - rare
    """
    results = data.get("results", [])

    stats = {
        "total_samples": 0,
        "single_round": 0,  # No reflection needed
        "multi_round": 0,   # Had reflection
        "answer_changed": 0,
        "corrected": 0,     # Wrong -> Correct
        "degraded": 0,      # Correct -> Wrong
        "changed_still_wrong": 0,
        "changed_still_correct": 0,
    }

    for result in results:
        stats["total_samples"] += 1
        rounds = result.get("reflection_rounds", [])
        num_rounds = result.get("num_rounds", len(rounds))
        ground_truth = result.get("ground_truth")
        final_correct = result.get("correct", False)

        if num_rounds <= 1 or len(rounds) <= 1:
            stats["single_round"] += 1
            continue

        stats["multi_round"] += 1

        # Get initial answer from round 1
        initial_answer_raw = rounds[0].get("answer", "") if rounds else ""
        initial_answer = (initial_answer_raw or "").upper()
        final_answer_raw = result.get("final_answer", "")
        final_answer = (final_answer_raw or "").upper()

        # Convert to boolean for comparison
        initial_pred = initial_answer == "YES"
        final_pred = final_answer == "YES"

        # Check if answer changed
        if initial_answer != final_answer:
            stats["answer_changed"] += 1

            initial_correct = (initial_pred == ground_truth)

            if not initial_correct and final_correct:
                stats["corrected"] += 1
            elif initial_correct and not final_correct:
                stats["degraded"] += 1
            elif not initial_correct and not final_correct:
                stats["changed_still_wrong"] += 1
            else:  # initial_correct and final_correct but different answer - shouldn't happen often
                stats["changed_still_correct"] += 1

    print(f"\n{'='*70}")
    print(f"REFLECTION EFFICACY: {name}")
    print(f"{'='*70}")
    print(f"\nTotal Samples: {stats['total_samples']}")
    print(f"Single Round (no reflection): {stats['single_round']}")
    print(f"Multi Round (had reflection): {stats['multi_round']}")
    print(f"\nOf {stats['multi_round']} multi-round samples:")
    print(f"  Answer Changed: {stats['answer_changed']}")
    print(f"    - Corrected (wrong -> correct): {stats['corrected']}")
    print(f"    - Degraded (correct -> wrong): {stats['degraded']}")
    print(f"    - Changed but still wrong: {stats['changed_still_wrong']}")
    print(f"    - Changed but still correct: {stats['changed_still_correct']}")

    # Compute efficacy rate
    if stats["answer_changed"] > 0:
        efficacy = stats["corrected"] / stats["answer_changed"]
        degradation = stats["degraded"] / stats["answer_changed"]
        print(f"\n  Correction Rate: {efficacy:.2%} ({stats['corrected']}/{stats['answer_changed']})")
        print(f"  Degradation Rate: {degradation:.2%} ({stats['degraded']}/{stats['answer_changed']})")
        net_benefit = stats["corrected"] - stats["degraded"]
        print(f"  Net Benefit: {net_benefit:+d} samples")

    return stats


def compute_avg_rounds(data, name):
    """
    Compute Average Rounds to Convergence.
    r_bar = (1/N) * sum(r_i), where r_i in {1, 2, 3}
    """
    results = data.get("results", [])
    metrics = data.get("metrics", {})

    # Can use pre-computed if available
    if "avg_rounds" in metrics:
        avg_rounds = metrics["avg_rounds"]
    else:
        total_rounds = sum(r.get("num_rounds", 1) for r in results)
        avg_rounds = total_rounds / len(results) if results else 0

    # Distribution of rounds
    round_dist = defaultdict(int)
    confidence_at_end = defaultdict(int)

    for result in results:
        num_rounds = result.get("num_rounds", 1)
        round_dist[num_rounds] += 1

        # Get final confidence
        rounds = result.get("reflection_rounds", [])
        if rounds:
            final_conf = rounds[-1].get("confidence", "unknown")
            confidence_at_end[final_conf] += 1

    print(f"\n{'='*70}")
    print(f"AVERAGE ROUNDS TO CONVERGENCE: {name}")
    print(f"{'='*70}")
    print(f"\nAverage Rounds: {avg_rounds:.3f}")
    print(f"\nRound Distribution:")
    for r in sorted(round_dist.keys()):
        pct = round_dist[r] / len(results) * 100 if results else 0
        print(f"  Round {r}: {round_dist[r]:4d} samples ({pct:5.1f}%)")

    print(f"\nFinal Confidence Distribution:")
    for conf in ["high", "medium", "low", "uncertain"]:
        if conf in confidence_at_end:
            pct = confidence_at_end[conf] / len(results) * 100 if results else 0
            print(f"  {conf:12s}: {confidence_at_end[conf]:4d} ({pct:5.1f}%)")

    return {
        "avg_rounds": avg_rounds,
        "round_distribution": dict(round_dist),
        "confidence_distribution": dict(confidence_at_end)
    }


def generate_latex_table(baseline_acc, cond_a_acc, cond_b_acc):
    """Generate LaTeX table for classification accuracy."""
    print(f"\n{'='*70}")
    print("LATEX TABLE: Classification Accuracy by Task Type")
    print(f"{'='*70}")

    task_order = [
        ("1d_line", "Line"),
        ("1d_arrow", "Arrow"),
        ("1d_muller", "Müller-Lyer"),
        ("2d_circle", "Circle"),
        ("2d_square", "Square"),
        ("2d_rectangle", "Rectangle"),
        ("2d_triangle_equilateral", "Equilateral △"),
        ("2d_triangle_right", "Right △"),
    ]

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l|ccc}")
    print(r"\hline")
    print(r"\textbf{Task Type} & \textbf{Baseline} & \textbf{Condition A} & \textbf{Condition B} \\")
    print(r"\hline")

    for task_key, task_name in task_order:
        base = baseline_acc["by_task"].get(task_key, {}).get("accuracy", 0)
        ca = cond_a_acc["by_task"].get(task_key, {}).get("accuracy", 0)
        cb = cond_b_acc["by_task"].get(task_key, {}).get("accuracy", 0)
        print(f"{task_name} & {base:.2%} & {ca:.2%} & {cb:.2%} \\\\")

    print(r"\hline")
    print(f"\\textbf{{Overall}} & \\textbf{{{baseline_acc['overall_accuracy']:.2%}}} & \\textbf{{{cond_a_acc['overall_accuracy']:.2%}}} & \\textbf{{{cond_b_acc['overall_accuracy']:.2%}}} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Classification accuracy across experimental conditions}")
    print(r"\label{tab:accuracy}")
    print(r"\end{table}")


def generate_comparison_table(stats_a, stats_b, rounds_a, rounds_b):
    """Generate comparison table for Condition A vs B."""
    print(f"\n{'='*70}")
    print("LATEX TABLE: Reflection Metrics Comparison")
    print(f"{'='*70}")

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l|cc}")
    print(r"\hline")
    print(r"\textbf{Metric} & \textbf{Condition A} & \textbf{Condition B} \\")
    print(r" & (Symbolic Reflection) & (Visual Reflection) \\")
    print(r"\hline")

    # Average rounds
    print(f"Avg. Rounds ($\\bar{{r}}$) & {rounds_a['avg_rounds']:.3f} & {rounds_b['avg_rounds']:.3f} \\\\")

    # Multi-round samples
    print(f"Multi-round Samples & {stats_a['multi_round']} & {stats_b['multi_round']} \\\\")

    # Answer changed
    print(f"Answers Changed & {stats_a['answer_changed']} & {stats_b['answer_changed']} \\\\")

    # Corrected
    print(f"Corrected (Wrong$\\rightarrow$Correct) & {stats_a['corrected']} & {stats_b['corrected']} \\\\")

    # Degraded
    print(f"Degraded (Correct$\\rightarrow$Wrong) & {stats_a['degraded']} & {stats_b['degraded']} \\\\")

    # Net benefit
    net_a = stats_a['corrected'] - stats_a['degraded']
    net_b = stats_b['corrected'] - stats_b['degraded']
    print(f"Net Benefit & {net_a:+d} & {net_b:+d} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Reflection efficacy comparison}")
    print(r"\label{tab:reflection}")
    print(r"\end{table}")


def main():
    results_dir = Path("results")

    # Load all three result files
    baseline_path = results_dir / "benchmark_results_openai-gpt5_42.json"
    cond_a_path = results_dir / "benchmark_v2_A_42.json"
    cond_b_path = results_dir / "benchmark_v2_B_42.json"

    print("Loading results...")
    baseline = load_results(baseline_path)
    cond_a = load_results(cond_a_path)
    cond_b = load_results(cond_b_path)

    # =========================================================================
    # 1. Classification Accuracy (all three)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" METRIC 1: CLASSIFICATION ACCURACY")
    print("=" * 70)

    baseline_acc = compute_classification_accuracy(baseline, "Baseline (VLM Direct)")
    cond_a_acc = compute_classification_accuracy(cond_a, "Condition A (Symbolic Reflection)")
    cond_b_acc = compute_classification_accuracy(cond_b, "Condition B (Visual Reflection)")

    # =========================================================================
    # 2. Reflection Efficacy (Condition A vs B only)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" METRIC 2: REFLECTION EFFICACY")
    print("=" * 70)

    stats_a = compute_reflection_efficacy(cond_a, "Condition A (Symbolic Reflection)")
    stats_b = compute_reflection_efficacy(cond_b, "Condition B (Visual Reflection)")

    # =========================================================================
    # 3. Average Rounds to Convergence (Condition A vs B only)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" METRIC 3: AVERAGE ROUNDS TO CONVERGENCE")
    print("=" * 70)

    rounds_a = compute_avg_rounds(cond_a, "Condition A (Symbolic Reflection)")
    rounds_b = compute_avg_rounds(cond_b, "Condition B (Visual Reflection)")

    # =========================================================================
    # Generate LaTeX Tables
    # =========================================================================
    print("\n" + "=" * 70)
    print(" LATEX TABLES FOR PAPER")
    print("=" * 70)

    generate_latex_table(baseline_acc, cond_a_acc, cond_b_acc)
    generate_comparison_table(stats_a, stats_b, rounds_a, rounds_b)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(" EXECUTIVE SUMMARY")
    print("=" * 70)

    print(f"\n1. CLASSIFICATION ACCURACY:")
    print(f"   Baseline:    {baseline_acc['overall_accuracy']:.2%} ({baseline_acc['correct']}/{baseline_acc['total']})")
    print(f"   Condition A: {cond_a_acc['overall_accuracy']:.2%} ({cond_a_acc['correct']}/{cond_a_acc['total']})")
    print(f"   Condition B: {cond_b_acc['overall_accuracy']:.2%} ({cond_b_acc['correct']}/{cond_b_acc['total']})")

    improvement_a = (cond_a_acc['overall_accuracy'] - baseline_acc['overall_accuracy']) * 100
    improvement_b = (cond_b_acc['overall_accuracy'] - baseline_acc['overall_accuracy']) * 100
    print(f"\n   Improvement over Baseline:")
    print(f"     Condition A: +{improvement_a:.1f} percentage points")
    print(f"     Condition B: +{improvement_b:.1f} percentage points")

    print(f"\n2. REFLECTION EFFICACY:")
    print(f"   Condition A: {stats_a['corrected']} corrected, {stats_a['degraded']} degraded (net: {stats_a['corrected']-stats_a['degraded']:+d})")
    print(f"   Condition B: {stats_b['corrected']} corrected, {stats_b['degraded']} degraded (net: {stats_b['corrected']-stats_b['degraded']:+d})")

    print(f"\n3. AVERAGE ROUNDS TO CONVERGENCE:")
    print(f"   Condition A: {rounds_a['avg_rounds']:.3f}")
    print(f"   Condition B: {rounds_b['avg_rounds']:.3f}")

    # Round 1 high-confidence rate
    r1_high_a = rounds_a['round_distribution'].get(1, 0) / cond_a_acc['total'] * 100
    r1_high_b = rounds_b['round_distribution'].get(1, 0) / cond_b_acc['total'] * 100
    print(f"\n   Single-round (high confidence) rate:")
    print(f"     Condition A: {r1_high_a:.1f}%")
    print(f"     Condition B: {r1_high_b:.1f}%")


if __name__ == "__main__":
    main()
