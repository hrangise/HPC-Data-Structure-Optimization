"""
Final Project: Optimization in High-Performance Computing (HPC)
Technique: Replace repeated linear list membership searches with hash-based set membership.

Run:
    python code.py

The program:
1. Builds a deterministic synthetic workload.
2. Checks that list-based and set-based implementations return identical results.
3. Measures repeated membership-search time using timeit.
4. Reports speedup and shallow container memory sizes.
5. Writes benchmark_results.csv for use in the report.

No third-party packages are required.
"""

import csv
import random
import statistics
import sys
import timeit
from pathlib import Path


def list_membership(data, queries):
    """Baseline: each query may scan the list from beginning to end."""
    return [query in data for query in queries]


def set_membership(data_set, queries):
    """Optimized: hash-based membership lookup."""
    return [query in data_set for query in queries]


def build_workload(data_size=20_000, query_count=5_000, seed=42):
    """Create repeatable data with an equal mix of hits and misses."""
    rng = random.Random(seed)
    data = list(range(data_size))
    hits = [rng.randrange(data_size) for _ in range(query_count // 2)]
    misses = [data_size + rng.randrange(data_size) for _ in range(query_count - len(hits))]
    queries = hits + misses
    rng.shuffle(queries)
    return data, queries


def benchmark():
    data, queries = build_workload()
    data_set = set(data)

    # Verify functional equivalence before measuring speed.
    baseline_result = list_membership(data, queries)
    optimized_result = set_membership(data_set, queries)
    if baseline_result != optimized_result:
        raise AssertionError("The two implementations returned different results.")

    # Setup (creating the list/set and queries) is outside the timed functions.
    baseline_timer = timeit.Timer(lambda: list_membership(data, queries))
    optimized_timer = timeit.Timer(lambda: set_membership(data_set, queries))

    repetitions = 5
    baseline_samples = baseline_timer.repeat(repeat=repetitions, number=1)
    optimized_samples = optimized_timer.repeat(repeat=repetitions, number=1)

    # The minimum repeat is used as a lower-bound estimate, as recommended
    # by Python's timeit documentation; all samples are still reported.
    baseline_best = min(baseline_samples)
    optimized_best = min(optimized_samples)
    speedup = baseline_best / optimized_best if optimized_best else float("inf")
    improvement = (1 - optimized_best / baseline_best) * 100 if baseline_best else 0

    rows = [
        {
            "implementation": "List (baseline)",
            "best_seconds": baseline_best,
            "median_seconds": statistics.median(baseline_samples),
            "shallow_container_bytes": sys.getsizeof(data),
        },
        {
            "implementation": "Set (optimized)",
            "best_seconds": optimized_best,
            "median_seconds": statistics.median(optimized_samples),
            "shallow_container_bytes": sys.getsizeof(data_set),
        },
    ]

    output_path = Path(__file__).with_name("benchmark_results.csv")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 68)
    print("HPC DATA-STRUCTURE OPTIMIZATION: LIST VS. SET MEMBERSHIP")
    print("=" * 68)
    print(f"Data elements: {len(data):,}")
    print(f"Membership queries: {len(queries):,} (50% hits, 50% misses)")
    print(f"Correctness check: PASSED ({len(queries):,} results match)")
    print("\nTiming results (seconds for one complete query batch):")
    print(f"List baseline samples: {[round(x, 6) for x in baseline_samples]}")
    print(f"Set optimized samples: {[round(x, 6) for x in optimized_samples]}")
    print(f"Best list time:         {baseline_best:.6f}")
    print(f"Best set time:          {optimized_best:.6f}")
    print(f"Measured speedup:       {speedup:.2f}x")
    print(f"Time reduction:         {improvement:.2f}%")
    print("\nShallow container sizes (does not include referenced integer objects):")
    print(f"List: {sys.getsizeof(data):,} bytes")
    print(f"Set:  {sys.getsizeof(data_set):,} bytes")
    print(f"\nCSV results saved to: {output_path}")
    print("\nInterpretation: set lookup is expected to be average O(1), while")
    print("list membership is O(n). Set construction and extra memory are trade-offs.")
    print("Results depend on Python version, hardware, workload, and data distribution.")


if __name__ == "__main__":
    benchmark()
