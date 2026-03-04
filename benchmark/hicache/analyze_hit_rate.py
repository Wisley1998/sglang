"""
Analyze Mamba vs Attention prefix hit rate from benchmark results.

Loads experiment result JSONL files, aggregates by scenario/request_rate,
generates comparison charts and a summary markdown table.

Usage:
    python benchmark/hicache/analyze_hit_rate.py \
        --results-dir results/ \
        --output-dir analysis/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: str) -> list:
    """Load all JSONL result files from a directory."""
    results = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["_source_file"] = path.name
                    results.append(record)
    return results


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return mean, mean, mean
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


def group_results(results: list) -> dict:
    """Group results by (dataset_name, request_rate)."""
    groups = defaultdict(list)
    for r in results:
        key = (r.get("dataset_name", "unknown"), r.get("request_rate", 0))
        groups[key].append(r)
    return dict(groups)


def generate_charts(groups: dict, output_dir: str):
    """Generate comparison charts if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect data for plotting
    scenarios = sorted(groups.keys())
    labels = [f"{ds}\nrr={rr}" for ds, rr in scenarios]

    attn_means, attn_lows, attn_highs = [], [], []
    mamba_means, mamba_lows, mamba_highs = [], [], []
    ttft_means, ttft_lows, ttft_highs = [], [], []

    for key in scenarios:
        records = groups[key]
        attn_vals = [r.get("attn_token_hit_rate", 0) for r in records]
        mamba_vals = [r.get("mamba_token_hit_rate", 0) for r in records]
        ttft_vals = [r.get("mean_ttft_ms", 0) for r in records]

        m, lo, hi = bootstrap_ci(attn_vals)
        attn_means.append(m)
        attn_lows.append(m - lo)
        attn_highs.append(hi - m)

        m, lo, hi = bootstrap_ci(mamba_vals)
        mamba_means.append(m)
        mamba_lows.append(m - lo)
        mamba_highs.append(hi - m)

        m, lo, hi = bootstrap_ci(ttft_vals)
        ttft_means.append(m)
        ttft_lows.append(m - lo)
        ttft_highs.append(hi - m)

    x = np.arange(len(scenarios))
    width = 0.35

    # Chart 1: Attention vs Mamba Token Hit Rate
    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.5), 6))
    bars1 = ax.bar(
        x - width / 2,
        attn_means,
        width,
        yerr=[attn_lows, attn_highs],
        label="Attn Potential Hit Rate",
        capsize=4,
    )
    bars2 = ax.bar(
        x + width / 2,
        mamba_means,
        width,
        yerr=[mamba_lows, mamba_highs],
        label="Mamba Hit Rate",
        capsize=4,
    )
    ax.set_ylabel("Token Hit Rate")
    ax.set_title("Attention vs Mamba Token Hit Rate by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hit_rate_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/hit_rate_comparison.png")

    # Chart 2: TTFT by Scenario
    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.5), 6))
    ax.bar(
        x,
        ttft_means,
        width * 1.5,
        yerr=[ttft_lows, ttft_highs],
        label="Mean TTFT",
        capsize=4,
        color="steelblue",
    )
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Mean Time to First Token by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ttft_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/ttft_comparison.png")


def generate_summary_table(groups: dict, output_dir: str):
    """Generate a markdown summary table."""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# Mamba vs Attention Prefix Hit Rate Summary",
        "",
        "| Scenario | Request Rate | Runs | Attn Hit Rate (95% CI) | Mamba Hit Rate (95% CI) | Mean TTFT ms (95% CI) |",
        "|----------|-------------|------|------------------------|-------------------------|-----------------------|",
    ]

    for key in sorted(groups.keys()):
        ds, rr = key
        records = groups[key]
        n = len(records)

        attn_vals = [r.get("attn_token_hit_rate", 0) for r in records]
        mamba_vals = [r.get("mamba_token_hit_rate", 0) for r in records]
        ttft_vals = [r.get("mean_ttft_ms", 0) for r in records]

        am, alo, ahi = bootstrap_ci(attn_vals)
        mm, mlo, mhi = bootstrap_ci(mamba_vals)
        tm, tlo, thi = bootstrap_ci(ttft_vals)

        lines.append(
            f"| {ds} | {rr} | {n} "
            f"| {am:.4f} [{alo:.4f}, {ahi:.4f}] "
            f"| {mm:.4f} [{mlo:.4f}, {mhi:.4f}] "
            f"| {tm:.1f} [{tlo:.1f}, {thi:.1f}] |"
        )

    lines.append("")

    output_path = os.path.join(output_dir, "summary.md")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Mamba vs Attention prefix hit rate results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory to save analysis outputs",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result records")
    groups = group_results(results)
    print(f"Found {len(groups)} scenario groups")

    generate_charts(groups, args.output_dir)
    generate_summary_table(groups, args.output_dir)


if __name__ == "__main__":
    main()
