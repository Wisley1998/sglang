#!/usr/bin/env python3
"""
LFC vs Standard Benchmark Visualization

Reads JSONL results from run_benchmarks.py and generates:
  1. TTFT CDF plots (one per experiment)
  2. TPOT CDF plots (one per experiment)
  3. Throughput timeline plots (one per experiment)
  4. Summary bar chart (all experiments)

Usage:
    python benchmark/lfc/plot_results.py [OPTIONS]

Options:
    --results-dir  Directory with JSONL results (default: benchmark/lfc/results)
    --plots-dir    Output directory for PNG plots (default: benchmark/lfc/plots)
    --experiments  Comma-separated experiment IDs (default: A,B,C,D,E,F)
    --dpi          Plot DPI (default: 150)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Style constants ---
COLOR_STD = "#2196F3"  # Blue
COLOR_LFC = "#F44336"  # Red
FIGURE_SIZE = (10, 6)
FONT_SIZE = 12
DPI = 150

EXPERIMENT_NAMES = {
    "A": "high_qps",
    "B": "many_groups",
    "C": "long_prefix",
    "D": "baseline",
    "E": "large_batch",
    "F": "very_high_qps",
}


def parse_args():
    parser = argparse.ArgumentParser(description="LFC Benchmark Visualization")
    parser.add_argument(
        "--results-dir",
        default="benchmark/lfc/results",
        help="Directory with JSONL results",
    )
    parser.add_argument(
        "--plots-dir",
        default="benchmark/lfc/plots",
        help="Output directory for PNG plots",
    )
    parser.add_argument(
        "--experiments",
        default="A,B,C,D,E,F",
        help="Comma-separated experiment IDs",
    )
    parser.add_argument("--dpi", type=int, default=DPI, help="Plot DPI")
    return parser.parse_args()


def load_result(path: str) -> dict | None:
    """Load a single JSONL result file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.loads(f.read())


def extract_ttfts(data: dict) -> np.ndarray:
    """Extract TTFT values in ms from result data.

    ttfts field is a list of floats in seconds.
    """
    raw = data.get("ttfts", [])
    values = []
    for item in raw:
        if isinstance(item, list):
            if len(item) > 0:
                values.append(item[0])
        else:
            values.append(item)
    # Convert seconds to milliseconds
    return np.array(values) * 1000.0


def extract_tpots(data: dict) -> np.ndarray:
    """Extract TPOT (mean ITL per request) values in ms.

    itls field is a list of lists of floats in seconds.
    """
    itls = data.get("itls", [])
    tpots = []
    for req_itls in itls:
        if isinstance(req_itls, list) and len(req_itls) > 0:
            tpots.append(np.mean(req_itls) * 1000.0)  # seconds to ms
    return np.array(tpots)


def compute_cdf(values: np.ndarray):
    """Compute CDF from array of values. Returns (sorted_values, cdf)."""
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, cdf


def reconstruct_throughput_timeline(
    data: dict, bin_size: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct output tokens/sec timeline from request data.

    Uses Poisson arrival times (seed=1) + TTFT + ITLs to determine
    when each token was generated, then bins into tokens/sec.
    All raw timing values (ttfts, itls) are in seconds.
    """
    # Use raw ttfts in seconds (not the ms-converted extract_ttfts)
    raw_ttfts = data.get("ttfts", [])
    ttfts_s = []
    for item in raw_ttfts:
        if isinstance(item, list):
            if len(item) > 0:
                ttfts_s.append(item[0])
        else:
            ttfts_s.append(item)
    ttfts_s = np.array(ttfts_s)

    itls_list = data.get("itls", [])
    request_rate = data.get("request_rate", 4)
    n_requests = len(ttfts_s)

    if n_requests == 0:
        return np.array([]), np.array([])

    # Reconstruct Poisson arrival times (same seed as benchmark)
    rng = np.random.RandomState(1)
    intervals = rng.exponential(1.0 / request_rate, size=n_requests)
    arrivals = np.cumsum(intervals)  # seconds

    # Collect all token generation timestamps
    token_times = []
    for i in range(n_requests):
        if i >= len(itls_list):
            break
        req_itls = itls_list[i]
        if not isinstance(req_itls, list) or len(req_itls) == 0:
            continue

        # First token time = arrival + TTFT (both in seconds)
        first_token_time = arrivals[i] + ttfts_s[i]
        token_times.append(first_token_time)

        # Subsequent tokens (ITLs are in seconds)
        t = first_token_time
        for itl_s in req_itls:
            t += itl_s
            token_times.append(t)

    if len(token_times) == 0:
        return np.array([]), np.array([])

    token_times = np.array(token_times)
    max_time = token_times.max()

    # Bin into time intervals
    bins = np.arange(0, max_time + bin_size, bin_size)
    counts, _ = np.histogram(token_times, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    throughput = counts / bin_size  # tokens per second

    return bin_centers, throughput


def plot_cdf(
    std_values: np.ndarray,
    lfc_values: np.ndarray,
    xlabel: str,
    title: str,
    output_path: str,
    dpi: int = DPI,
):
    """Plot CDF comparison of Standard vs LFC."""
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Standard CDF
    if len(std_values) > 0:
        sx, sy = compute_cdf(std_values)
        std_mean = np.mean(std_values)
        std_median = np.median(std_values)
        std_p90 = np.percentile(std_values, 90)
        std_p99 = np.percentile(std_values, 99)
        ax.plot(
            sx, sy,
            color=COLOR_STD,
            linewidth=2,
            label=f"Standard (mean={std_mean:.0f}, med={std_median:.0f})",
        )
        ax.axvline(std_p90, color=COLOR_STD, linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(std_p99, color=COLOR_STD, linestyle=":", alpha=0.5, linewidth=1)

    # LFC CDF
    if len(lfc_values) > 0:
        lx, ly = compute_cdf(lfc_values)
        lfc_mean = np.mean(lfc_values)
        lfc_median = np.median(lfc_values)
        lfc_p90 = np.percentile(lfc_values, 90)
        lfc_p99 = np.percentile(lfc_values, 99)
        ax.plot(
            lx, ly,
            color=COLOR_LFC,
            linewidth=2,
            label=f"LFC (mean={lfc_mean:.0f}, med={lfc_median:.0f})",
        )
        ax.axvline(lfc_p90, color=COLOR_LFC, linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(lfc_p99, color=COLOR_LFC, linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_throughput_timeline(
    std_data: dict | None,
    lfc_data: dict | None,
    title: str,
    output_path: str,
    dpi: int = DPI,
):
    """Plot throughput (tokens/sec) timeline comparison."""
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if std_data is not None:
        t, tp = reconstruct_throughput_timeline(std_data)
        if len(t) > 0:
            # Smooth with rolling average (window=3)
            if len(tp) >= 3:
                kernel = np.ones(3) / 3
                tp_smooth = np.convolve(tp, kernel, mode="same")
            else:
                tp_smooth = tp
            ax.plot(
                t, tp_smooth,
                color=COLOR_STD,
                linewidth=2,
                label=f"Standard ({std_data.get('output_throughput', 0):.0f} tok/s avg)",
                alpha=0.8,
            )

    if lfc_data is not None:
        t, tp = reconstruct_throughput_timeline(lfc_data)
        if len(t) > 0:
            if len(tp) >= 3:
                kernel = np.ones(3) / 3
                tp_smooth = np.convolve(tp, kernel, mode="same")
            else:
                tp_smooth = tp
            ax.plot(
                t, tp_smooth,
                color=COLOR_LFC,
                linewidth=2,
                label=f"LFC ({lfc_data.get('output_throughput', 0):.0f} tok/s avg)",
                alpha=0.8,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Output Tokens/sec")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_bars(
    results: dict[str, tuple[dict | None, dict | None]],
    experiment_ids: list[str],
    output_path: str,
    dpi: int = DPI,
):
    """Plot summary bar chart comparing Standard vs LFC across experiments."""
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    metrics = [
        ("Mean TTFT (ms)", "mean_ttft_ms"),
        ("P90 TTFT (ms)", "p99_ttft_ms"),  # Using P99 as P90 not in output
        ("Throughput (tok/s)", "output_throughput"),
    ]

    # Check if p90_ttft_ms exists, otherwise fall back
    sample = None
    for eid in experiment_ids:
        if eid in results and results[eid][0] is not None:
            sample = results[eid][0]
            break
    if sample and "p90_ttft_ms" in sample:
        metrics[1] = ("P90 TTFT (ms)", "p90_ttft_ms")

    x = np.arange(len(experiment_ids))
    width = 0.35

    for ax_idx, (label, key) in enumerate(metrics):
        ax = axes[ax_idx]
        std_vals = []
        lfc_vals = []

        for eid in experiment_ids:
            std_data, lfc_data = results.get(eid, (None, None))
            std_vals.append(std_data.get(key, 0) if std_data else 0)
            lfc_vals.append(lfc_data.get(key, 0) if lfc_data else 0)

        bars_std = ax.bar(
            x - width / 2,
            std_vals,
            width,
            label="Standard",
            color=COLOR_STD,
            alpha=0.8,
        )
        bars_lfc = ax.bar(
            x + width / 2,
            lfc_vals,
            width,
            label="LFC",
            color=COLOR_LFC,
            alpha=0.8,
        )

        # Add delta labels on LFC bars
        for i, (sv, lv) in enumerate(zip(std_vals, lfc_vals)):
            if sv > 0:
                delta_pct = (lv - sv) / sv * 100
                va = "bottom" if delta_pct >= 0 else "top"
                ax.annotate(
                    f"{delta_pct:+.0f}%",
                    xy=(x[i] + width / 2, lv),
                    ha="center",
                    va=va,
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{eid}\n{EXPERIMENT_NAMES[eid]}" for eid in experiment_ids],
            fontsize=9,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("LFC vs Standard: Summary Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()

    # Resolve relative paths from script directory
    script_dir = Path(__file__).resolve().parent
    if not os.path.isabs(args.results_dir):
        args.results_dir = str(script_dir / Path(args.results_dir).name)
    if not os.path.isabs(args.plots_dir):
        args.plots_dir = str(script_dir / Path(args.plots_dir).name)

    os.makedirs(args.plots_dir, exist_ok=True)

    experiment_ids = [x.strip() for x in args.experiments.split(",")]

    print(f"Results dir: {args.results_dir}")
    print(f"Plots dir: {args.plots_dir}")
    print(f"Experiments: {experiment_ids}")

    # Load all results
    results: dict[str, tuple[dict | None, dict | None]] = {}
    for eid in experiment_ids:
        std_path = os.path.join(args.results_dir, f"std_{eid}.jsonl")
        lfc_path = os.path.join(args.results_dir, f"lfc_{eid}.jsonl")
        std_data = load_result(std_path)
        lfc_data = load_result(lfc_path)
        results[eid] = (std_data, lfc_data)

        if std_data is None and lfc_data is None:
            print(f"  Warning: no results for experiment {eid}")
        elif std_data is None:
            print(f"  Warning: no Standard results for experiment {eid}")
        elif lfc_data is None:
            print(f"  Warning: no LFC results for experiment {eid}")

    # Generate per-experiment plots
    for eid in experiment_ids:
        std_data, lfc_data = results[eid]
        name = EXPERIMENT_NAMES.get(eid, eid)

        if std_data is None and lfc_data is None:
            print(f"\nSkipping experiment {eid} ({name}): no results")
            continue

        print(f"\nPlotting experiment {eid} ({name})...")

        # TTFT CDF
        std_ttfts = extract_ttfts(std_data) if std_data else np.array([])
        lfc_ttfts = extract_ttfts(lfc_data) if lfc_data else np.array([])
        if len(std_ttfts) > 0 or len(lfc_ttfts) > 0:
            plot_cdf(
                std_ttfts,
                lfc_ttfts,
                xlabel="TTFT (ms)",
                title=f"TTFT CDF — {eid}: {name}",
                output_path=os.path.join(args.plots_dir, f"{eid}_ttft_cdf.png"),
                dpi=args.dpi,
            )

        # TPOT CDF
        std_tpots = extract_tpots(std_data) if std_data else np.array([])
        lfc_tpots = extract_tpots(lfc_data) if lfc_data else np.array([])
        if len(std_tpots) > 0 or len(lfc_tpots) > 0:
            plot_cdf(
                std_tpots,
                lfc_tpots,
                xlabel="TPOT (ms)",
                title=f"TPOT CDF — {eid}: {name}",
                output_path=os.path.join(args.plots_dir, f"{eid}_tpot_cdf.png"),
                dpi=args.dpi,
            )

        # Throughput timeline
        plot_throughput_timeline(
            std_data,
            lfc_data,
            title=f"Throughput Timeline — {eid}: {name}",
            output_path=os.path.join(
                args.plots_dir, f"{eid}_throughput_timeline.png"
            ),
            dpi=args.dpi,
        )

    # Summary bar chart
    print("\nPlotting summary comparison...")
    # Only include experiments that have at least one result
    valid_ids = [
        eid
        for eid in experiment_ids
        if results[eid][0] is not None or results[eid][1] is not None
    ]
    if valid_ids:
        plot_summary_bars(
            results,
            valid_ids,
            output_path=os.path.join(args.plots_dir, "summary_comparison.png"),
            dpi=args.dpi,
        )

    print(f"\nAll plots saved to: {args.plots_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
