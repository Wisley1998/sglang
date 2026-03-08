#!/usr/bin/env python3
"""
Controlled variance test: run the same experiment N times per mode,
alternating LFC / Standard to eliminate temporal bias.

Design:
  - Single experiment (D_baseline by default)
  - N rounds (default 5), each round = 1 LFC run + 1 Standard run
  - Fresh server per run
  - Results collected into JSON for statistical analysis
  - Prints mean, std, min, max, and 95% CI for key metrics

Usage:
    python benchmark/lfc/run_variance_test.py [OPTIONS]

Options:
    --rounds        Number of rounds (default: 5)
    --experiment    Experiment ID (default: D)
    --model         Model path
    --tp            Tensor parallelism (default: 4)
    --port          Server port (default: 30000)
    --mem-fraction  Static memory fraction (default: 0.60)
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Experiment:
    id: str
    name: str
    prefix_len: int
    num_prompts: int
    groups: int
    per_group: int
    rate: float
    question_len: int


EXPERIMENTS = {
    "A": Experiment("A", "high_qps", 4096, 128, 8, 16, 8, 128),
    "B": Experiment("B", "many_groups", 4096, 128, 16, 8, 4, 128),
    "C": Experiment("C", "long_prefix", 8192, 128, 8, 16, 4, 128),
    "D": Experiment("D", "baseline", 4096, 128, 8, 16, 4, 128),
    "E": Experiment("E", "large_batch", 4096, 256, 8, 32, 4, 128),
}


def parse_args():
    parser = argparse.ArgumentParser(description="LFC variance test")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--experiment", default="D")
    parser.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction", type=float, default=0.60)
    parser.add_argument("--server-timeout", type=int, default=900)
    return parser.parse_args()


def wait_for_server(port: int, timeout: int = 600) -> bool:
    import urllib.request
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="GET"), timeout=5
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def start_server(args, lfc: bool, log_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    # Disable strict memory check to avoid false positives
    env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    if lfc:
        env["SGLANG_LFC_ENABLED"] = "1"
    else:
        env.pop("SGLANG_LFC_ENABLED", None)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--tp", str(args.tp),
        "--port", str(args.port),
        "--mem-fraction-static", str(args.mem_fraction),
    ]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    proc._log_file = log_file

    if not wait_for_server(args.port, timeout=args.server_timeout):
        log_file.flush()
        log_file.close()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        raise RuntimeError(f"Server failed to start (log: {log_path})")

    return proc


def kill_server(proc: subprocess.Popen):
    if proc is None:
        return
    log_file = getattr(proc, "_log_file", None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception:
            pass


def run_benchmark(exp: Experiment, model: str, port: int, output_file: str) -> dict:
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--model", model,
        "--dataset-name", "generated-shared-prefix",
        "--gsp-system-prompt-len", str(exp.prefix_len),
        "--gsp-question-len", str(exp.question_len),
        "--gsp-output-len", "256",
        "--gsp-num-groups", str(exp.groups),
        "--gsp-prompts-per-group", str(exp.per_group),
        "--num-prompts", str(exp.num_prompts),
        "--request-rate", str(exp.rate),
        "--port", str(port),
        "--seed", "1",
        "--flush-cache",
        "--output-file", output_file,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr[-300:]}")

    with open(output_file) as f:
        # Handle multi-line JSONL — take last line
        lines = f.read().strip().split("\n")
        data = json.loads(lines[-1])

    return {
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "median_ttft_ms": data.get("median_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "output_throughput": data.get("output_throughput"),
        "completed": data.get("completed"),
    }


def compute_stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {}
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
        se = std / math.sqrt(n)
        # t-value for 95% CI with n-1 df (approximate for small n)
        t_vals = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        t = t_vals.get(n, 2.0)
        ci_lo = mean - t * se
        ci_hi = mean + t * se
    else:
        std = 0
        ci_lo = ci_hi = mean
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n": n,
        "values": values,
    }


def main():
    args = parse_args()
    exp = EXPERIMENTS[args.experiment]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent / f"variance_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Variance Test: experiment={exp.id} ({exp.name}), rounds={args.rounds}")
    print(f"Results dir: {results_dir}")
    print(f"Model: {args.model}, TP={args.tp}, Port={args.port}")
    print(f"Pattern: alternating LFC → STD per round\n")

    all_results = {"lfc": [], "std": []}

    for round_idx in range(1, args.rounds + 1):
        for mode in ["lfc", "std"]:
            is_lfc = mode == "lfc"
            label = "LFC" if is_lfc else "STD"
            run_id = f"r{round_idx}_{mode}"

            print(f"{'='*60}")
            print(f"Round {round_idx}/{args.rounds} — {label}")
            print(f"{'='*60}")

            log_path = str(results_dir / f"server_{run_id}.log")
            output_file = str(results_dir / f"result_{run_id}.jsonl")

            proc = None
            try:
                proc = start_server(args, lfc=is_lfc, log_path=log_path)
                print(f"  Server PID: {proc.pid}, ready")

                metrics = run_benchmark(exp, args.model, args.port, output_file)
                metrics["round"] = round_idx
                metrics["mode"] = mode
                all_results[mode].append(metrics)

                print(f"  TTFT: {metrics['mean_ttft_ms']:.1f}ms (mean), "
                      f"{metrics['median_ttft_ms']:.1f}ms (med), "
                      f"TP: {metrics['output_throughput']:.1f} t/s")

            except Exception as e:
                print(f"  FAILED: {e}")
                all_results[mode].append({"round": round_idx, "mode": mode, "error": str(e)})

            finally:
                kill_server(proc)
                time.sleep(5)

    # Save raw results
    raw_path = results_dir / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Statistical analysis
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS — Experiment {exp.id} ({exp.name})")
    print(f"{'='*80}\n")

    metrics_to_compare = ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                          "mean_tpot_ms", "output_throughput"]

    for metric in metrics_to_compare:
        lfc_vals = [r[metric] for r in all_results["lfc"] if metric in r and r[metric] is not None]
        std_vals = [r[metric] for r in all_results["std"] if metric in r and r[metric] is not None]

        if not lfc_vals or not std_vals:
            continue

        lfc_stats = compute_stats(lfc_vals)
        std_stats = compute_stats(std_vals)

        delta_pct = (lfc_stats["mean"] - std_stats["mean"]) / std_stats["mean"] * 100

        print(f"--- {metric} ---")
        print(f"  STD: {std_stats['mean']:8.1f} ± {std_stats['std']:6.1f}  "
              f"[{std_stats['ci95_lo']:.1f}, {std_stats['ci95_hi']:.1f}] 95% CI  "
              f"(range: {std_stats['min']:.1f} – {std_stats['max']:.1f})")
        print(f"  LFC: {lfc_stats['mean']:8.1f} ± {lfc_stats['std']:6.1f}  "
              f"[{lfc_stats['ci95_lo']:.1f}, {lfc_stats['ci95_hi']:.1f}] 95% CI  "
              f"(range: {lfc_stats['min']:.1f} – {lfc_stats['max']:.1f})")
        print(f"  Δ: {delta_pct:+.1f}%")

        # Check if CIs overlap
        overlap = not (lfc_stats["ci95_lo"] > std_stats["ci95_hi"] or
                       std_stats["ci95_lo"] > lfc_stats["ci95_hi"])
        print(f"  CIs overlap: {'YES → no significant difference' if overlap else 'NO → significant difference'}")
        print()

    # Per-round comparison
    print("Per-round results:")
    print(f"{'Round':>6} {'STD TTFT':>10} {'LFC TTFT':>10} {'Δ':>8}")
    print("-" * 40)
    for i in range(args.rounds):
        std_r = all_results["std"][i] if i < len(all_results["std"]) else {}
        lfc_r = all_results["lfc"][i] if i < len(all_results["lfc"]) else {}
        std_ttft = std_r.get("mean_ttft_ms", float("nan"))
        lfc_ttft = lfc_r.get("mean_ttft_ms", float("nan"))
        if std_ttft and lfc_ttft and std_ttft == std_ttft and lfc_ttft == lfc_ttft:
            delta = (lfc_ttft - std_ttft) / std_ttft * 100
            print(f"{i+1:>6} {std_ttft:>9.1f}ms {lfc_ttft:>9.1f}ms {delta:>+7.1f}%")
        else:
            print(f"{i+1:>6} {'ERR':>10} {'ERR':>10}")

    print(f"\nRaw results saved to: {raw_path}")
    print("Done!")


if __name__ == "__main__":
    main()
