#!/usr/bin/env python3
"""
LFC vs Standard Benchmarking Automation

Runs 6 experiments × 2 modes (LFC, Standard) using sglang.bench_serving,
plus optional torch profiler traces for the D_baseline experiment.

Usage:
    python benchmark/lfc/run_benchmarks.py [OPTIONS]

Options:
    --model          Model path (default: Qwen/Qwen3-Next-80B-A3B-Instruct)
    --tp             Tensor parallelism (default: 4)
    --port           Server port (default: 30000)
    --mem-fraction   Static memory fraction (default: 0.60)
    --results-dir    Output directory for JSONL results (default: benchmark/lfc/results)
    --profiles-dir   Output directory for profiler traces (default: benchmark/lfc/profiles)
    --experiments    Comma-separated experiment IDs to run (default: A,B,C,D,E,F)
    --modes          Comma-separated modes to run (default: lfc,std)
    --profile        Enable torch profiler for D_baseline experiment
    --skip-server    Don't start/stop servers (assume already running)
    --dry-run        Print commands without executing
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
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


EXPERIMENTS = [
    Experiment("A", "high_qps", 4096, 128, 8, 16, 8, 128),
    Experiment("B", "many_groups", 4096, 128, 16, 8, 4, 128),
    Experiment("C", "long_prefix", 8192, 128, 8, 16, 4, 128),
    Experiment("D", "baseline", 4096, 128, 8, 16, 4, 128),
    Experiment("E", "large_batch", 4096, 256, 8, 32, 4, 128),
    Experiment("F", "very_high_qps", 4096, 128, 8, 16, 16, 128),
]

EXPERIMENT_MAP = {e.id: e for e in EXPERIMENTS}


def parse_args():
    parser = argparse.ArgumentParser(description="LFC vs Standard Benchmarking")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="Model path",
    )
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallelism")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument(
        "--mem-fraction", type=float, default=0.60, help="Static memory fraction"
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark/lfc/results",
        help="Output directory for JSONL results",
    )
    parser.add_argument(
        "--profiles-dir",
        default="benchmark/lfc/profiles",
        help="Output directory for profiler traces",
    )
    parser.add_argument(
        "--experiments",
        default="A,B,C,D,E,F",
        help="Comma-separated experiment IDs",
    )
    parser.add_argument(
        "--modes",
        default="lfc,std",
        help="Comma-separated modes to run (lfc, std)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler for D_baseline",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Don't start/stop servers (assume already running)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=900,
        help="Server startup timeout in seconds (default: 900)",
    )
    return parser.parse_args()


def wait_for_server(port: int, timeout: int = 600, interval: int = 5) -> bool:
    """Poll server health endpoint until ready or timeout."""
    import urllib.request

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"  Server ready on port {port}")
                    return True
        except Exception:
            pass
        time.sleep(interval)
    print(f"  ERROR: Server on port {port} did not become healthy within {timeout}s")
    return False


def start_server(
    model: str,
    tp: int,
    port: int,
    mem_fraction: float,
    lfc: bool,
    profile_dir: str | None = None,
    dry_run: bool = False,
    log_dir: str | None = None,
    log_suffix: str = "",
    timeout: int = 900,
) -> subprocess.Popen | None:
    """Start an SGLang server process."""
    env = os.environ.copy()
    if lfc:
        env["SGLANG_LFC_ENABLED"] = "1"
    if profile_dir:
        env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--tp",
        str(tp),
        "--port",
        str(port),
        "--mem-fraction-static",
        str(mem_fraction),
    ]

    mode_label = "LFC" if lfc else "Standard"
    print(f"\n{'='*60}")
    print(f"Starting {mode_label} server")
    print(f"  Command: {' '.join(cmd)}")
    if lfc:
        print("  Env: SGLANG_LFC_ENABLED=1")
    if profile_dir:
        print(f"  Env: SGLANG_TORCH_PROFILER_DIR={profile_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Skipping server start")
        return None

    # Log server output to file for diagnosis
    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_name = f"server_{log_suffix}.log" if log_suffix else f"server_{mode_label.lower()}.log"
        log_path = os.path.join(log_dir, log_name)
        log_file = open(log_path, "w")
        print(f"  Server log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"  Server PID: {proc.pid}")

    if not wait_for_server(port, timeout=timeout):
        # Print tail of server log for diagnosis
        if log_file:
            log_file.flush()
            log_path = log_file.name
            log_file.close()
            log_file = None
            print(f"  Last 30 lines of server log ({log_path}):")
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                for line in lines[-30:]:
                    print(f"    {line.rstrip()}")
            except Exception:
                pass
        kill_server(proc)
        raise RuntimeError(f"{mode_label} server failed to start within {timeout}s")

    # Store log_file handle on proc so we can close it later
    proc._log_file = log_file  # type: ignore[attr-defined]
    return proc


def kill_server(proc: subprocess.Popen | None):
    """Kill server process and its process group."""
    if proc is None:
        return
    # Close log file if attached
    log_file = getattr(proc, "_log_file", None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass
    print(f"  Killing server (PID {proc.pid})...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception:
            pass
    print("  Server stopped")


def run_benchmark(
    exp: Experiment,
    mode: str,
    model: str,
    port: int,
    results_dir: str,
    profile: bool = False,
    dry_run: bool = False,
) -> str:
    """Run a single benchmark experiment. Returns the output file path."""
    output_file = os.path.join(results_dir, f"{mode}_{exp.id}.jsonl")

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--model",
        model,
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-system-prompt-len",
        str(exp.prefix_len),
        "--gsp-question-len",
        str(exp.question_len),
        "--gsp-output-len",
        "256",
        "--gsp-num-groups",
        str(exp.groups),
        "--gsp-prompts-per-group",
        str(exp.per_group),
        "--num-prompts",
        str(exp.num_prompts),
        "--request-rate",
        str(exp.rate),
        "--port",
        str(port),
        "--seed",
        "1",
        "--flush-cache",
        "--output-file",
        output_file,
        "--output-details",
    ]
    if profile:
        cmd.append("--profile")

    mode_label = "LFC" if mode == "lfc" else "Standard"
    print(f"\n--- Experiment {exp.id} ({exp.name}) [{mode_label}] ---")
    print(f"  prefix={exp.prefix_len}, prompts={exp.num_prompts}, "
          f"groups={exp.groups}, per_group={exp.per_group}, "
          f"rate={exp.rate}, question_len={exp.question_len}")
    print(f"  Output: {output_file}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return output_file

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout per experiment
    )

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        raise RuntimeError(
            f"Benchmark {exp.id} ({mode}) failed: {result.stderr[-200:]}"
        )

    # Print summary from output
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                data = json.loads(f.read())
            print(f"  Mean TTFT: {data.get('mean_ttft_ms', 'N/A'):.1f} ms")
            print(f"  Median TTFT: {data.get('median_ttft_ms', 'N/A'):.1f} ms")
            print(f"  P99 TTFT: {data.get('p99_ttft_ms', 'N/A'):.1f} ms")
            print(f"  Output throughput: {data.get('output_throughput', 'N/A'):.1f} tok/s")
            print(f"  Completed: {data.get('completed', 'N/A')}")
        except Exception as e:
            print(f"  Warning: could not parse results: {e}")
    else:
        print(f"  Warning: output file not found at {output_file}")

    return output_file


def _server_alive(port: int) -> bool:
    """Quick check if server is still responding."""
    import urllib.request

    try:
        req = urllib.request.Request(f"http://localhost:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_single_experiment(
    exp: Experiment,
    mode: str,
    args: argparse.Namespace,
) -> str | None:
    """Run a single experiment with a fresh server.

    Each experiment gets its own server instance to ensure fair comparison:
    start server -> run benchmark -> kill server.
    """
    is_lfc = mode == "lfc"
    mode_label = "LFC" if is_lfc else "Standard"
    do_profile = args.profile and exp.id == "D"

    # Set up profiling directory if needed
    profile_dir = None
    if do_profile:
        profile_dir = os.path.join(
            args.profiles_dir, f"{'lfc' if is_lfc else 'std'}_profile"
        )
        os.makedirs(profile_dir, exist_ok=True)

    # Per-experiment server log
    log_suffix = f"{mode}_{exp.id}"

    proc = None
    try:
        if not args.skip_server:
            proc = start_server(
                model=args.model,
                tp=args.tp,
                port=args.port,
                mem_fraction=args.mem_fraction,
                lfc=is_lfc,
                profile_dir=profile_dir,
                dry_run=args.dry_run,
                log_dir=args.results_dir,
                log_suffix=log_suffix,
                timeout=args.server_timeout,
            )

        output_file = run_benchmark(
            exp=exp,
            mode=mode,
            model=args.model,
            port=args.port,
            results_dir=args.results_dir,
            profile=do_profile,
            dry_run=args.dry_run,
        )
        return output_file

    except RuntimeError as e:
        print(f"  ERROR: {e}")
        print(f"  Skipping experiment {exp.id} ({exp.name}) [{mode_label}]")
        return None

    finally:
        if not args.skip_server:
            kill_server(proc)
            # Brief pause to let ports and GPU memory fully release
            if not args.dry_run:
                time.sleep(5)


def print_summary(results_dir: str, experiment_ids: list[str]):
    """Print a comparison table from collected results."""
    print(f"\n{'='*80}")
    print("SUMMARY: LFC vs Standard")
    print(f"{'='*80}")
    header = (
        f"{'Exp':>4} {'Name':<16} {'Mode':<8} "
        f"{'Mean TTFT':>10} {'Med TTFT':>10} {'P99 TTFT':>10} "
        f"{'Throughput':>12} {'Completed':>10}"
    )
    print(header)
    print("-" * len(header))

    for exp_id in experiment_ids:
        exp = EXPERIMENT_MAP[exp_id]
        for mode in ["std", "lfc"]:
            path = os.path.join(results_dir, f"{mode}_{exp_id}.jsonl")
            if not os.path.exists(path):
                print(f"  {exp_id:>4} {exp.name:<16} {mode:<8} -- results not found --")
                continue
            try:
                with open(path) as f:
                    data = json.loads(f.read())
                mean_ttft = data.get("mean_ttft_ms", float("nan"))
                med_ttft = data.get("median_ttft_ms", float("nan"))
                p99_ttft = data.get("p99_ttft_ms", float("nan"))
                throughput = data.get("output_throughput", float("nan"))
                completed = data.get("completed", "?")
                print(
                    f"  {exp_id:>4} {exp.name:<16} {mode:<8} "
                    f"{mean_ttft:>9.1f}ms {med_ttft:>9.1f}ms {p99_ttft:>9.1f}ms "
                    f"{throughput:>10.1f} t/s {completed:>10}"
                )
            except Exception as e:
                print(f"  {exp_id:>4} {exp.name:<16} {mode:<8} -- error: {e} --")

    # Print deltas
    print(f"\n{'Exp':>4} {'Name':<16} {'TTFT Δ':>12} {'Throughput Δ':>14}")
    print("-" * 50)
    for exp_id in experiment_ids:
        exp = EXPERIMENT_MAP[exp_id]
        std_path = os.path.join(results_dir, f"std_{exp_id}.jsonl")
        lfc_path = os.path.join(results_dir, f"lfc_{exp_id}.jsonl")
        if not (os.path.exists(std_path) and os.path.exists(lfc_path)):
            continue
        try:
            with open(std_path) as f:
                std = json.loads(f.read())
            with open(lfc_path) as f:
                lfc = json.loads(f.read())
            ttft_delta = (
                (lfc["mean_ttft_ms"] - std["mean_ttft_ms"]) / std["mean_ttft_ms"] * 100
            )
            tp_delta = (
                (lfc["output_throughput"] - std["output_throughput"])
                / std["output_throughput"]
                * 100
            )
            print(
                f"  {exp_id:>4} {exp.name:<16} "
                f"{ttft_delta:>+10.1f}% {tp_delta:>+12.1f}%"
            )
        except Exception:
            pass


def main():
    args = parse_args()

    # Resolve relative paths from repo root
    script_dir = Path(__file__).resolve().parent
    if not os.path.isabs(args.results_dir):
        args.results_dir = str(script_dir / Path(args.results_dir).name)
    if not os.path.isabs(args.profiles_dir):
        args.profiles_dir = str(script_dir / Path(args.profiles_dir).name)

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.profiles_dir, exist_ok=True)

    experiment_ids = [x.strip() for x in args.experiments.split(",")]
    modes = [x.strip() for x in args.modes.split(",")]

    # Validate experiment IDs
    for eid in experiment_ids:
        if eid not in EXPERIMENT_MAP:
            print(f"ERROR: Unknown experiment ID '{eid}'. Valid: {list(EXPERIMENT_MAP)}")
            sys.exit(1)

    experiments = [EXPERIMENT_MAP[eid] for eid in experiment_ids]

    print(f"Experiments: {experiment_ids}")
    print(f"Modes: {modes}")
    print(f"Results dir: {args.results_dir}")
    print(f"Model: {args.model}")
    print(f"TP: {args.tp}, Port: {args.port}, Mem fraction: {args.mem_fraction}")
    if args.profile:
        print(f"Profiling enabled for D_baseline")
    if args.dry_run:
        print("*** DRY RUN MODE ***")

    all_output_files = []

    # Run each (mode, experiment) pair with a fresh server
    for mode in modes:
        if mode not in ("lfc", "std"):
            print(f"ERROR: Unknown mode '{mode}'. Valid: lfc, std")
            sys.exit(1)
        for exp in experiments:
            output_file = run_single_experiment(exp, mode, args)
            if output_file:
                all_output_files.append(output_file)

    # Print comparison summary
    print_summary(args.results_dir, experiment_ids)

    print(f"\nAll results saved to: {args.results_dir}")
    if args.profile:
        print(f"Profiler traces saved to: {args.profiles_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
