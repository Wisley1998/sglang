# SDAR-8B Final Benchmark Report

Date: 2026-05-28

This report contains the final benchmark conclusion and reproduction protocol for `JetLM/SDAR-8B-Chat`, using the same three official `sglang.bench_serving` benchmark shapes as `LLADA2_1_MINI_BENCHMARK_REPORT.md`.

## Final Results

Metrics:

- TTFT speedup = `baseline mean TTFT / ours mean TTFT`.
- Throughput gain = `ours output token throughput / baseline output token throughput - 1`.
- Throughput is `Output token throughput` reported by `sglang.bench_serving`.

| Bench | Baseline TTFT | Ours TTFT | TTFT Speedup | Baseline tok/s | Ours tok/s | Throughput Gain |
|---|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 2589.84 ms | 2594.28 ms | **1.00x**, TTFT +0.2% | 54.46 | 54.31 | **-0.3%** |
| Throughput, 1xA100, c=100 | 6123.06 ms | 5115.27 ms | **1.20x**, TTFT -16.5% | 175.82 | 243.75 | **+38.6%** |
| Throughput, TP=4, c=600 | 34135.29 ms | 28667.61 ms | **1.19x**, TTFT -16.0% | 290.48 | 570.22 | **+96.3%** |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 31939.05 ms | 25680.57 ms | **1.24x**, TTFT -19.6% | 248.50 | 455.34 | **+83.2%** |

Available tail latency metrics from the existing `sglang.bench_serving` artifacts:

- E2E latency P95 is not reported in the saved summary artifacts, so it is intentionally omitted.
- E2E P90/P99, TTFT P99, and ITL P95/P99 are reported directly by `sglang.bench_serving`.

| Bench | Baseline E2E P90 | Ours E2E P90 | Baseline E2E P99 | Ours E2E P99 | Baseline TTFT P99 | Ours TTFT P99 | Baseline ITL P95/P99 | Ours ITL P95/P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 10473.76 ms | 10520.27 ms | 12932.50 ms | 12950.83 ms | 3932.58 ms | 3932.73 ms | 68.04 ms / 79.30 ms | 67.71 ms / 79.46 ms |
| Throughput, 1xA100, c=100 | 522903.69 ms | 370630.16 ms | 589898.78 ms | 417684.82 ms | 11193.41 ms | 10202.99 ms | 1245.36 ms / 1350.31 ms | 1076.70 ms / 1274.23 ms |
| Throughput, TP=4, c=600 | 1553315.55 ms | 818967.09 ms | 1631997.23 ms | 850451.32 ms | 50184.92 ms | 44114.26 ms | 2698.32 ms / 3089.37 ms | 2023.08 ms / 2203.79 ms |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 4384662.78 ms | 2441624.07 ms | 5027413.33 ms | 2832309.17 ms | 91501.92 ms | 85963.65 ms | 3729.31 ms / 3960.21 ms | 2809.81 ms / 3304.78 ms |

Final conclusion: compared with the SGLang baseline, the latest optimized stack is effectively tied on the serial latency benchmark and substantially faster on the throughput benchmarks. The TP=4 high-concurrency run shows the largest SDAR-8B gain: output token throughput improves by **+96.3%** while mean TTFT improves by **16.0%**. The long-context steady-state benchmark also improves output token throughput by **+83.2%** while mean TTFT improves by **19.6%**.

## Final Benchmark Protocol

Common settings:

- Model: `JetLM/SDAR-8B-Chat`
- Benchmark client: `python -m sglang.bench_serving`
- Dataset: `random`
- Requested random input/output length: `1000/1000`
- Request rate: `inf`
- Seed: `1`
- Warmup requests: `1`
- dLLM algorithm: `JointThreshold`
- Attention backend: `flashinfer`
- Baseline environment: `.venv-sglang-clean`
- Ours environment: `.venv-sglang-step2`
- Ours config: `experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json`
- Raw artifacts: `experiments/pr_benchmark_runs/sdar_8b_official_20260528/`

The final optimized config enables Step 3 physical shrink/dead-slot skip, multi-shrink, bucket padding, sync coalescing, reorder fusion, P0 reduced logits, P1 vectorized step, the scheduler TTFT admission fix, and the small-batch fast path.

### 1. Latency Benchmark, n=10, c=1

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-clean/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --tp 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --max-running-requests 1 \
  --attention-backend flashinfer \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30200
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --dllm-algorithm-config experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json \
  --tp 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --max-running-requests 1 \
  --attention-backend flashinfer \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30200
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30200 \
  --model JetLM/SDAR-8B-Chat \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf \
  --seed 1 \
  --warmup-requests 1 \
  --output-details \
  --disable-tqdm
```

### 2. Throughput Benchmark, 1xA100, c=100

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-clean/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --tp 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --max-running-requests 100 \
  --attention-backend flashinfer \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30200
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --dllm-algorithm-config experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json \
  --tp 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --max-running-requests 100 \
  --attention-backend flashinfer \
  --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 96 100 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30200
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30200 \
  --model JetLM/SDAR-8B-Chat \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 500 \
  --max-concurrency 100 \
  --request-rate inf \
  --seed 1 \
  --warmup-requests 1 \
  --output-details \
  --disable-tqdm
```

### 3. Throughput Benchmark, TP=4, c=600

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \
.venv-sglang-clean/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.7 \
  --max-running-requests 600 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 256 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30200
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \
.venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --dllm-algorithm-config experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.7 \
  --max-running-requests 600 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 512 \
  --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 96 112 128 160 192 224 256 320 384 448 512 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30200
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30200 \
  --model JetLM/SDAR-8B-Chat \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 1000 \
  --max-concurrency 600 \
  --request-rate inf \
  --seed 1 \
  --warmup-requests 1 \
  --output-details \
  --disable-tqdm
```

### 4. Long-Context Steady-State Benchmark, TP=4, c=600, 2048/2048

This run follows the official `bench_serving` steady-state recommendation by setting `num-prompts = 3000 = 5 * max-concurrency`.

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \
.venv-sglang-clean/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.7 \
  --max-running-requests 600 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 256 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30300
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \
.venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat \
  --dllm-algorithm JointThreshold \
  --dllm-algorithm-config experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.7 \
  --max-running-requests 600 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 512 \
  --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 96 112 128 160 192 224 256 320 384 448 512 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30300
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30300 \
  --model JetLM/SDAR-8B-Chat \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 2048 \
  --num-prompts 3000 \
  --max-concurrency 600 \
  --request-rate inf \
  --seed 1 \
  --warmup-requests 1 \
  --output-details \
  --disable-tqdm
```

## Raw Files

- Latency artifacts: `experiments/pr_benchmark_runs/sdar_8b_official_20260528/latency/`
- Single-A100 throughput artifacts: `experiments/pr_benchmark_runs/sdar_8b_official_20260528/throughput_1xa100/`
- TP=4 throughput artifacts: `experiments/pr_benchmark_runs/sdar_8b_official_20260528/throughput_tp4/`
- Long-context baseline artifacts: `experiments/pr_benchmark_runs/long_context_tp4_c600_2048_np3000/sdar_8b/baseline/`
- Long-context ours artifacts: `experiments/pr_benchmark_runs/long_context_tp4_c600_2048_np3000/sdar_8b/best/`
- Runner: `experiments/pr_benchmark_runs/sdar_8b_official_20260528/run_sdar_official_bench.py`
- Long-context runner: `experiments/pr_benchmark_runs/long_context_tp4_c600_2048/run_long_context_bench.py`
