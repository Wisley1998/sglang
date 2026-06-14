# LLaDA2.1-mini Final Benchmark Report

Date: 2026-05-28

This report contains only the final benchmark conclusion and the final reproduction protocol for `inclusionAI/LLaDA2.1-mini`. All exploratory, failed, historical, and incorrectly configured runs are excluded from the conclusion.

## Final Results

Metrics:

- TTFT speedup = `baseline mean TTFT / ours mean TTFT`.
- Throughput gain = `ours output token throughput / baseline output token throughput - 1`.
- Throughput is `Output token throughput` reported by `sglang.bench_serving`.

| Bench | Baseline TTFT | Ours TTFT | TTFT Speedup | Baseline tok/s | Ours tok/s | Throughput Gain |
|---|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 414.19 ms | 406.40 ms | **1.02x**, TTFT -1.9% | 162.85 | 163.84 | **+0.6%** |
| Throughput, 1xA100, c=100 | 7381.46 ms | 5538.26 ms | **1.33x**, TTFT -25.0% | 605.59 | 807.54 | **+33.3%** |
| Throughput, TP=4, c=600 | 70440.02 ms | 29648.71 ms | **2.38x**, TTFT -57.9% | 393.40 | 931.70 | **+136.8%** |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 66454.22 ms | 29875.60 ms | **2.22x**, TTFT -55.0% | 409.73 | 1059.07 | **+158.5%** |

Available tail latency metrics from the existing `sglang.bench_serving` artifacts:

- E2E latency P95 is not reported in the saved summary artifacts, so it is intentionally omitted.
- E2E P90/P99, TTFT P99, and ITL P95/P99 are reported directly by `sglang.bench_serving`.

| Bench | Baseline E2E P90 | Ours E2E P90 | Baseline E2E P99 | Ours E2E P99 | Baseline TTFT P99 | Ours TTFT P99 | Baseline ITL P95/P99 | Ours ITL P95/P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 5182.84 ms | 5134.79 ms | 6622.58 ms | 6584.36 ms | 636.06 ms | 611.46 ms | 10.30 ms / 12.22 ms | 10.63 ms / 12.19 ms |
| Throughput, 1xA100, c=100 | 138846.87 ms | 102260.97 ms | 159515.20 ms | 118069.06 ms | 11997.68 ms | 8749.33 ms | 199.26 ms / 266.23 ms | 140.51 ms / 189.05 ms |
| Throughput, TP=4, c=600 | 1155576.56 ms | 462305.22 ms | 1204583.65 ms | 492629.65 ms | 601049.82 ms | 212997.88 ms | 1899.97 ms / 2137.04 ms | 585.13 ms / 813.98 ms |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 2613678.49 ms | 942004.11 ms | 3040436.51 ms | 1107994.08 ms | 124467.60 ms | 48432.67 ms | 2368.71 ms / 2496.73 ms | 638.06 ms / 823.89 ms |

Final conclusion: compared with the SGLang baseline, the latest optimized stack is effectively tied on the latency benchmark and substantially faster on the throughput benchmarks. The long-context steady-state benchmark is the strongest LLaDA2.1-mini result: output token throughput improves by **+158.5%** while mean TTFT improves by **55.0%**.

## Final Benchmark Protocol

Common settings:

- Model: `inclusionAI/LLaDA2.1-mini`
- Benchmark client: `python -m sglang.bench_serving`
- Dataset: `random`
- Random input/output length: `1000/1000`
- Request rate: `inf`
- Seed: `1`
- Warmup requests: `1`
- dLLM algorithm: `JointThreshold`
- Attention backend: `flashinfer`
- Baseline environment: `.venv-sglang-clean`
- Ours environment: `.venv-sglang-step2`
- Ours config: `experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json`

The final optimized config enables Step 3 physical shrink/dead-slot skip, multi-shrink, bucket padding, sync coalescing, reorder fusion, P0 reduced logits, P1 vectorized step, the scheduler TTFT admission fix, and the small-batch fast path.

### 1. Latency Benchmark, n=10, c=1

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-clean/bin/python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.1-mini \
  --dllm-algorithm JointThreshold \
  --tp 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --max-running-requests 1 \
  --attention-backend flashinfer \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30000
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.1-mini \
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
  --port 30000
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model inclusionAI/LLaDA2.1-mini \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf \
  --seed 1 \
  --output-details \
  --disable-tqdm
```

### 2. Throughput Benchmark, 1xA100, c=100

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-clean/bin/python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.1-mini \
  --dllm-algorithm JointThreshold \
  --tp 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --max-running-requests 100 \
  --attention-backend flashinfer \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30000
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.1-mini \
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
  --port 30000
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model inclusionAI/LLaDA2.1-mini \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 500 \
  --max-concurrency 100 \
  --request-rate inf \
  --seed 1 \
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
  --model-path inclusionAI/LLaDA2.1-mini \
  --dllm-algorithm JointThreshold \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.25 \
  --max-running-requests 600 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 256 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30000
```

Ours server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \
.venv-sglang-step2/bin/python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.1-mini \
  --dllm-algorithm JointThreshold \
  --dllm-algorithm-config experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.25 \
  --max-running-requests 600 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 512 \
  --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 96 112 128 160 192 224 256 320 384 448 512 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30000
```

Client:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model inclusionAI/LLaDA2.1-mini \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 1000 \
  --max-concurrency 600 \
  --request-rate inf \
  --seed 1 \
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
  --model-path inclusionAI/LLaDA2.1-mini \
  --dllm-algorithm JointThreshold \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.25 \
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
  --model-path inclusionAI/LLaDA2.1-mini \
  --dllm-algorithm JointThreshold \
  --dllm-algorithm-config experiments/live_verification/diagnose_20260522/iter_level_experiments/best_p01_v2ms_algoconfig.json \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.25 \
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
  --model inclusionAI/LLaDA2.1-mini \
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

Raw artifacts:

- Baseline: `experiments/pr_benchmark_runs/long_context_tp4_c600_2048_np3000/llada2_1_mini/baseline/`
- Ours: `experiments/pr_benchmark_runs/long_context_tp4_c600_2048_np3000/llada2_1_mini/best/`
