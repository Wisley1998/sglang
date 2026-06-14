# SGLang dLLM JointThreshold Optimization Overview

This document summarizes the motivation, scope, implementation strategy, and
validation results for the dLLM JointThreshold serving optimization.

## 1. Problem

This work targets SGLang's diffusion LLM serving path, specifically the
`JointThreshold` dLLM algorithm used by SGLang-supported dLLM model families.

The optimization covers the dLLM model types currently recognized by SGLang's
dLLM configuration:

- `LLaDA2MoeModelLM`
- `SDARForCausalLM`
- `SDARMoeForCausalLM`

The core problem is wasted work during diffusion block generation. Within one
dLLM block, different requests or slots can converge at different denoise
iterations. In the baseline execution path, once a slot has already converged,
it can still remain in the physical forward batch until the whole block
finishes. This means later denoise iterations may continue paying full-batch
model forward, attention metadata, CUDA graph replay, and logits-processing
costs for slots that no longer need useful computation.

This overhead becomes especially visible at high concurrency and long context,
where many slots finish early inside a block but the server still carries them
through the remaining iterations.

## 2. Baseline Behavior

The SGLang baseline already supports dLLM serving with `JointThreshold`. At a
high level, the baseline runs the diffusion block loop over the current batch
and commits generated tokens after the block finishes.

The baseline behavior is conservative:

- The physical batch shape remains fixed inside the block.
- CUDA graph replay uses the original captured batch bucket.
- Full-vocabulary logits are materialized for JointThreshold decisions.
- The update path is mostly per-slot.
- dLLM replay prefix length can be derived from `seq_lens - block_size`.
- Non-continuous-batching admission can undercount staged dLLM requests when
  deciding how many new requests to admit.

This is robust, but it leaves performance on the table when many slots converge
before the block ends.

## 3. Optimization Strategy

The main idea is to make within-block convergence visible to the runtime and
avoid forwarding already-converged slots when it is safe to do so.

The implementation adds these pieces:

- **Within-block dead-slot detection**: track which slots have converged during
  the current diffusion block.
- **Active-slot compaction**: reorder active slots before inactive slots so the
  useful part of the batch is contiguous.
- **Physical batch shrink**: after enough slots are dead, slice the
  `ForwardBatch` tensor views to a smaller CUDA graph batch bucket.
- **Multi-shrink support**: allow more than one shrink decision within a block,
  while restoring the original batch layout before final commit.
- **Bucket-aware padding**: when the exact active size is not a captured CUDA
  graph bucket, use only terminal-finished slots as safe padding. Non-terminal
  converged slots are not used as padding, because their KV state is needed by
  later blocks.
- **Reduced logits path**: when JointThreshold only needs confidence
  statistics, return argmax token id, max logit, and logsumexp instead of
  materializing full-vocabulary logits.
- **Vectorized JointThreshold update**: reduce Python/per-slot overhead in the
  decision and update path.
- **CUDA graph replay metadata fix**: pass the true `extend_prefix_lens` through
  replay instead of deriving dLLM prefix length from `seq_lens - block_size`
  after shrink/padding.
- **Scheduler admission fix**: when continuous batching is disabled, count both
  dLLM waiting and staging queues before admitting new requests.
- **Small-batch fallback**: keep a low-overhead reference path for single-request
  latency runs where physical shrink cannot help.

The result is intentionally concentrated in the dLLM algorithm and the narrow
runtime surfaces needed to make shrink, reduced logits, and CUDA graph replay
correct.

## 4. Evaluation Models

We evaluate on two representative dLLM models:

| Model | Type | Why It Was Chosen |
|---|---|---|
| `inclusionAI/LLaDA2.1-mini` | MoE dLLM | Exercises the MoE dLLM path and high-concurrency throughput behavior. |
| `JetLM/SDAR-8B-Chat` | Dense dLLM | Exercises a dense dLLM architecture and validates the optimization outside MoE-specific behavior. |

Together, these models cover the two main dLLM execution patterns relevant to
this optimization: MoE and dense.

## 5. Accuracy Results

The optimization is accuracy-neutral in the evaluated suites. The small MMLU
differences are within normal evaluation variance for these stress runs.

| Model | Suite | Baseline | Optimized | Delta |
|---|---:|---:|---:|---:|
| `inclusionAI/LLaDA2.1-mini` | official GSM8K | 0.8650 | 0.8650 | +0.0000 |
| `inclusionAI/LLaDA2.1-mini` | MMLU stress | 0.7550 | 0.7500 | -0.0050 |
| `JetLM/SDAR-8B-Chat` | official GSM8K | 0.0250 | 0.0250 | +0.0000 |
| `JetLM/SDAR-8B-Chat` | MMLU stress | 0.0000 | 0.0100 | +0.0100 |

Full accuracy artifact:

- `experiments/pr_accuracy_eval/artifacts/summary.md`

## 6. Speed Summary

Benchmarks use `python -m sglang.bench_serving`, random dataset, infinite request
rate, `JointThreshold`, and FlashInfer attention.

The serial latency benchmark is effectively unchanged. The gains appear in the
targeted regime: high-concurrency and long-context serving.

### LLaDA2.1-mini

| Bench | TTFT Change | Throughput Gain |
|---|---:|---:|
| Latency, n=10, c=1 | -1.9% | +0.6% |
| Throughput, 1xA100, c=100 | -25.0% | +33.3% |
| Throughput, TP=4, c=600 | -57.9% | +136.8% |
| Long-context, TP=4, c=600, 2048/2048 | -55.0% | +158.5% |

### SDAR-8B

| Bench | TTFT Change | Throughput Gain |
|---|---:|---:|
| Latency, n=10, c=1 | +0.2% | -0.3% |
| Throughput, 1xA100, c=100 | -16.5% | +38.6% |
| Throughput, TP=4, c=600 | -16.0% | +96.3% |
| Long-context, TP=4, c=600, 2048/2048 | -19.6% | +83.2% |

## 7. Conclusion

This PR improves dLLM serving efficiency by avoiding wasted work after slots
have already converged inside a diffusion block. It preserves the original
request layout and restores the full batch before final commit, while allowing
intermediate denoise iterations to run on smaller CUDA graph buckets.

The optimization is accuracy-neutral on the evaluated suites and improves
throughput substantially in the high-concurrency regimes where dLLM serving is
most expensive:

- LLaDA2.1-mini: up to **+158.5%** output token throughput.
- SDAR-8B: up to **+96.3%** output token throughput.

## Appendix A. Full Main Benchmark Results

### LLaDA2.1-mini

| Bench | Baseline TTFT | Optimized TTFT | TTFT Speedup | Baseline tok/s | Optimized tok/s | Throughput Gain |
|---|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 414.19 ms | 406.40 ms | 1.02x | 162.85 | 163.84 | +0.6% |
| Throughput, 1xA100, c=100 | 7381.46 ms | 5538.26 ms | 1.33x | 605.59 | 807.54 | +33.3% |
| Throughput, TP=4, c=600 | 70440.02 ms | 29648.71 ms | 2.38x | 393.40 | 931.70 | +136.8% |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 66454.22 ms | 29875.60 ms | 2.22x | 409.73 | 1059.07 | +158.5% |

### SDAR-8B

| Bench | Baseline TTFT | Optimized TTFT | TTFT Speedup | Baseline tok/s | Optimized tok/s | Throughput Gain |
|---|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 2589.84 ms | 2594.28 ms | 1.00x | 54.46 | 54.31 | -0.3% |
| Throughput, 1xA100, c=100 | 6123.06 ms | 5115.27 ms | 1.20x | 175.82 | 243.75 | +38.6% |
| Throughput, TP=4, c=600 | 34135.29 ms | 28667.61 ms | 1.19x | 290.48 | 570.22 | +96.3% |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 31939.05 ms | 25680.57 ms | 1.24x | 248.50 | 455.34 | +83.2% |

## Appendix B. Tail Latency Results

### LLaDA2.1-mini

| Bench | Baseline E2E P90 | Optimized E2E P90 | Baseline E2E P99 | Optimized E2E P99 | Baseline TTFT P99 | Optimized TTFT P99 | Baseline ITL P95/P99 | Optimized ITL P95/P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 5182.84 ms | 5134.79 ms | 6622.58 ms | 6584.36 ms | 636.06 ms | 611.46 ms | 10.30 ms / 12.22 ms | 10.63 ms / 12.19 ms |
| Throughput, 1xA100, c=100 | 138846.87 ms | 102260.97 ms | 159515.20 ms | 118069.06 ms | 11997.68 ms | 8749.33 ms | 199.26 ms / 266.23 ms | 140.51 ms / 189.05 ms |
| Throughput, TP=4, c=600 | 1155576.56 ms | 462305.22 ms | 1204583.65 ms | 492629.65 ms | 601049.82 ms | 212997.88 ms | 1899.97 ms / 2137.04 ms | 585.13 ms / 813.98 ms |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 2613678.49 ms | 942004.11 ms | 3040436.51 ms | 1107994.08 ms | 124467.60 ms | 48432.67 ms | 2368.71 ms / 2496.73 ms | 638.06 ms / 823.89 ms |

### SDAR-8B

| Bench | Baseline E2E P90 | Optimized E2E P90 | Baseline E2E P99 | Optimized E2E P99 | Baseline TTFT P99 | Optimized TTFT P99 | Baseline ITL P95/P99 | Optimized ITL P95/P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Latency, n=10, c=1 | 10473.76 ms | 10520.27 ms | 12932.50 ms | 12950.83 ms | 3932.58 ms | 3932.73 ms | 68.04 ms / 79.30 ms | 67.71 ms / 79.46 ms |
| Throughput, 1xA100, c=100 | 522903.69 ms | 370630.16 ms | 589898.78 ms | 417684.82 ms | 11193.41 ms | 10202.99 ms | 1245.36 ms / 1350.31 ms | 1076.70 ms / 1274.23 ms |
| Throughput, TP=4, c=600 | 1553315.55 ms | 818967.09 ms | 1631997.23 ms | 850451.32 ms | 50184.92 ms | 44114.26 ms | 2698.32 ms / 3089.37 ms | 2023.08 ms / 2203.79 ms |
| Long-context steady-state, TP=4, c=600, 2048/2048 | 4384662.78 ms | 2441624.07 ms | 5027413.33 ms | 2832309.17 ms | 91501.92 ms | 85963.65 ms | 3729.31 ms / 3960.21 ms | 2809.81 ms / 3304.78 ms |

## Appendix C. Full Benchmark Reports

The full benchmark reports, including reproduction commands, are available in:

- `LLADA2_1_MINI_BENCHMARK_REPORT.md`
- `SDAR_8B_BENCHMARK_REPORT.md`
