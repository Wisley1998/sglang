# LFC Performance Analysis: Why LFC Is Still 10% Slower Than Standard

## Background

LFC (Linear Factor Caching) stores intermediate factors (k, v, g, beta) during prefill, enabling SSM state reconstruction at tombstoned radix tree nodes without full recomputation. In the shared-prefix benchmark (8 groups x 16 prompts, prefix=4096, question=128), the results are:

| Metric | Standard | LFC | Delta |
|--------|----------|-----|-------|
| Mean TTFT (ms) | 442.60 | 487.09 | **+10.1%** |
| Median TTFT (ms) | 360.56 | 398.40 | +10.5% |
| Mamba Hit Rate | 32.04% | 40.30% | +8.3pp |
| Attn Hit Rate | 90.82% | 90.82% | 0 |

LFC successfully increases mamba hit rate (+8pp) but the overhead of the LFC pipeline itself **exceeds** the savings from additional mamba reuse. This document identifies the overhead sources ranked by impact.

## Model Architecture: Qwen3-Next-80B-A3B-Instruct

- **Total layers:** 48
- **GDN (linear attention) layers:** 36 (layers where `(l+1) % 4 != 0`)
- **Full attention layers:** 12 (every 4th layer)
- **Factor capture runs on:** all 36 GDN layers
- **TP=4:** each GPU handles H_k=4, H_v=8, K=128, V=128 per GDN layer

---

## Overhead #1 (Critical): `.tolist()` GPU Synchronization — 36-72x per Forward

**Location:**
- `hybrid_linear_attn_backend.py:795` — factor capture path: `start_locs = query_start_loc[:len(reqs) + 1].tolist()`
- `hybrid_linear_attn_backend.py:746` — reconstruction path: `cache_indices_list = cache_indices[:num_reqs].tolist()`

**Problem:** `.tolist()` on a CUDA tensor forces `cudaDeviceSynchronize`, stalling the GPU pipeline. These calls happen **inside per-layer forward**, meaning 36-72 implicit CUDA sync points per forward pass.

**Impact:** Each sync stalls the GPU for ~5-20us while waiting for all prior kernels to finish. At 72 syncs x 10us = ~0.7ms per forward. For 128 requests this is ~90ms total wasted time.

**Fix:** Precompute `start_locs` and `cache_indices_list` once at the beginning of the `forward_extend` call and pass them through to all layers. Both tensors are constant across layers.

---

## Overhead #2 (Critical): Unconditional Factor Capture on ALL Requests

**Location:** `schedule_batch.py:2027`
```python
req._needs_factor_capture = True  # Set for ALL reqs unconditionally
```

**Problem:** Every request in the batch captures factors in every GDN layer, including:
- Requests that already have full mamba state via CoW (copy-on-write) — their factors are wasted
- Requests where factors will be immediately evicted due to memory budget — their factors are wasted

For a batch of B requests where only F need factor capture, `(B - F) / B` of the capture work is wasted.

**Cost per request:** 4 `.clone()` x 36 GDN layers = **144 GPU memory allocations + copies** per request per forward. For the shared-prefix benchmark with batch size 16, that's 2,304 unnecessary GPU clones per forward batch if all 16 got CoW hits.

**Fix:** Only set `_needs_factor_capture = True` on requests that:
1. Are in extend (prefill) mode, AND
2. Did NOT get a full mamba state from CoW, AND
3. Have new tokens to process (extend_input_len > 0)

---

## Overhead #3 (High): 4x `.clone()` Per Layer Per Request

**Location:** `hybrid_linear_attn_backend.py:809-812`
```python
req.pending_lfc_factors[layer_id] = (
    key[0, start_idx:end_idx].clone(),
    value[0, start_idx:end_idx].clone(),
    g[0, start_idx:end_idx].clone(),
    beta[0, start_idx:end_idx].clone(),
)
```

**Problem:** Each `.clone()` allocates new GPU memory and launches a memcpy kernel. For B=16 requests x 36 GDN layers, that's **2,304 individual GPU memory allocations** per forward pass. Each allocation goes through PyTorch's CUDA memory allocator, which has overhead beyond the raw memcpy.

**Per-request memory copied per forward (all 36 layers):**
- k_factors: 36 layers x tokens x 4 heads x 128 dim x 2 bytes (bf16)
- v_factors: 36 layers x tokens x 8 heads x 128 dim x 2 bytes
- g_factors: 36 layers x tokens x 8 heads x 2 bytes
- beta_factors: 36 layers x tokens x 8 heads x 2 bytes

For a 4334-token prefix: ~36 x 4334 x (4x128 + 8x128 + 8 + 8) x 2 ≈ **478 MB per request**.

**Fix options:**
1. **Batch the clones**: Instead of 4 separate clones, use a single `torch.cat` or pre-allocated buffer to store all 4 factor types contiguously, reducing allocation count by 4x.
2. **Skip capture for CoW-hit requests** (see Overhead #2).
3. **Use a pre-allocated factor staging buffer**: Allocate a fixed buffer at init time and copy into it using indexed operations, avoiding per-call `cudaMalloc`.

---

## Overhead #4 (Medium): Python For-Loop in GPU Kernel Boundary

**Location:** `hybrid_linear_attn_backend.py:796-818`
```python
for i, req in enumerate(reqs):
    if not getattr(req, '_needs_factor_capture', False):
        continue
    ...
    req.pending_lfc_factors[layer_id] = (...)
```

**Problem:** A Python-level loop runs between GPU kernel launches, once per GDN layer. This creates "GPU bubbles" — the GPU finishes one kernel but must wait for the CPU to iterate through Python, do attribute lookups, dict insertions, and launch the next kernel.

**Cost:** With B=16 requests x 36 layers: 576 loop iterations with Python overhead (attribute lookups, dict operations). Estimated ~2-5ms total CPU time that causes GPU stalls.

**Fix:** Replace the per-request Python loop with a batched tensor operation:
```python
# Instead of cloning per-request, slice all at once:
all_k_factors = key[0].split(seq_lens)  # single call splits into list of views
```
Or better: let the reconstruction kernel operate on the original concatenated tensor directly using start/end indices, avoiding clones entirely.

---

## Overhead #5 (Medium): Reconstruction Per-Request Sequential Loop

**Location:** `hybrid_linear_attn_backend.py:747-769`
```python
for i in range(num_reqs):
    cache_idx = cache_indices_list[i]
    req_factors = lfc_reconstruction_factors.get(i)
    if req_factors is not None and layer_id in req_factors:
        snapshot = lfc_reconstruct_state(...)
        ssm_states[cache_idx] = snapshot.squeeze(0)
```

**Problem:** Each request's reconstruction runs as a separate Triton kernel launch with N=1. For B=16 requests all needing reconstruction in the same batch, that's 16 separate kernel launches per layer, each operating on a tiny workload (N=1, H_v=8).

**Cost:** 16 kernel launches x 36 layers = 576 Triton kernel launches. Even at ~6ms per call (our micro-benchmark), the per-call overhead for N=1 is dominated by kernel launch latency (~50-100us), not compute.

**Fix:** Batch all requests needing reconstruction into a single kernel call with N=batch_size. The Triton kernel already supports N>1 in its grid:
```python
# Instead of looping over requests:
# Gather all snapshots and factors into batched tensors
# Call kernel once with N=num_reqs_needing_reconstruction
```

---

## Overhead #6 (Low-Medium): `torch.cat()` Factor Chain Merging

**Location:** `mamba_radix_cache.py:494-511`
```python
req.lfc_reconstruction_factors[layer_id] = tuple(
    torch.cat([c[j] for c in chunk_list], dim=0)
    for j in range(len(chunk_list[0]))
)
```

**Problem:** When the factor chain has multiple nodes, factors from each node must be concatenated per layer. With 36 layers and a chain of length C, this is 4 x 36 x (C-1) = up to 144C `torch.cat` operations. Each cat allocates new GPU memory.

**Impact:** This runs once per cache miss, not per forward. But during burst arrivals (many requests from the same group), multiple requests may simultaneously trigger chain merging, causing GPU memory allocation storms.

**Fix:** Pre-allocate a reconstruction buffer and copy into it rather than creating new tensors via cat.

---

## Overhead #7 (Low): Factor Splitting During Tree Node Split

**Location:** `mamba_radix_cache.py:1067-1079`

When the radix tree splits a node, factors must be split too: 8 `.clone()` calls x 36 layers = **288 GPU allocations** per split. Tree splits happen when a new request diverges from a cached path (the first time tombstoning occurs for a prefix group).

**Fix:** Store factors as a single contiguous tensor with start/end indices. Splitting becomes a metadata update (O(1)) instead of 288 GPU clones.

---

## Prioritized Optimization Plan

### Phase 1: Low-Hanging Fruit (Expected: eliminate ~50% of overhead)

| Fix | Expected Impact | Effort |
|-----|----------------|--------|
| **Cache `.tolist()` results** — compute once at forward_extend entry | Eliminate 36-72 CUDA syncs per forward | Small |
| **Conditional factor capture** — only capture for requests that need it | Eliminate 50-80% of `.clone()` calls | Small |
| **Cache `is_lfc_enabled()`** — check once per forward, not per layer | Minor cleanup | Trivial |

### Phase 2: Batching Optimizations (Expected: eliminate remaining overhead)

| Fix | Expected Impact | Effort |
|-----|----------------|--------|
| **Batch reconstruction** — call Triton kernel with N>1 | Reduce 576 kernel launches to 36 | Medium |
| **Batch factor capture** — single split+clone per batch | Reduce 2,304 clones to ~100 | Medium |
| **Pre-allocated factor buffer** — avoid per-clone cudaMalloc | Eliminate allocation overhead | Medium |

### Phase 3: Structural Improvements (Expected: further 2-3% improvement)

| Fix | Expected Impact | Effort |
|-----|----------------|--------|
| **Zero-copy factor splitting** — indices instead of clone | Eliminate 288 clones per split | Medium |
| **Cache factor memory estimates** — store size on node | Minor CPU savings | Trivial |
| **Pre-allocated reconstruction buffer** — avoid cat allocations | Reduce memory fragmentation | Medium |

---

## Expected Outcome

With Phase 1 alone, LFC should be **faster than Standard** for shared-prefix workloads:
- Standard TTFT ≈ 443ms (full recompute for tombstoned prefixes)
- LFC TTFT should be ≈ 350-400ms (fast reconstruction via Triton kernel, minimal capture overhead)

With Phase 1+2, the gap should widen further, achieving the theoretical ~10-20% TTFT improvement from LFC.
