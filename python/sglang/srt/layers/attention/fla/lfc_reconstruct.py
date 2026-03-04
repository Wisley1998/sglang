"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
LFC (Linear Factor Caching) State Reconstruction Kernel.

This module provides fast state reconstruction from cached factors for
hybrid attention+SSM models (GLA/Gated DeltaNet).

The key insight is that instead of recomputing the full SSM state from the
snapshot position, we can store intermediate factors (k, v, g, beta) and
apply them to reconstruct the state much faster.

Theoretical speedup: ~3.16x (skips projections)
Storage overhead: ~1.05% of full state size
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _lfc_reconstruct_state_kernel(
    # Pointers to matrices
    snapshot_ptr,  # [N, H_v, K, V]
    k_factors_ptr,  # [N, delta, H_k, K]
    v_factors_ptr,  # [N, delta, H_v, V]
    g_factors_ptr,  # [N, delta, H_v]
    beta_factors_ptr,  # [N, delta, H_v]
    output_ptr,  # [N, H_v, K, V]
    # Matrix dimensions
    N,  # batch size
    delta,  # number of timesteps to replay
    H_v,  # number of value heads (grid dimension)
    H_k,  # number of key heads (H_k <= H_v, H_v % H_k == 0)
    K,  # key dimension
    V,  # value dimension
    # Strides
    stride_snapshot_n,
    stride_snapshot_h,
    stride_snapshot_k,
    stride_k_n,
    stride_k_t,
    stride_k_h,
    stride_v_n,
    stride_v_t,
    stride_v_h,
    stride_g_n,
    stride_g_t,
    stride_output_n,
    stride_output_h,
    stride_output_k,
    # Block sizes
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Triton kernel for LFC state reconstruction with GQA support.

    For each timestep t in [0, delta):
        state = state * g[t] + beta[t] * outer(k[t], v[t])

    where outer(k, v) is the outer product k @ v^T.

    GQA mapping: each key head is shared by (H_v // H_k) value heads.
    Key head index for value head pid_h is: pid_h // (H_v // H_k).
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)  # value head index

    # GQA: map value head to key head via integer division
    pid_h_k = pid_h // (H_v // H_k)

    # Load initial state from snapshot (indexed by value head)
    state_ptrs = snapshot_ptr + pid_n * stride_snapshot_n + pid_h * stride_snapshot_h

    # Create masks and offsets for K and V dimensions
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    # Load snapshot state
    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for ki in range(0, K, BLOCK_K):
        for vi in range(0, V, BLOCK_V):
            k_idx = ki + k_offs
            v_idx = vi + v_offs
            k_valid = k_idx < K
            v_valid = v_idx < V
            mask = k_valid[:, None] & v_valid[None, :]
            ptrs = state_ptrs + k_idx[:, None] * stride_snapshot_k + v_idx[None, :]
            chunk = tl.load(ptrs, mask=mask, other=0.0)
            state += tl.where(mask, chunk, 0.0)

    # Apply each factor step
    for t in range(delta):
        # Load gating factor g[t] (indexed by value head)
        g_ptr = g_factors_ptr + pid_n * stride_g_n + t * stride_g_t + pid_h
        g = tl.load(g_ptr)

        # Load beta factor beta[t] (indexed by value head)
        beta_ptr = beta_factors_ptr + pid_n * stride_g_n + t * stride_g_t + pid_h
        beta = tl.load(beta_ptr)

        # Load k[t] vector: [K] (indexed by key head via GQA mapping)
        k_ptrs = k_factors_ptr + pid_n * stride_k_n + t * stride_k_t + pid_h_k * stride_k_h
        k_vec = tl.load(k_ptrs + k_offs, mask=k_mask, other=0.0)

        # Load v[t] vector: [V] (indexed by value head)
        v_ptrs = v_factors_ptr + pid_n * stride_v_n + t * stride_v_t + pid_h * stride_v_h
        v_vec = tl.load(v_ptrs + v_offs, mask=v_mask, other=0.0)

        # Compute outer product: k @ v^T
        outer = k_vec[:, None] * v_vec[None, :]

        # Update state: state = state * exp(g) + beta * outer
        state = state * tl.exp(g) + beta * outer

    # Store output state (indexed by value head)
    output_ptrs = output_ptr + pid_n * stride_output_n + pid_h * stride_output_h
    for ki in range(0, K, BLOCK_K):
        for vi in range(0, V, BLOCK_V):
            k_idx = ki + k_offs
            v_idx = vi + v_offs
            k_valid = k_idx < K
            v_valid = v_idx < V
            mask = k_valid[:, None] & v_valid[None, :]
            ptrs = output_ptrs + k_idx[:, None] * stride_output_k + v_idx[None, :]
            # Extract the corresponding chunk of state
            chunk_start_k = ki
            chunk_start_v = vi
            chunk = state
            tl.store(ptrs, chunk, mask=mask)


def lfc_reconstruct_state(
    snapshot: torch.Tensor,
    k_factors: torch.Tensor,
    v_factors: torch.Tensor,
    g_factors: torch.Tensor,
    beta_factors: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct SSM state by applying cached factors to snapshot.

    Uses a fused Triton kernel that processes the entire delta loop in a
    single GPU kernel launch, avoiding the overhead of Python-loop-based
    per-timestep kernel dispatches.

    Supports GQA-style asymmetric head counts where H_k <= H_v and
    H_v % H_k == 0. Each key head is shared by (H_v // H_k) value heads.

    Args:
        snapshot: Initial state tensor [N, H_v, K, V]
        k_factors: Cached key factors [N, delta, H_k, K] (H_k <= H_v)
        v_factors: Cached value factors [N, delta, H_v, V]
        g_factors: Cached gating factors [N, delta, H_v]
        beta_factors: Cached beta factors [N, delta, H_v]

    Returns:
        Reconstructed state [N, H_v, K, V]
    """
    N, H_v, K, V = snapshot.shape
    delta = k_factors.shape[1]
    H_k = k_factors.shape[2]

    # Validate input shapes
    assert k_factors.shape == (N, delta, H_k, K), f"k_factors shape mismatch: {k_factors.shape}"
    assert v_factors.shape == (N, delta, H_v, V), f"v_factors shape mismatch: {v_factors.shape}"
    assert g_factors.shape == (N, delta, H_v), f"g_factors shape mismatch: {g_factors.shape}"
    assert beta_factors.shape == (N, delta, H_v), f"beta_factors shape mismatch: {beta_factors.shape}"
    assert H_v % H_k == 0, f"H_v ({H_v}) must be divisible by H_k ({H_k})"

    if delta == 0:
        return snapshot.clone()

    # Ensure contiguous float32 inputs for the Triton kernel
    snapshot_f32 = snapshot.contiguous().float()
    k_factors = k_factors.contiguous().float()
    v_factors = v_factors.contiguous().float()
    g_factors = g_factors.contiguous().float()
    beta_factors = beta_factors.contiguous().float()

    output = torch.empty_like(snapshot_f32)

    # Block sizes must cover the full K and V dimensions (power-of-2)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_V = triton.next_power_of_2(V)

    grid = (N, H_v)
    _lfc_reconstruct_state_kernel[grid](
        snapshot_f32,
        k_factors,
        v_factors,
        g_factors,
        beta_factors,
        output,
        N, delta, H_v, H_k, K, V,
        # snapshot strides [N, H_v, K, V]
        snapshot_f32.stride(0), snapshot_f32.stride(1), snapshot_f32.stride(2),
        # k_factors strides [N, delta, H_k, K]
        k_factors.stride(0), k_factors.stride(1), k_factors.stride(2),
        # v_factors strides [N, delta, H_v, V]
        v_factors.stride(0), v_factors.stride(1), v_factors.stride(2),
        # g_factors strides [N, delta, H_v]
        g_factors.stride(0), g_factors.stride(1),
        # output strides [N, H_v, K, V]
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    return output.to(snapshot.dtype)


@triton.jit
def _lfc_reconstruct_state_batched_kernel(
    # Pointers to matrices
    snapshot_ptr,  # [N, H_v, K, V]
    k_factors_ptr,  # [N, MAX_DELTA, H_k, K] (zero-padded)
    v_factors_ptr,  # [N, MAX_DELTA, H_v, V] (zero-padded)
    g_factors_ptr,  # [N, MAX_DELTA, H_v] (zero-padded)
    beta_factors_ptr,  # [N, MAX_DELTA, H_v] (zero-padded)
    delta_per_req_ptr,  # [N] per-request delta values
    output_ptr,  # [N, H_v, K, V]
    # Matrix dimensions
    N,
    MAX_DELTA,  # padded delta (loop bound)
    H_v,
    H_k,
    K,
    V,
    # Strides
    stride_snapshot_n,
    stride_snapshot_h,
    stride_snapshot_k,
    stride_k_n,
    stride_k_t,
    stride_k_h,
    stride_v_n,
    stride_v_t,
    stride_v_h,
    stride_g_n,
    stride_g_t,
    stride_output_n,
    stride_output_h,
    stride_output_k,
    # Block sizes
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Batched Triton kernel for LFC state reconstruction.

    Each program handles one (request, head) pair. Per-request delta values
    allow different requests to have different numbers of factor timesteps,
    with zero-padded regions skipped via conditional application.

    When t >= my_delta, g=0 so exp(g)=1 and beta=0, making the update a no-op.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    # GQA: map value head to key head
    pid_h_k = pid_h // (H_v // H_k)

    # Load per-request delta
    my_delta = tl.load(delta_per_req_ptr + pid_n)

    # Load initial state from snapshot
    state_ptrs = snapshot_ptr + pid_n * stride_snapshot_n + pid_h * stride_snapshot_h
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    mask = k_mask[:, None] & v_mask[None, :]

    ptrs = state_ptrs + k_offs[:, None] * stride_snapshot_k + v_offs[None, :]
    state = tl.load(ptrs, mask=mask, other=0.0)

    # Apply factor steps (padded region is zero -> no-op)
    for t in range(MAX_DELTA):
        should_apply = t < my_delta

        # Load gating factor g[t]
        g_addr = g_factors_ptr + pid_n * stride_g_n + t * stride_g_t + pid_h
        g = tl.load(g_addr)
        g = tl.where(should_apply, g, 0.0)

        # Load beta factor beta[t]
        beta_addr = beta_factors_ptr + pid_n * stride_g_n + t * stride_g_t + pid_h
        beta_val = tl.load(beta_addr)
        beta_val = tl.where(should_apply, beta_val, 0.0)

        # Load k[t] vector (key head via GQA mapping)
        k_base = k_factors_ptr + pid_n * stride_k_n + t * stride_k_t + pid_h_k * stride_k_h
        k_vec = tl.load(k_base + k_offs, mask=k_mask, other=0.0)

        # Load v[t] vector (value head)
        v_base = v_factors_ptr + pid_n * stride_v_n + t * stride_v_t + pid_h * stride_v_h
        v_vec = tl.load(v_base + v_offs, mask=v_mask, other=0.0)

        # outer product + gated update
        outer = k_vec[:, None] * v_vec[None, :]
        state = state * tl.exp(g) + beta_val * outer

    # Store output state
    output_ptrs = output_ptr + pid_n * stride_output_n + pid_h * stride_output_h
    out = output_ptrs + k_offs[:, None] * stride_output_k + v_offs[None, :]
    tl.store(out, state, mask=mask)


def lfc_reconstruct_state_batched(
    snapshots: torch.Tensor,
    k_factors_list: list,
    v_factors_list: list,
    g_factors_list: list,
    beta_factors_list: list,
) -> torch.Tensor:
    """
    Batched LFC state reconstruction — single kernel launch for N requests.

    Instead of launching N separate kernels (one per request), this function
    pads factors to the max delta across the batch and launches one kernel
    with grid=(N, H_v). Per-request deltas ensure only valid timesteps
    are applied.

    Args:
        snapshots: Initial states [N, H_v, K, V]
        k_factors_list: List of N tensors, each [delta_i, H_k, K]
        v_factors_list: List of N tensors, each [delta_i, H_v, V]
        g_factors_list: List of N tensors, each [delta_i, H_v]
        beta_factors_list: List of N tensors, each [delta_i, H_v]

    Returns:
        Reconstructed states [N, H_v, K, V]
    """
    N, H_v, K, V = snapshots.shape
    H_k = k_factors_list[0].shape[1]
    device = snapshots.device

    deltas = [kf.shape[0] for kf in k_factors_list]
    max_delta = max(deltas)

    if max_delta == 0:
        return snapshots.clone()

    # Pad and stack factors into [N, max_delta, ...] tensors (zero-padded)
    k_batched = torch.zeros(N, max_delta, H_k, K, device=device, dtype=torch.float32)
    v_batched = torch.zeros(N, max_delta, H_v, V, device=device, dtype=torch.float32)
    g_batched = torch.zeros(N, max_delta, H_v, device=device, dtype=torch.float32)
    beta_batched = torch.zeros(N, max_delta, H_v, device=device, dtype=torch.float32)

    for i in range(N):
        d = deltas[i]
        if d > 0:
            k_batched[i, :d] = k_factors_list[i].float()
            v_batched[i, :d] = v_factors_list[i].float()
            g_batched[i, :d] = g_factors_list[i].float()
            beta_batched[i, :d] = beta_factors_list[i].float()

    delta_tensor = torch.tensor(deltas, dtype=torch.int32, device=device)

    snapshot_f32 = snapshots.contiguous().float()
    output = torch.empty_like(snapshot_f32)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_V = triton.next_power_of_2(V)

    grid = (N, H_v)
    _lfc_reconstruct_state_batched_kernel[grid](
        snapshot_f32,
        k_batched,
        v_batched,
        g_batched,
        beta_batched,
        delta_tensor,
        output,
        N, max_delta, H_v, H_k, K, V,
        # snapshot strides [N, H_v, K, V]
        snapshot_f32.stride(0), snapshot_f32.stride(1), snapshot_f32.stride(2),
        # k_factors strides [N, max_delta, H_k, K]
        k_batched.stride(0), k_batched.stride(1), k_batched.stride(2),
        # v_factors strides [N, max_delta, H_v, V]
        v_batched.stride(0), v_batched.stride(1), v_batched.stride(2),
        # g_factors strides [N, max_delta, H_v]
        g_batched.stride(0), g_batched.stride(1),
        # output strides [N, H_v, K, V]
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    return output.to(snapshots.dtype)


def lfc_reconstruct_state_fused(
    snapshot: torch.Tensor,
    k_factors: torch.Tensor,
    v_factors: torch.Tensor,
    g_factors: torch.Tensor,
    beta_factors: torch.Tensor,
) -> torch.Tensor:
    """
    Fused version of LFC state reconstruction using einsum for better performance.

    This computes the same result as lfc_reconstruct_state but in a more
    memory-efficient way by fusing operations.

    Supports GQA-style asymmetric head counts (H_k <= H_v).
    """
    N, H_v, K, V = snapshot.shape
    delta = k_factors.shape[1]
    H_k = k_factors.shape[2]

    state = snapshot.clone().float()

    # GQA: expand k_factors once outside the loop (only when H_k != H_v)
    if H_k < H_v:
        k_factors = k_factors.repeat_interleave(H_v // H_k, dim=2)  # [N, delta, H_v, K]

    for t in range(delta):
        g = g_factors[:, t:t+1, :].unsqueeze(-1).unsqueeze(-1)  # [N, 1, H_v, 1, 1]
        beta = beta_factors[:, t:t+1, :].unsqueeze(-1).unsqueeze(-1)  # [N, 1, H_v, 1, 1]

        k_t = k_factors[:, t, :, :]  # [N, H_v, K] (already expanded)
        v_t = v_factors[:, t, :, :]  # [N, H_v, V]

        outer = torch.einsum("nhk,nhv->nhkv", k_t, v_t)

        state = state * torch.exp(g.squeeze(1)) + beta.squeeze(1) * outer

    return state.to(snapshot.dtype)


class LFCFactorCache:
    """
    Cache for storing LFC factors associated with radix tree nodes.

    This class manages the storage and retrieval of factors (k, v, g, beta)
    that are computed during prefill and cached for fast state reconstruction.
    """

    def __init__(
        self,
        max_slots: int,
        max_factors_per_slot: int,
        num_key_heads: int,
        key_dim: int,
        value_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_value_heads: Optional[int] = None,
    ):
        """
        Initialize the LFC factor cache.

        Args:
            max_slots: Maximum number of cache slots
            max_factors_per_slot: Maximum number of factors per slot (typically K-1 for snapshot interval K)
            num_key_heads: Number of key heads (H_k)
            key_dim: Key dimension per head
            value_dim: Value dimension per head
            dtype: Data type for storage
            device: Device for tensors
            num_value_heads: Number of value heads (H_v). Defaults to num_key_heads
                for backward compatibility (H_k == H_v, e.g. FalconH1).
        """
        self.max_slots = max_slots
        self.max_factors_per_slot = max_factors_per_slot
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads if num_value_heads is not None else num_key_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dtype = dtype
        self.device = device

        # Allocate factor pools
        # k_factor: [num_slots, max_factors, num_key_heads, key_dim]
        self.k_factor_pool = torch.zeros(
            (max_slots, max_factors_per_slot, self.num_key_heads, key_dim),
            dtype=dtype,
            device=device,
        )

        # v_factor: [num_slots, max_factors, num_value_heads, value_dim]
        self.v_factor_pool = torch.zeros(
            (max_slots, max_factors_per_slot, self.num_value_heads, value_dim),
            dtype=dtype,
            device=device,
        )

        # g_factor: [num_slots, max_factors, num_value_heads]
        self.g_factor_pool = torch.zeros(
            (max_slots, max_factors_per_slot, self.num_value_heads),
            dtype=dtype,
            device=device,
        )

        # beta_factor: [num_slots, max_factors, num_value_heads]
        self.beta_factor_pool = torch.zeros(
            (max_slots, max_factors_per_slot, self.num_value_heads),
            dtype=dtype,
            device=device,
        )

        # Track valid factor lengths per slot
        self.factor_valid_len = torch.zeros(max_slots, dtype=torch.int32, device=device)

        # Free slot tracking
        self.free_slots = list(range(max_slots))

    def alloc(self, need_size: int = 1) -> Optional[torch.Tensor]:
        """Allocate cache slots."""
        if need_size > len(self.free_slots):
            return None

        select_indices = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return torch.tensor(select_indices, dtype=torch.int64, device=self.device)

    def free(self, indices: torch.Tensor):
        """Free cache slots."""
        if indices.numel() == 0:
            return

        indices_list = indices.cpu().tolist()
        self.free_slots.extend(indices_list)

        # Clear the freed slots
        self.factor_valid_len[indices] = 0

    def store_factors(
        self,
        slot_idx: int,
        k_factors: torch.Tensor,
        v_factors: torch.Tensor,
        g_factors: torch.Tensor,
        beta_factors: torch.Tensor,
    ):
        """
        Store factors for a given slot.

        Args:
            slot_idx: Slot index
            k_factors: Key factors [delta, num_heads, key_dim]
            v_factors: Value factors [delta, num_heads, value_dim]
            g_factors: Gating factors [delta, num_heads]
            beta_factors: Beta factors [delta, num_heads]
        """
        delta = k_factors.shape[0]
        assert delta <= self.max_factors_per_slot, f"delta {delta} exceeds max {self.max_factors_per_slot}"

        self.k_factor_pool[slot_idx, :delta] = k_factors
        self.v_factor_pool[slot_idx, :delta] = v_factors
        self.g_factor_pool[slot_idx, :delta] = g_factors
        self.beta_factor_pool[slot_idx, :delta] = beta_factors
        self.factor_valid_len[slot_idx] = delta

    def load_factors(
        self, slot_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load factors for a given slot.

        Args:
            slot_idx: Slot index

        Returns:
            Tuple of (k_factors, v_factors, g_factors, beta_factors)
        """
        valid_len = self.factor_valid_len[slot_idx].item()

        k_factors = self.k_factor_pool[slot_idx, :valid_len]
        v_factors = self.v_factor_pool[slot_idx, :valid_len]
        g_factors = self.g_factor_pool[slot_idx, :valid_len]
        beta_factors = self.beta_factor_pool[slot_idx, :valid_len]

        return k_factors, v_factors, g_factors, beta_factors

    def get_valid_len(self, slot_idx: int) -> int:
        """Get the number of valid factors stored in a slot."""
        return self.factor_valid_len[slot_idx].item()

    def has_factors(self, slot_idx: int) -> bool:
        """Check if a slot has any stored factors."""
        return self.factor_valid_len[slot_idx].item() > 0

    def clear(self):
        """Clear all stored factors."""
        self.k_factor_pool.zero_()
        self.v_factor_pool.zero_()
        self.g_factor_pool.zero_()
        self.beta_factor_pool.zero_()
        self.factor_valid_len.zero_()
        self.free_slots = list(range(self.max_slots))

    def mem_usage_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        return (
            self.k_factor_pool.numel() * self.k_factor_pool.element_size() +
            self.v_factor_pool.numel() * self.v_factor_pool.element_size() +
            self.g_factor_pool.numel() * self.g_factor_pool.element_size() +
            self.beta_factor_pool.numel() * self.beta_factor_pool.element_size() +
            self.factor_valid_len.numel() * self.factor_valid_len.element_size()
        )


def lfc_reconstruct_all_gdn_layers(
    temporal: torch.Tensor,
    mamba_map: dict,
    reconstruction_factors: dict,
    cache_indices_list: list,
    gdn_layer_ids: list,
) -> bool:
    """
    Fused cross-layer GDN reconstruction — single kernel launch for all layers.

    Instead of launching one batched kernel per GDN layer (18 launches),
    this function packs all (layer, request) pairs into a single batch
    and calls lfc_reconstruct_state_batched once.

    Args:
        temporal: Full SSM state tensor [num_physical_layers, num_slots, H_v, K, V]
        mamba_map: Dict mapping logical layer_id -> physical layer index
        reconstruction_factors: Dict mapping req_idx -> {layer_id: (k, v, g, beta)}
        cache_indices_list: List of cache indices per request
        gdn_layer_ids: List of GDN layer IDs (logical) to reconstruct

    Returns:
        True if any reconstructions were performed, False otherwise.
    """
    if not reconstruction_factors:
        return False

    gdn_layer_set = set(gdn_layer_ids)

    # Collect all (layer, request) pairs needing GDN reconstruction
    snapshots_list = []
    k_list, v_list, g_list, beta_list = [], [], [], []
    scatter_targets = []  # (physical_layer_idx, cache_idx) for scatter-back

    for req_idx, req_factors in reconstruction_factors.items():
        if req_factors is None:
            continue
        cache_idx = cache_indices_list[req_idx]
        for layer_id in gdn_layer_ids:
            if layer_id not in req_factors:
                continue
            phys_idx = mamba_map[layer_id]
            snapshot = temporal[phys_idx, cache_idx]  # [H_v, K, V]
            k_f, v_f, g_f, beta_f = req_factors[layer_id]
            snapshots_list.append(snapshot)
            k_list.append(k_f)
            v_list.append(v_f)
            g_list.append(g_f)
            beta_list.append(beta_f)
            scatter_targets.append((phys_idx, cache_idx))

    if not scatter_targets:
        return False

    n_pairs = len(scatter_targets)

    if n_pairs == 1:
        # Single pair: use non-batched kernel to avoid padding overhead
        phys_idx, cache_idx = scatter_targets[0]
        result = lfc_reconstruct_state(
            snapshots_list[0].unsqueeze(0),
            k_list[0].unsqueeze(0),
            v_list[0].unsqueeze(0),
            g_list[0].unsqueeze(0),
            beta_list[0].unsqueeze(0),
        )
        temporal[phys_idx, cache_idx] = result.squeeze(0).to(temporal.dtype)
    else:
        # Batch all (layer, request) pairs into single kernel launch
        snapshots = torch.stack(snapshots_list)  # [N_pairs, H_v, K, V]
        result = lfc_reconstruct_state_batched(
            snapshots, k_list, v_list, g_list, beta_list,
        )
        # Scatter results back to per-layer SSM state pools
        result_typed = result.to(temporal.dtype, copy=False)
        for i, (phys_idx, cache_idx) in enumerate(scatter_targets):
            temporal[phys_idx, cache_idx] = result_typed[i]

    return True
