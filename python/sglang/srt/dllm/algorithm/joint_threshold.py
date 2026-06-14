import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class ShrinkState:
    """Saved forward_batch handles, used to restore after physical shrink.

    When enough slots have converged within a block, the active prefix of the
    batch can be sliced to a smaller CUDA graph bucket. The sliced views share
    storage with these saved full tensors, so in-place mutations on active
    slots remain visible after restoring the original batch shape.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    orig_seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    batch_size: int
    seq_lens_sum: int


@dataclass
class RequestAlgoState:
    """Per-request continuation state for the JointThreshold algorithm.

    The current scheduler resets block-local fields at each block boundary.
    Keeping the state on the request still makes the algorithm compatible with
    finer-grained scheduling, where slots may resume mid-block.
    """

    # Block-local (reset at the start of each new block in current scheduler)
    post_edit_steps: int = 0
    finished: bool = False
    active_forward_count: int = 0
    already_dead: bool = False
    # Per-slot iteration counter inside the current diffusion block.
    iter_in_block: int = 0
    # Cross-block progress marker, currently advisory.
    blocks_completed: int = 0


@dataclass
class BlockIterState:
    """Per-block, per-iter mutable state held outside the inner loop.

    `run()` owns one of these objects while it advances a diffusion block. The
    state keeps slot-local tensors, request references, and shrink/restore
    metadata together so the update loop and the early-exit path share one
    consistent view of the batch.
    """

    # Shapes / context (set once at prepare time)
    batch_size: int
    block_size: int
    device: torch.device
    max_iterations: int

    # Per-slot static for the duration of one block
    prompt_masks: list
    start_list: list
    prefix_lens: torch.Tensor
    prefix_lens_cpu: torch.Tensor | None
    original_seq_lens: torch.Tensor
    original_seq_lens_cpu: torch.Tensor | None
    original_seq_lens_sum: int
    cuda_graph_bs: set
    skip_supported_for_batch: bool

    # Per-slot mutable across iters (tensor mirrors of per-req algo_state)
    post_edit_steps: torch.Tensor
    finished: torch.Tensor
    active_forward_counts: torch.Tensor
    already_dead_mask: torch.Tensor
    logical_order: torch.Tensor
    reqs: list | None
    # Per-request continuation state in slot order. Synthetic entries keep the
    # tensor seed/persist path uniform when request objects are unavailable.
    algo_states: list = field(default_factory=list)

    # Iter-loop counters
    # The outer loop continues while any active slot still has remaining iters.
    # `iteration_idx` is kept as a coarse block-level progress marker.
    iter_in_block: torch.Tensor | None = None
    iteration_idx: int = 0
    measured_forwards_since_yield: int = 0
    any_changed_in_last_step: bool = False

    # Last forward output handles (filled by step())
    logits_output: Any | None = None
    can_run_cuda_graph: bool | None = None
    last_changed_slots: torch.Tensor | None = None

    # Saved forward_batch views for physical shrink; None means not shrunk.
    shrink_state: ShrinkState | None = None
    shrink_stack: list = field(default_factory=list)


class JointThreshold(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)
        yield_every = config.algorithm_config.get("yield_every", None)
        if yield_every is None and config.algorithm_config.get("half_step", False):
            yield_every = self.block_size // 2
        if yield_every is None:
            yield_every = config.yield_every
        try:
            yield_every = int(yield_every) if yield_every is not None else None
        except (TypeError, ValueError):
            yield_every = None
        self.yield_every = yield_every if yield_every and yield_every > 0 else None
        self.enable_dead_slot_skip = bool(
            config.algorithm_config.get("enable_dead_slot_skip", False)
        )
        # Physically shrink finished slots out of the forward batch so later
        # denoise iterations can dispatch to a smaller CUDA graph bucket.
        self.enable_physical_shrink = bool(
            config.algorithm_config.get("enable_physical_shrink", False)
        )
        # Allow repeated shrink decisions within the same diffusion block.
        self.enable_multi_shrink = bool(
            config.algorithm_config.get("enable_multi_shrink", False)
        )
        # Use terminal-finished slots as safe padding for CUDA graph buckets.
        self.enable_bucket_padding = bool(
            config.algorithm_config.get("enable_bucket_padding", False)
        )
        # Coalesce host syncs on the shrink decision path.
        self.enable_sync_coalesce = bool(
            config.algorithm_config.get("enable_sync_coalesce", False)
        )
        # Reorder slot-aligned tensors with fewer GPU indexing operations.
        self.enable_reorder_fusion = bool(
            config.algorithm_config.get("enable_reorder_fusion", False)
        )
        # Small batches cannot benefit from within-block physical shrink: when
        # bs=1 there is no dead slot to remove. Keep this enabled by default so
        # single-request latency paths avoid the state-machine bookkeeping while
        # high-concurrency paths keep the optimized behavior.
        self.enable_small_batch_fast_path = bool(
            config.algorithm_config.get("enable_small_batch_fast_path", True)
        )
        self.small_batch_fast_path_min_bs = int(
            config.algorithm_config.get("small_batch_fast_path_min_bs", 2)
        )
        self.force_reference_small_batch = (
            self.enable_small_batch_fast_path
            and config.max_running_requests < self.small_batch_fast_path_min_bs
        )
        if self.force_reference_small_batch:
            os.environ.pop("SGLANG_DLLM_REDUCED_LOGITS", None)
        # Batch-vectorized JointThreshold decision/update path.
        self.enable_vectorized_step = (
            bool(config.algorithm_config.get("enable_vectorized_step", False))
            and not self.force_reference_small_batch
        )
        # Ask the logits processor for confidence statistics instead of full
        # vocabulary logits when the update path can consume them directly.
        self.enable_reduced_logits = (
            bool(config.algorithm_config.get("enable_reduced_logits", False))
            and not self.force_reference_small_batch
        )
        if self.enable_reduced_logits:
            os.environ["SGLANG_DLLM_REDUCED_LOGITS"] = "1"

    def _small_batch_fast_path_active(self, batch_size: int) -> bool:
        return (
            self.enable_small_batch_fast_path
            and batch_size < self.small_batch_fast_path_min_bs
        )

    def _can_use_reference_small_batch_path(self, forward_batch: ForwardBatch) -> bool:
        if (
            not self.force_reference_small_batch
            or forward_batch.batch_size >= self.small_batch_fast_path_min_bs
        ):
            return False
        return True

    def _run_reference_block(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        """Clean JointThreshold loop for single-request serving.

        The state-machine path is useful for batched shrink decisions, but it
        adds fixed overhead when the server can only run one request.
        """
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        start_list = []
        prompt_masks = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]

            prompt_mask = block_input_ids != self.mask_id
            prompt_masks.append(prompt_mask)
            start_list.append(prompt_mask.sum().item())

        post_edit_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        any_changed_in_last_step = False

        max_iterations = self.block_size + self.max_post_edit_steps
        for _ in range(max_iterations):
            if finished.all():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            any_changed_in_last_step = False

            for i in range(batch_size):
                if finished[i]:
                    continue

                block_start = i * self.block_size
                block_end = block_start + self.block_size

                curr_input_ids = forward_batch.input_ids[block_start:block_end]
                curr_logits = logits_output.full_logits[block_start:block_end]
                curr_prompt_mask = prompt_masks[i]

                if self.penalty_lambda > 0:
                    prev_ids = curr_input_ids[:-1]
                    curr_logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )

                mask_index = curr_input_ids == self.mask_id
                has_mask = mask_index.any()

                mask_transfer_index = torch.zeros_like(mask_index)
                if has_mask:
                    confidence = torch.where(mask_index, p, -np.inf)
                    mask_transfer_index = confidence > self.threshold

                    if not mask_transfer_index.any():
                        _, select_index = torch.topk(confidence, k=1)
                        mask_transfer_index[select_index] = True
                else:
                    post_edit_steps[i] += 1
                    if post_edit_steps[i] > self.max_post_edit_steps:
                        finished[i] = True
                        continue

                edit_mask = ~mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )

                transfer_index = mask_transfer_index | edit_transfer_index
                if not transfer_index.any():
                    finished[i] = True
                    continue

                curr_input_ids[transfer_index] = x[transfer_index]
                any_changed_in_last_step = True

        if any_changed_in_last_step:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph

    def _timed_forward(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        *,
        prompt_masks: list | None = None,
        case: str,
        measured: bool,
        preadvance: bool,
        forward_index: int,
        active_forward_counts: torch.Tensor | None,
        targets: list[int] | None,
    ):
        try:
            old_reduced_logits = getattr(forward_batch, "dllm_reduced_logits", None)
            use_reduced_logits = (
                self.enable_reduced_logits
                and self.enable_vectorized_step
                and self.penalty_lambda <= 0
            )
            forward_batch.dllm_reduced_logits = bool(use_reduced_logits)
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        finally:
            if old_reduced_logits is None:
                try:
                    delattr(forward_batch, "dllm_reduced_logits")
                except AttributeError:
                    pass
            else:
                forward_batch.dllm_reduced_logits = old_reduced_logits
        return out

    # ------------------------------------------------------------------
    # Tensor reorder helpers for keeping active slots contiguous.
    # ------------------------------------------------------------------
    @staticmethod
    def _reorder_slot_tensor(tensor: torch.Tensor | None, order: torch.Tensor) -> None:
        if tensor is None:
            return
        index = order if tensor.device.type != "cpu" else order.detach().cpu()
        tensor.copy_(tensor.index_select(0, index).clone())

    def _reorder_block_tensor(
        self,
        tensor: torch.Tensor | None,
        order: torch.Tensor,
        batch_size: int,
    ) -> None:
        if tensor is None:
            return
        index = order if tensor.device.type != "cpu" else order.detach().cpu()
        view_shape = (batch_size, self.block_size) + tuple(tensor.shape[1:])
        tensor.copy_(
            tensor.view(view_shape).index_select(0, index).clone().view_as(tensor)
        )

    @staticmethod
    def _reorder_list(values: list, order: torch.Tensor) -> list:
        order_cpu = [int(v) for v in order.detach().cpu().tolist()]
        return [values[i] for i in order_cpu]

    def _reorder_slots(
        self,
        forward_batch: ForwardBatch,
        state: BlockIterState,
        order: torch.Tensor,
    ) -> None:
        if torch.equal(
            order, torch.arange(state.batch_size, dtype=torch.long, device=state.device)
        ):
            return

        bs = state.batch_size
        self._reorder_block_tensor(forward_batch.input_ids, order, bs)
        self._reorder_block_tensor(forward_batch.positions, order, bs)
        self._reorder_block_tensor(forward_batch.out_cache_loc, order, bs)
        # Group compatible slot tensors so one index_select reorders several
        # fields that must stay aligned.
        if self.enable_reorder_fusion:
            self._reorder_fused(forward_batch, state, order)
        else:
            self._reorder_slot_tensor(forward_batch.req_pool_indices, order)
            self._reorder_slot_tensor(forward_batch.seq_lens, order)
            self._reorder_slot_tensor(forward_batch.seq_lens_cpu, order)
            self._reorder_slot_tensor(forward_batch.orig_seq_lens, order)
            self._reorder_slot_tensor(forward_batch.extend_prefix_lens, order)
            self._reorder_slot_tensor(forward_batch.extend_seq_lens, order)
        if state.prefix_lens_cpu is not None:
            state.prefix_lens_cpu = state.prefix_lens_cpu.index_select(
                0, order.detach().cpu()
            ).clone()

        state.prompt_masks = self._reorder_list(state.prompt_masks, order)
        state.start_list = self._reorder_list(state.start_list, order)
        if state.reqs is not None:
            state.reqs = self._reorder_list(state.reqs, order)
            forward_batch.reqs = state.reqs
        if state.algo_states:
            state.algo_states = self._reorder_list(state.algo_states, order)

        if self.enable_reorder_fusion:
            # Keep state tensors aligned with the reordered forward batch.
            int_tensors = [
                state.post_edit_steps,
                state.active_forward_counts,
            ]
            if state.iter_in_block is not None:
                int_tensors.append(state.iter_in_block)
            int_tensors.append(state.logical_order.to(torch.int64).to(torch.int32))
            stacked_int = torch.stack(
                [t.to(torch.int32) for t in int_tensors], dim=0
            ).index_select(1, order)
            state.post_edit_steps = stacked_int[0].clone()
            state.active_forward_counts = stacked_int[1].clone()
            i_off = 2
            if state.iter_in_block is not None:
                state.iter_in_block = stacked_int[i_off].clone()
                i_off += 1
            state.logical_order = stacked_int[i_off].to(torch.int64).clone()

            stacked_bool = torch.stack(
                [state.finished, state.already_dead_mask], dim=0
            ).index_select(1, order)
            state.finished = stacked_bool[0].clone()
            state.already_dead_mask = stacked_bool[1].clone()
        else:
            state.finished = state.finished.index_select(0, order).clone()
            state.post_edit_steps = state.post_edit_steps.index_select(0, order).clone()
            state.active_forward_counts = state.active_forward_counts.index_select(
                0, order
            ).clone()
            state.already_dead_mask = state.already_dead_mask.index_select(
                0, order
            ).clone()
            if state.iter_in_block is not None:
                state.iter_in_block = state.iter_in_block.index_select(0, order).clone()
            state.logical_order = state.logical_order.index_select(0, order).clone()

    @staticmethod
    def _reorder_fused(
        forward_batch: ForwardBatch,
        state: BlockIterState,
        order: torch.Tensor,
    ) -> None:
        # Reorder all forward_batch per-slot tensors with one stacked select.
        device = order.device
        i32_handles = [
            ("req_pool_indices", forward_batch.req_pool_indices),
            ("seq_lens", forward_batch.seq_lens),
            ("orig_seq_lens", forward_batch.orig_seq_lens),
            ("extend_prefix_lens", forward_batch.extend_prefix_lens),
            ("extend_seq_lens", forward_batch.extend_seq_lens),
        ]
        valid = [(n, t) for (n, t) in i32_handles if t is not None]
        if valid:
            stacked = torch.stack(
                [t.to(torch.int64) for (_, t) in valid], dim=0
            ).index_select(1, order)
            for j, (name, t) in enumerate(valid):
                t.copy_(stacked[j].to(t.dtype))
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_cpu.copy_(
                forward_batch.seq_lens_cpu.index_select(0, order.detach().cpu())
            )

    @staticmethod
    def _refresh_seq_lens_sum(forward_batch: ForwardBatch) -> None:
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum().item())
        else:
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens.sum().item())

    def _request_finishes_in_this_block_mask(
        self, state: BlockIterState
    ) -> torch.Tensor:
        if state.reqs is None or not state.skip_supported_for_batch:
            return torch.zeros(state.batch_size, dtype=torch.bool, device=state.device)
        values = []
        for i, req in enumerate(state.reqs):
            output_len = len(getattr(req, "output_ids", []) or [])
            sampling_params = getattr(req, "sampling_params", None)
            max_new_tokens = getattr(sampling_params, "max_new_tokens", None)
            if max_new_tokens is None:
                values.append(False)
                continue
            block_tokens = self.block_size - int(state.start_list[i])
            values.append(output_len + block_tokens >= int(max_new_tokens))
        return torch.tensor(values, dtype=torch.bool, device=state.device)

    def _restore_seq_lens(
        self, forward_batch: ForwardBatch, state: BlockIterState
    ) -> None:
        restore_order = torch.argsort(state.logical_order)
        self._reorder_slots(forward_batch, state, restore_order)
        forward_batch.seq_lens.copy_(state.original_seq_lens)
        if (
            state.original_seq_lens_cpu is not None
            and forward_batch.seq_lens_cpu is not None
        ):
            forward_batch.seq_lens_cpu.copy_(state.original_seq_lens_cpu)
        forward_batch.seq_lens_sum = state.original_seq_lens_sum

    # ------------------------------------------------------------------
    # Block setup -- once per `run()` call
    # ------------------------------------------------------------------
    def _prepare_block_state(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> BlockIterState:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        start_list = []
        prompt_masks = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            prompt_mask = block_input_ids != self.mask_id
            prompt_masks.append(prompt_mask)
            start_list.append(prompt_mask.sum().item())

        # Attach per-request continuation state while resetting block-local
        # counters for the new diffusion block.
        reqs = getattr(forward_batch, "reqs", None)
        algo_states: list[RequestAlgoState] = []
        for i in range(batch_size):
            req = reqs[i] if (reqs is not None and i < len(reqs)) else None
            if req is None:
                algo_states.append(RequestAlgoState())
                continue
            s = getattr(req, "dllm_algo_state", None)
            if not isinstance(s, RequestAlgoState):
                s = RequestAlgoState()
                req.dllm_algo_state = s
            s.post_edit_steps = 0
            s.finished = False
            s.active_forward_count = 0
            s.already_dead = False
            s.iter_in_block = 0
            algo_states.append(s)

        post_edit_steps = torch.tensor(
            [s.post_edit_steps for s in algo_states], dtype=torch.int32, device=device
        )
        finished = torch.tensor(
            [s.finished for s in algo_states], dtype=torch.bool, device=device
        )
        active_forward_counts = torch.tensor(
            [s.active_forward_count for s in algo_states],
            dtype=torch.int32,
            device=device,
        )
        already_dead_mask = torch.tensor(
            [s.already_dead for s in algo_states], dtype=torch.bool, device=device
        )
        last_changed_slots = torch.zeros(batch_size, dtype=torch.bool, device=device)
        iter_in_block = torch.tensor(
            [s.iter_in_block for s in algo_states], dtype=torch.int32, device=device
        )
        logical_order = torch.arange(batch_size, dtype=torch.long, device=device)

        original_seq_lens = forward_batch.seq_lens.clone()
        original_seq_lens_cpu = (
            forward_batch.seq_lens_cpu.clone()
            if forward_batch.seq_lens_cpu is not None
            else None
        )
        original_seq_lens_sum = forward_batch.seq_lens_sum
        prefix_lens = getattr(forward_batch, "prefix_lens", None)
        if prefix_lens is None:
            prefix_lens = forward_batch.seq_lens - self.block_size
        prefix_lens_cpu = (
            torch.as_tensor(
                forward_batch.extend_prefix_lens_cpu,
                dtype=forward_batch.seq_lens_cpu.dtype,
                device="cpu",
            )
            if (
                forward_batch.seq_lens_cpu is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            )
            else None
        )
        if prefix_lens_cpu is None and forward_batch.seq_lens_cpu is not None:
            prefix_lens_cpu = prefix_lens.detach().to(
                device="cpu", dtype=forward_batch.seq_lens_cpu.dtype
            )

        cuda_graph_bs = set(
            getattr(model_runner.server_args, "cuda_graph_bs", []) or []
        )
        # The CUDA graph runner pads non-bucket batch sizes to the next
        # captured bucket, so shrink support only depends on the optimization
        # being enabled and the batch being large enough to benefit.
        skip_supported_for_batch = bool(
            self.enable_dead_slot_skip
            and not self._small_batch_fast_path_active(batch_size)
        )

        return BlockIterState(
            batch_size=batch_size,
            block_size=self.block_size,
            device=device,
            max_iterations=self.block_size + self.max_post_edit_steps,
            prompt_masks=prompt_masks,
            start_list=start_list,
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            original_seq_lens=original_seq_lens,
            original_seq_lens_cpu=original_seq_lens_cpu,
            original_seq_lens_sum=original_seq_lens_sum,
            cuda_graph_bs=cuda_graph_bs,
            skip_supported_for_batch=skip_supported_for_batch,
            post_edit_steps=post_edit_steps,
            finished=finished,
            active_forward_counts=active_forward_counts,
            already_dead_mask=already_dead_mask,
            last_changed_slots=last_changed_slots,
            iter_in_block=iter_in_block,
            logical_order=logical_order,
            reqs=reqs,
            algo_states=algo_states,
        )

    # ------------------------------------------------------------------
    # One iteration of the denoise loop. Returns True if the block is
    # done (finished.all()) and the run() loop should break.
    # ------------------------------------------------------------------
    def _step_update_vectorized(
        self,
        forward_batch: ForwardBatch,
        state: BlockIterState,
        *,
        batch_size: int,
        preadvance: bool,
    ) -> None:
        block_size = self.block_size
        flat_n = batch_size * block_size
        logits = None
        if getattr(state.logits_output, "full_logits", None) is not None:
            logits = state.logits_output.full_logits[:flat_n].view(
                batch_size, block_size, -1
            )
        input_ids = forward_batch.input_ids[: batch_size * block_size].view(
            batch_size, block_size
        )

        process_slot = ~state.finished[:batch_size]
        if not bool(process_slot.any().item()):
            return

        if logits is not None and self.penalty_lambda > 0:
            prev_ids = input_ids[:, :-1]
            logits[:, 1:, :].scatter_(
                2, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
            )

        if logits is not None:
            max_logits, token_ids = torch.max(logits, dim=-1)
            log_denom = torch.logsumexp(logits, dim=-1)
        else:
            token_ids = state.logits_output.dllm_argmax_token_ids[:flat_n].view(
                batch_size, block_size
            )
            max_logits = state.logits_output.dllm_max_logits[:flat_n].view(
                batch_size, block_size
            )
            log_denom = state.logits_output.dllm_logsumexp[:flat_n].view(
                batch_size, block_size
            )
        confidence = torch.exp(max_logits.float() - log_denom.float())

        mask_index = input_ids == self.mask_id
        has_mask = mask_index.any(dim=1)
        prompt_mask = torch.stack(state.prompt_masks[:batch_size], dim=0)

        mask_confidence = torch.where(mask_index, confidence, -torch.inf)
        mask_transfer = mask_confidence > self.threshold
        forced_update = process_slot & has_mask & ~mask_transfer.any(dim=1)
        if bool(forced_update.any().item()):
            force_idx = torch.argmax(mask_confidence, dim=1)
            mask_transfer[forced_update, force_idx[forced_update]] = True
        mask_transfer &= process_slot[:, None] & has_mask[:, None]

        no_mask_process = process_slot & ~has_mask
        if bool(no_mask_process.any().item()):
            state.post_edit_steps[:batch_size][no_mask_process] += 1
        post_edit_over = no_mask_process & (
            state.post_edit_steps[:batch_size] > self.max_post_edit_steps
        )

        edit_mask = ~mask_index & ~prompt_mask
        edit_transfer = (
            (confidence > self.edit_threshold) & (input_ids != token_ids) & edit_mask
        )
        edit_transfer &= process_slot[:, None] & ~post_edit_over[:, None]

        transfer = mask_transfer | edit_transfer
        changed_slot = transfer.any(dim=1)
        if state.last_changed_slots is not None:
            state.last_changed_slots[:batch_size] = changed_slot
        newly_finished = process_slot & ~post_edit_over & ~changed_slot
        state.finished[:batch_size] |= post_edit_over | newly_finished

        if not bool(changed_slot.any().item()):
            return

        input_ids[transfer] = token_ids[transfer]
        state.any_changed_in_last_step = True

    def step(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        state: BlockIterState,
    ) -> bool:
        if state.finished.all():
            return True

        # When physical shrink is active, forward_batch.batch_size may be
        # smaller than state.batch_size (which still tracks the original slot
        # count for per-slot tensor sizing). The per-slot loop processes only
        # the currently active slots (which sit at indices [0, current_bs) by
        # the reorder invariant from the yield path).
        batch_size = forward_batch.batch_size
        device = state.device

        preadvance = False
        measured = True
        state.active_forward_counts[~state.finished] += 1
        out = self._timed_forward(
            model_runner,
            forward_batch,
            prompt_masks=state.prompt_masks,
            case="",
            measured=measured,
            preadvance=preadvance,
            forward_index=0,
            active_forward_counts=state.active_forward_counts,
            targets=None,
        )
        if measured:
            state.measured_forwards_since_yield += 1
        state.logits_output = out.logits_output
        state.can_run_cuda_graph = out.can_run_graph

        state.any_changed_in_last_step = False
        if state.last_changed_slots is not None:
            state.last_changed_slots.zero_()

        if self.enable_vectorized_step:
            self._step_update_vectorized(
                forward_batch, state, batch_size=batch_size, preadvance=preadvance
            )
        else:
            for i in range(batch_size):
                if state.finished[i]:
                    continue

                block_start = i * self.block_size
                block_end = block_start + self.block_size

                curr_input_ids = forward_batch.input_ids[block_start:block_end]
                curr_logits = state.logits_output.full_logits[block_start:block_end]
                curr_prompt_mask = state.prompt_masks[i]

                if self.penalty_lambda > 0:
                    prev_ids = curr_input_ids[:-1]
                    curr_logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(curr_logits, dim=-1)
                max_logits = torch.gather(curr_logits, -1, x.unsqueeze(-1)).squeeze(-1)
                log_denom = torch.logsumexp(curr_logits, dim=-1)
                p = torch.exp(max_logits.float() - log_denom.float())

                mask_index = curr_input_ids == self.mask_id
                has_mask = mask_index.any()

                # Mask to token (M2T)
                mask_transfer_index = torch.zeros_like(mask_index)
                if has_mask:
                    confidence = torch.where(mask_index, p, -np.inf)
                    mask_transfer_index = confidence > self.threshold
                    if not mask_transfer_index.any():
                        _, select_index = torch.topk(confidence, k=1)
                        mask_transfer_index[select_index] = True
                else:
                    state.post_edit_steps[i] += 1
                    if state.post_edit_steps[i] > self.max_post_edit_steps:
                        state.finished[i] = True
                        continue

                # Token to token (T2T)
                edit_mask = ~mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )

                transfer_index = mask_transfer_index | edit_transfer_index
                if not transfer_index.any():
                    state.finished[i] = True
                    continue

                curr_input_ids[transfer_index] = x[transfer_index]
                state.any_changed_in_last_step = True
                if state.last_changed_slots is not None:
                    state.last_changed_slots[i] = True

        # Physically shrink the batch when enough slots have converged.
        #
        # Only terminal-finished slots are safe as padding: their KV cache will
        # be released after this block. A non-terminal converged slot must keep
        # this block's KV intact for the next block.
        single_shrink_guard_ok = state.shrink_state is None or self.enable_multi_shrink
        if (
            self.yield_every is not None
            and state.measured_forwards_since_yield >= self.yield_every
            and bool(
                (((state.iter_in_block + 1) < state.max_iterations) & ~state.finished)
                .any()
                .item()
            )
            and not state.finished.all()
            and single_shrink_guard_ok
            and state.skip_supported_for_batch
        ):
            terminal_mask = self._request_finishes_in_this_block_mask(state)
            all_dead = state.finished & ~state.already_dead_mask
            n_active = int((~all_dead).sum().item())
            current_bs = forward_batch.batch_size
            # If the exact active size is not captured, use the next larger
            # bucket only when terminal-finished slots can safely fill it.
            target_bs = n_active if n_active in state.cuda_graph_bs else 0
            padding_terminal_dead_slots: list[int] = []
            if (
                self.enable_bucket_padding
                and target_bs == 0
                and n_active > 0
                and n_active < current_bs
            ):
                candidates = sorted(
                    b for b in state.cuda_graph_bs if n_active <= b < current_bs
                )
                if candidates:
                    cand_bs = candidates[0]
                    deficit = cand_bs - n_active
                    terminal_dead_mask = (
                        state.finished
                        & ~state.already_dead_mask
                        & terminal_mask.to(state.device)
                    )
                    n_terminal_dead = int(terminal_dead_mask.sum().item())
                    if n_terminal_dead >= deficit:
                        td_idx = terminal_dead_mask.nonzero(as_tuple=False).flatten()
                        padding_terminal_dead_slots = (
                            td_idx[:deficit].detach().cpu().tolist()
                        )
                        target_bs = cand_bs
            if self.enable_physical_shrink and state.cuda_graph_bs:
                if target_bs and target_bs < current_bs:
                    active_idx = (~all_dead).nonzero(as_tuple=False).flatten()
                    dead_idx = all_dead.nonzero(as_tuple=False).flatten()
                    if padding_terminal_dead_slots:
                        # Active slots occupy the front; terminal-finished
                        # padding fills the rest of the captured bucket.
                        pad_set = set(padding_terminal_dead_slots)
                        pad_t = torch.tensor(
                            padding_terminal_dead_slots,
                            dtype=torch.long,
                            device=state.device,
                        )
                        dead_mask_cpu = all_dead.detach().cpu().tolist()
                        remaining_dead = [
                            i
                            for i in range(state.batch_size)
                            if dead_mask_cpu[i] and i not in pad_set
                        ]
                        remaining_dead_t = (
                            torch.tensor(
                                remaining_dead,
                                dtype=torch.long,
                                device=state.device,
                            )
                            if remaining_dead
                            else torch.empty(0, dtype=torch.long, device=state.device)
                        )
                        order = torch.cat([active_idx, pad_t, remaining_dead_t])
                    else:
                        order = torch.cat([active_idx, dead_idx])
                    self._reorder_slots(forward_batch, state, order)
                    self._refresh_seq_lens_sum(forward_batch)
                    state.already_dead_mask |= all_dead
                    self._shrink_forward_batch(forward_batch, state, target_bs)
            state.measured_forwards_since_yield = 0

        state.iter_in_block[~state.finished] += 1
        state.iteration_idx += 1
        return bool(state.finished.all())

    def _shrink_forward_batch(
        self, forward_batch: ForwardBatch, state: BlockIterState, target_bs: int
    ) -> None:
        """Slice forward_batch tensors to target_bs slots in-place.

        Pre-condition: callers have already reordered so the first
        `target_bs` slot indices are still-active (or padding with dead slots
        whose seq_lens are set to prefix_lens, so attention skips them).
        Sliced views share storage with state.shrink_state.* so any in-place
        mutations of token IDs / KV cache during forward are visible after
        restore.
        """
        new_state = ShrinkState(
            input_ids=forward_batch.input_ids,
            positions=forward_batch.positions,
            out_cache_loc=forward_batch.out_cache_loc,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            orig_seq_lens=forward_batch.orig_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_seq_lens=forward_batch.extend_seq_lens,
            batch_size=forward_batch.batch_size,
            seq_lens_sum=forward_batch.seq_lens_sum,
        )
        # Preserve older views so repeated shrinks can restore the original
        # full batch shape at block finalization.
        if self.enable_multi_shrink and state.shrink_state is not None:
            state.shrink_stack.append(state.shrink_state)
        state.shrink_state = new_state
        flat_n = target_bs * self.block_size
        forward_batch.input_ids = forward_batch.input_ids[:flat_n]
        forward_batch.positions = forward_batch.positions[:flat_n]
        forward_batch.out_cache_loc = forward_batch.out_cache_loc[:flat_n]
        forward_batch.req_pool_indices = forward_batch.req_pool_indices[:target_bs]
        forward_batch.seq_lens = forward_batch.seq_lens[:target_bs]
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu[:target_bs]
        forward_batch.orig_seq_lens = forward_batch.orig_seq_lens[:target_bs]
        forward_batch.extend_prefix_lens = forward_batch.extend_prefix_lens[:target_bs]
        forward_batch.extend_seq_lens = forward_batch.extend_seq_lens[:target_bs]
        forward_batch.batch_size = target_bs
        forward_batch.seq_lens_sum = int(forward_batch.seq_lens.sum().item())

    @staticmethod
    def _restore_forward_batch(
        forward_batch: ForwardBatch, state: BlockIterState
    ) -> None:
        """Restore full-size forward_batch tensors from saved shrink_state.

        With repeated shrinks, the oldest saved view owns the original
        full-size handles. Intermediate saves are narrower views into the same
        storage.
        """
        if state.shrink_state is None and not state.shrink_stack:
            return
        # Pick the oldest save; for single-shrink this is just shrink_state.
        s = state.shrink_stack[0] if state.shrink_stack else state.shrink_state
        forward_batch.input_ids = s.input_ids
        forward_batch.positions = s.positions
        forward_batch.out_cache_loc = s.out_cache_loc
        forward_batch.req_pool_indices = s.req_pool_indices
        forward_batch.seq_lens = s.seq_lens
        forward_batch.seq_lens_cpu = s.seq_lens_cpu
        forward_batch.orig_seq_lens = s.orig_seq_lens
        forward_batch.extend_prefix_lens = s.extend_prefix_lens
        forward_batch.extend_seq_lens = s.extend_seq_lens
        forward_batch.batch_size = s.batch_size
        forward_batch.seq_lens_sum = s.seq_lens_sum
        state.shrink_state = None
        state.shrink_stack = []

    @staticmethod
    def _persist_algo_states(state: BlockIterState) -> None:
        """Write block-final per-slot tensors back to RequestAlgoState."""
        if not state.algo_states:
            return
        post_edit_cpu = state.post_edit_steps.detach().cpu().tolist()
        finished_cpu = state.finished.detach().cpu().tolist()
        active_cpu = state.active_forward_counts.detach().cpu().tolist()
        dead_cpu = state.already_dead_mask.detach().cpu().tolist()
        iter_cpu = (
            state.iter_in_block.detach().cpu().tolist()
            if state.iter_in_block is not None
            else [0] * len(state.algo_states)
        )
        for i, s in enumerate(state.algo_states):
            s.post_edit_steps = int(post_edit_cpu[i])
            s.finished = bool(finished_cpu[i])
            s.active_forward_count = int(active_cpu[i])
            s.already_dead = bool(dead_cpu[i])
            s.iter_in_block = int(iter_cpu[i])
            if s.finished:
                s.blocks_completed += 1

    # ------------------------------------------------------------------
    # Final commit-forward + return
    # ------------------------------------------------------------------
    def _finalize_block(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        state: BlockIterState,
    ):
        # Restore the full batch before the commit forward so downstream code
        # sees the original slot layout and complete KV state.
        self._restore_forward_batch(forward_batch, state)
        try:
            if state.any_changed_in_last_step:
                if state.last_changed_slots is not None:
                    state.active_forward_counts[state.last_changed_slots] += 1
                else:
                    state.active_forward_counts[~state.finished] += 1
                out = self._timed_forward(
                    model_runner,
                    forward_batch,
                    prompt_masks=state.prompt_masks,
                    case="",
                    measured=True,
                    preadvance=False,
                    forward_index=0,
                    active_forward_counts=state.active_forward_counts,
                    targets=None,
                )
                state.logits_output = out.logits_output
                state.can_run_cuda_graph = out.can_run_graph
        finally:
            self._restore_seq_lens(forward_batch, state)
        self._persist_algo_states(state)

        batch_size = state.batch_size
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, state.start_list[i] :] for i in range(batch_size)
        ]

        return state.logits_output, next_token_ids_list, state.can_run_cuda_graph

    # ------------------------------------------------------------------
    # Public entry point: drive one diffusion block through iterative updates,
    # optional physical shrink, and final commit.
    # ------------------------------------------------------------------
    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        if self._can_use_reference_small_batch_path(forward_batch):
            return self._run_reference_block(model_runner, forward_batch)

        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        # Early-exit path for blocks that contain no mask tokens.
        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            active_forward_counts = torch.zeros(
                batch_size, dtype=torch.int32, device=device
            )
            active_forward_counts += 1
            out = self._timed_forward(
                model_runner,
                forward_batch,
                case="",
                measured=True,
                preadvance=False,
                forward_index=0,
                active_forward_counts=active_forward_counts,
                targets=[0] * batch_size,
            )
            return out.logits_output, [], out.can_run_graph

        # Normal block path: prepare state, loop step(), finalize.
        state = self._prepare_block_state(model_runner, forward_batch)
        # Continue while any non-finished slot still has remaining iterations.
        while bool(
            ((state.iter_in_block < state.max_iterations) & ~state.finished)
            .any()
            .item()
        ):
            if self.step(model_runner, forward_batch, state):
                break
        return self._finalize_block(model_runner, forward_batch, state)


Algorithm = JointThreshold
