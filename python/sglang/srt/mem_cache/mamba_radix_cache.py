from __future__ import annotations

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
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import heapq
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    get_child_key,
)
from sglang.srt.utils.catchup_timing import is_lfc_enabled

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        # LFC factors for reconstructing SSM state when mamba_value is tombstoned
        # Dict of {layer_id: (k, v, g, beta)} or None
        self.lfc_factors: Optional[dict] = None
        # CPU-offloaded LFC factors (pinned memory), used when GPU budget is exceeded
        self.lfc_host_factors: Optional[dict] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        # store the host indices of KV cache
        self.host_value = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "MambaRadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = tree_cache.mamba_lru_list_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = tree_cache.full_lru_list_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
            logger.error(msg)
            raise Exception(msg)


class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, TokenToKVPoolAllocator)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator

        assert (
            params.page_size == 1
        ), "Only support page_size=1 in mamba radix cache now."
        self.page_size = params.page_size
        self.disable = params.disable

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.key_match_fn = _key_match_page_size1
        self.get_child_key_fn = get_child_key
        self.reset()

    ##### Public API #####

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)
        # Track nodes with deferred LFC promotions (node id(obj) -> node ref)
        self._pending_promo_nodes: set = set()

        # LFC memory budget management (only active when LFC is enabled)
        if is_lfc_enabled():
            from sglang.srt.environ import envs

            budget_gb = envs.SGLANG_LFC_MEMORY_BUDGET_GB.value
            self.lfc_memory_budget_bytes = int(budget_gb * 1024**3)
            self.lfc_current_memory_bytes = 0
            # CPU pinned memory budget for offloaded factors
            host_budget_gb = envs.SGLANG_LFC_HOST_MEMORY_BUDGET_GB.value
            self.lfc_host_memory_budget_bytes = int(host_budget_gb * 1024**3)
            self.lfc_host_current_memory_bytes = 0
            # Min-heap of (factor_value, node_id, node_ref) for eviction
            self.lfc_factor_heap: list = []
            # Map node_id -> bool for quick heap validity check (lazy deletion)
            self.lfc_factor_nodes: dict = {}

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            key: A RadixKey contains token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        cow_mamba: bool = kwargs.get("cow_mamba", False)
        req: Req = kwargs.get("req", None)

        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, last_node, attn_match_len = self._match_prefix_helper(key)

        # Instrumentation: log match_prefix decisions
        if cow_mamba and req is not None:
            matched_tokens = sum(len(v) for v in value) if value else 0
            has_mamba = last_node.mamba_value is not None
            has_lfc = getattr(last_node, 'lfc_factors', None) is not None
            lfc_on = is_lfc_enabled()
            node_key_len = len(last_node.key) if hasattr(last_node, 'key') else -1
            n_children = len(last_node.children) if hasattr(last_node, 'children') else -1
            is_leaf = (n_children == 0)
            if matched_tokens > 0 or attn_match_len > 0:
                logger.warning(
                    f"[LFC-TRACE] match_prefix: matched={matched_tokens}, "
                    f"attn_match={attn_match_len}, "
                    f"mamba={'Y' if has_mamba else 'N'}, "
                    f"lfc={'Y' if has_lfc else 'N'}, "
                    f"lfc_on={lfc_on}, "
                    f"nid={last_node.id}, "
                    f"key_len={node_key_len}, children={n_children}, "
                    f"leaf={is_leaf}"
                )

        # copy mamba state to req local space if cow is true
        # First, realize any deferred LFC promotion on this node
        if cow_mamba and last_node.mamba_value is None:
            self._realize_pending_promotion(last_node)

        if cow_mamba and last_node.mamba_value is not None:
            req._got_mamba_cow = True  # Flag: state came from cached snapshot
            # for reqs without mamba cache
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                # try to alloc again, protect last_node from eviction
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict_mamba(1)
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                src_index = last_node.mamba_value
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
                req.mamba_pool_idx = dst_index[0]
            else:
                src_index = last_node.mamba_value
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)

        elif cow_mamba and is_lfc_enabled() and (last_node.lfc_factors is not None or last_node.lfc_host_factors is not None):
            req._got_mamba_cow = True  # Flag: state came from LFC reconstruction
            # LFC path: reconstruct SSM state from ancestor + factor chain
            # 1. Walk up to nearest ancestor with full SSM state (or root)
            ancestor = last_node.parent
            factor_chain = [last_node]
            while ancestor and ancestor != self.root_node and ancestor.mamba_value is None:
                if ancestor.lfc_factors is not None or ancestor.lfc_host_factors is not None:
                    factor_chain.append(ancestor)
                ancestor = ancestor.parent

            # ancestor now has mamba_value or is root_node
            if ancestor is None:
                ancestor = self.root_node
            assert ancestor.mamba_value is not None or ancestor == self.root_node, \
                "Ancestor must have SSM state or be root"

            logger.debug(
                f"[LFC] Reconstruction triggered: "
                f"factor_chain_len={len(factor_chain)}, "
                f"ancestor_id={ancestor.id}"
            )

            # 2. Allocate or get dst_index for req
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict_mamba(1)
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)

            # 3. Copy ancestor's SSM state as starting point
            if ancestor.mamba_value is not None:
                self.req_to_token_pool.mamba_pool.copy_from(
                    ancestor.mamba_value, dst_index
                )
            else:
                # Ancestor is root with no mamba_value: explicitly zero the slot.
                # Recycled pool slots may contain stale data from previous requests.
                pool = self.req_to_token_pool.mamba_pool
                for i in range(len(pool.mamba_cache.conv)):
                    pool.mamba_cache.conv[i][:, dst_index] = 0
                pool.mamba_cache.temporal[:, dst_index] = 0

            # 4. Store reconstruction factors on request for use during forward
            # The reconstruction will happen in forward_extend where we have access to
            # the layer-specific state. We pass the factor chain to the request.
            # Note: Use lfc_reconstruction_factors (tree -> forward) not pending_lfc_factors (forward -> tree)
            # Pre-merge: collect factor chunks per layer, then concat into single tensors
            # so that each layer only needs one kernel call during reconstruction.
            layer_chunks = {}
            _host_loaded_nodes = []
            for node in reversed(factor_chain):
                if node.lfc_factors is None and node.lfc_host_factors is not None:
                    self._lfc_load_from_host(node)
                    _host_loaded_nodes.append(node)
                if node.lfc_factors is not None:
                    for layer_id, factors in node.lfc_factors.items():
                        if layer_id not in layer_chunks:
                            layer_chunks[layer_id] = []
                        layer_chunks[layer_id].append(factors)

            req.lfc_reconstruction_factors = {}
            for layer_id, chunk_list in layer_chunks.items():
                if len(chunk_list) == 1:
                    if _host_loaded_nodes:
                        # Clone to decouple from ephemeral GPU tensors
                        req.lfc_reconstruction_factors[layer_id] = tuple(
                            f.clone() for f in chunk_list[0]
                        )
                    else:
                        req.lfc_reconstruction_factors[layer_id] = chunk_list[0]
                else:
                    # Single allocation + indexed copy (avoids torch.cat's per-call
                    # dispatch overhead and internal temporary allocation)
                    num_factors = len(chunk_list[0])
                    result = []
                    for j in range(num_factors):
                        total_tokens = sum(c[j].shape[0] for c in chunk_list)
                        ref = chunk_list[0][j]
                        buf = torch.empty(
                            (total_tokens, *ref.shape[1:]),
                            dtype=ref.dtype, device=ref.device,
                        )
                        offset = 0
                        for c in chunk_list:
                            n = c[j].shape[0]
                            buf[offset:offset + n] = c[j]
                            offset += n
                        result.append(buf)
                    req.lfc_reconstruction_factors[layer_id] = tuple(result)

            # Release ephemeral GPU copies from host-loaded nodes
            for node in _host_loaded_nodes:
                node.lfc_factors = None

            # 5. Allocate promotion slot to cache reconstruction result on prefix node.
            # After forward, the reconstructed state will be assigned to last_node.mamba_value
            # so future requests can use cheap CoW instead of expensive reconstruction.
            if last_node.mamba_value is None:
                promo_slot = self.req_to_token_pool.mamba_pool.alloc(1)
                if promo_slot is not None:
                    # Copy ancestor's state as starting point for the promo slot
                    if ancestor.mamba_value is not None:
                        self.req_to_token_pool.mamba_pool.copy_from(
                            ancestor.mamba_value, promo_slot
                        )
                    else:
                        pool = self.req_to_token_pool.mamba_pool
                        for i in range(len(pool.mamba_cache.conv)):
                            pool.mamba_cache.conv[i][:, promo_slot] = 0
                        pool.mamba_cache.temporal[:, promo_slot] = 0
                    req._lfc_promo_slot = promo_slot[0]
                    req._lfc_promo_node = last_node
                    logger.debug(
                        f"[LFC] Allocated promo slot {promo_slot[0].item()} "
                        f"for node {last_node.id}"
                    )

        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            mamba_branching_seqlen=attn_match_len,
        )

    def insert(
        self, key: RadixKey, value=None, mamba_value=None, lfc_factors=None
    ) -> Tuple[int, bool]:
        if self.disable:
            return 0

        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        return self._insert_helper(
            self.root_node, key, value, mamba_value, lfc_factors
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        page_aligned_len = len(kv_indices)
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        # insert the token_ids and kv_indices into the radix tree
        # Note: the insert function already frees the overlapped kv_indices
        mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()

        if is_insert:
            # Get pending LFC factors from request if available (LFC-only)
            lfc_factors = (
                getattr(req, "pending_lfc_factors", None)
                if is_lfc_enabled()
                else None
            )
            new_prefix_len, mamba_exist = self.insert(
                RadixKey(token_ids[:page_aligned_len], req.extra_key),
                page_aligned_kv_indices,
                mamba_value,
                lfc_factors=lfc_factors,
            )
            self.token_to_kv_pool_allocator.free(
                kv_indices[len(req.prefix_indices) : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[len(req.prefix_indices) : page_aligned_len]
            )
            mamba_exist = True

        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req.req_pool_idx, free_mamba_cache=mamba_exist)
            self.dec_lock_ref(req.last_node)
        else:  # for abort case
            self.req_to_token_pool.mamba_pool.free(mamba_value)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        page_aligned_len = len(kv_indices)
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
        page_aligned_token_ids = token_ids[:page_aligned_len]

        mamba_value = self.req_to_token_pool.get_mamba_indices(
            req.req_pool_idx
        ).unsqueeze(-1)
        # radix tree mamba value is forked from req space
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)

        # if alloc mamba cache failed, do evict and alloc again
        if mamba_value_forked is None:
            self.evict_mamba(1)
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"
        # Get pending LFC factors from request if available (LFC-only)
        lfc_factors = (
            getattr(req, "pending_lfc_factors", None)
            if is_lfc_enabled()
            else None
        )
        new_prefix_len, mamba_exist = self.insert(
            RadixKey(page_aligned_token_ids, req.extra_key),
            page_aligned_kv_indices,
            mamba_value_forked,
            lfc_factors=lfc_factors,
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )
        # there is a mamba cache in radix cache, release it
        if mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            RadixKey(page_aligned_token_ids, req.extra_key)
        )
        (new_indices, new_last_node) = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_forked)

        assert len(req.prefix_indices) <= len(
            new_indices
        ), f"{req.prefix_indices=}, {new_indices=}"
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        # 1. a leaf node, free full tokens and mamba (if present)
        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)
        if x.mamba_value is not None:
            self.req_to_token_pool.mamba_pool.free(x.mamba_value)
            mamba_num_evicted = len(x.mamba_value)
        else:
            # Factor-only leaf node (proactively tombstoned): no mamba to free
            mamba_num_evicted = 0

        # Free pending promotion slot if present (deferred snapshot not yet realized)
        pending_promo = getattr(x, '_lfc_pending_mamba_value', None)
        if pending_promo is not None:
            self.req_to_token_pool.mamba_pool.free(pending_promo)
            self._pending_promo_nodes.discard(id(x))
            del x._lfc_pending_mamba_value

        # 2. get the next node, update the lru lists
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        if x.mamba_value is not None:
            self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node
        self._delete_leaf(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def promote_lfc_snapshots(self, reqs):
        """Defer promotion of reconstruction snapshots to prefix tree nodes.

        After reconstruction, the promo slot contains the reconstructed SSM state.
        We store it as a pending promotion on the node. The actual assignment to
        mamba_value happens in the next match_prefix call, where lock_refs are
        properly managed.
        """
        for req in reqs:
            promo_slot = getattr(req, '_lfc_promo_slot', None)
            promo_node = getattr(req, '_lfc_promo_node', None)
            if promo_slot is None or promo_node is None:
                continue
            # Only promote if node still lacks mamba_value and no pending promotion
            if promo_node.mamba_value is None and not hasattr(promo_node, '_lfc_pending_mamba_value'):
                promo_node._lfc_pending_mamba_value = promo_slot.unsqueeze(0)
                self._pending_promo_nodes.add(id(promo_node))
                logger.debug(
                    f"[LFC] Deferred promotion to node {promo_node.id}"
                )
            else:
                # Node already has snapshot or pending promotion, free promo slot
                self.req_to_token_pool.mamba_pool.free(promo_slot.unsqueeze(0))
            req._lfc_promo_slot = None
            req._lfc_promo_node = None

    def _realize_pending_promotion(self, node: TreeNode):
        """Realize a deferred LFC snapshot promotion on a node.

        Called from match_prefix before the CoW check, so that lock_refs
        are properly managed (inc_lock_ref will see mamba_value).
        """
        pending = getattr(node, '_lfc_pending_mamba_value', None)
        if pending is not None:
            node.mamba_value = pending
            del node._lfc_pending_mamba_value
            self._pending_promo_nodes.discard(id(node))
            self.mamba_evictable_size_ += len(pending)
            node.last_access_time = get_last_access_time()
            self.mamba_lru_list.insert_mru(node)
            # Also reposition in full_lru_list since last_access_time changed
            if node.id in self.full_lru_list.cache:
                self.full_lru_list._remove_node(node)
                self.full_lru_list._add_node(node)
            logger.debug(
                f"[LFC] Realized promotion on node {node.id}"
            )

    def realize_all_pending_promotions(self):
        """Realize all deferred LFC promotions. Safe to call during idle time
        when no requests hold lock_refs."""
        # Walk all nodes to find pending promotions (set tracks by id())
        if not self._pending_promo_nodes:
            return
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if id(node) in self._pending_promo_nodes:
                self._realize_pending_promotion(node)
            stack.extend(node.children.values())

    def evict_mamba(self, mamba_num: int) -> None:
        if self.disable or mamba_num <= 0:
            return
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

    def evict(self, full_num_tokens: int) -> None:
        if self.disable or full_num_tokens <= 0:
            return

        full_num_evicted = 0
        # get the least recently used leaf node that is not locked
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

    def inc_lock_ref(self, node: TreeNode) -> Optional[int]:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return None

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return None

    def dec_lock_ref(self, node: TreeNode):
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return

        if node.mamba_value is not None:
            if node.mamba_lock_ref > 0:
                if node.mamba_lock_ref == 1:
                    self.mamba_evictable_size_ += len(node.mamba_value)
                    self.mamba_protected_size_ -= len(node.mamba_value)
                node.mamba_lock_ref -= 1
            # else: mamba_value was acquired after this request locked the node
            #       (e.g., via deferred snapshot promotion). Skip decrement.

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

    def sanity_check(self):
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    # Note: this is expensive, only use for debug
    def full_lru_list_evictable_size(self) -> int:
        return self.full_lru_list.sanity_check_evictable_size()

    # Note: this is expensive, only use for debug
    def mamba_lru_list_evictable_size(self) -> int:
        return self.mamba_lru_list.sanity_check_evictable_size()

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    @staticmethod
    def _estimate_factor_memory(lfc_factors: dict) -> int:
        """Estimate GPU memory usage of LFC factors in bytes."""
        total = 0
        for layer_id, factors in lfc_factors.items():
            for t in factors:
                total += t.numel() * t.element_size()
        return total

    @staticmethod
    def _compute_factor_value(node: TreeNode) -> float:
        """Compute value score for a node's LFC factors.

        Higher value = more useful to keep.
        Formula: hit_count * num_children / len(node.key)
        Internal nodes (with children) get massive bonus to prevent chain breaks.
        """
        key_len = len(node.key) if node.key else 1
        num_children = len(node.children)
        base_value = node.hit_count * max(num_children, 1) / key_len
        # Internal nodes are chain-critical — near-infinite eviction protection
        if num_children > 0:
            base_value += 1e6
        return base_value

    def _lfc_try_store_factors(self, node: TreeNode, lfc_factors: dict, gap_fill: bool = False) -> bool:
        """Try to store LFC factors on a node, respecting memory budget.

        Returns True if factors were stored, False if rejected.
        Only called when LFC is enabled.

        Args:
            gap_fill: If True, skip break-even check. Gap filling compares
                factors vs NO caching (tombstoned node), so even expensive
                factors are worthwhile. Memory budget is the only constraint.

        Mutual exclusion: factors are NOT stored if the node already has a valid
        SSM state snapshot (mamba_value), since the snapshot already encodes all
        information from position 0 to the node's position.
        """
        if lfc_factors is None:
            return False

        # Mutual exclusion: snapshot already covers this node → factors redundant
        if node.mamba_value is not None:
            return False

        # Note: break-even check removed. Factors come from a separate memory
        # budget (SGLANG_LFC_MEMORY_BUDGET_GB), independent of the mamba pool.
        # Even for long nodes, factors are valuable when mamba pool snapshots
        # are evicted under pressure. The budget cap below is the only constraint.

        factor_bytes = self._estimate_factor_memory(lfc_factors)

        # Budget not exceeded: store directly
        if self.lfc_current_memory_bytes + factor_bytes <= self.lfc_memory_budget_bytes:
            node.lfc_factors = lfc_factors
            self.lfc_current_memory_bytes += factor_bytes
            factor_value = self._compute_factor_value(node)
            heapq.heappush(
                self.lfc_factor_heap, (factor_value, node.id, node)
            )
            self.lfc_factor_nodes[node.id] = True
            return True

        # Budget exceeded: try to evict lowest-value factors
        new_value = self._compute_factor_value(node)

        # Evict until we have room or no more low-value nodes
        while (
            self.lfc_current_memory_bytes + factor_bytes > self.lfc_memory_budget_bytes
            and self.lfc_factor_heap
        ):
            min_value, min_node_id, min_node = self.lfc_factor_heap[0]

            # Lazy deletion: skip nodes already removed from tracking
            if min_node_id not in self.lfc_factor_nodes:
                heapq.heappop(self.lfc_factor_heap)
                continue

            # Skip internal nodes — their factors are chain-critical
            if len(min_node.children) > 0:
                heapq.heappop(self.lfc_factor_heap)
                continue

            # New node isn't more valuable than what we'd evict
            if new_value <= min_value:
                break

            # Evict the lowest-value node's factors
            heapq.heappop(self.lfc_factor_heap)
            del self.lfc_factor_nodes[min_node_id]
            if min_node.lfc_factors is not None:
                evicted_bytes = self._estimate_factor_memory(min_node.lfc_factors)
                self.lfc_current_memory_bytes -= evicted_bytes
                self._lfc_offload_to_host(min_node)
                min_node.lfc_factors = None

        # Check if we freed enough
        if self.lfc_current_memory_bytes + factor_bytes <= self.lfc_memory_budget_bytes:
            node.lfc_factors = lfc_factors
            self.lfc_current_memory_bytes += factor_bytes
            heapq.heappush(
                self.lfc_factor_heap, (new_value, node.id, node)
            )
            self.lfc_factor_nodes[node.id] = True
            return True

        # Last resort: force-store for internal nodes to prevent chain gaps
        if len(node.children) > 0:
            node.lfc_factors = lfc_factors
            self.lfc_current_memory_bytes += factor_bytes
            factor_value = self._compute_factor_value(node)
            heapq.heappush(self.lfc_factor_heap, (factor_value, node.id, node))
            self.lfc_factor_nodes[node.id] = True
            return True

        return False

    def _lfc_remove_factors(self, node: TreeNode):
        """Remove LFC factors from a node and update memory tracking.

        Only called when LFC is enabled.
        """
        if node.lfc_factors is not None:
            factor_bytes = self._estimate_factor_memory(node.lfc_factors)
            self.lfc_current_memory_bytes -= factor_bytes
            node.lfc_factors = None
        self._lfc_remove_host_factors(node)
        # Lazy removal from heap - just remove from tracking dict
        if node.id in self.lfc_factor_nodes:
            del self.lfc_factor_nodes[node.id]

    def _lfc_offload_to_host(self, node: TreeNode):
        """Move GPU factors to CPU pinned memory. Called during eviction."""
        if node.lfc_factors is None:
            return
        factor_bytes = self._estimate_factor_memory(node.lfc_factors)
        if self.lfc_host_current_memory_bytes + factor_bytes > self.lfc_host_memory_budget_bytes:
            logger.warning(
                f"[LFC-OFFLOAD] No host room for node {node.id} "
                f"({factor_bytes/1024/1024:.1f}MB), discarding. "
                f"Host used: {self.lfc_host_current_memory_bytes/1024/1024:.1f}MB / "
                f"{self.lfc_host_memory_budget_bytes/1024/1024:.1f}MB"
            )
            return  # No host room — discard
        host_factors = {}
        for layer_id, factor_tuple in node.lfc_factors.items():
            host_factors[layer_id] = tuple(
                torch.empty_like(t, device="cpu", pin_memory=True).copy_(t)
                for t in factor_tuple
            )
        node.lfc_host_factors = host_factors
        self.lfc_host_current_memory_bytes += factor_bytes
        logger.warning(
            f"[LFC-OFFLOAD] Offloaded node {node.id} to host "
            f"({factor_bytes/1024/1024:.1f}MB, {len(node.lfc_factors)} layers). "
            f"GPU: {self.lfc_current_memory_bytes/1024/1024:.1f}MB / "
            f"{self.lfc_memory_budget_bytes/1024/1024:.1f}MB, "
            f"Host: {self.lfc_host_current_memory_bytes/1024/1024:.1f}MB"
        )

    def _lfc_load_from_host(self, node: TreeNode) -> bool:
        """Load host factors to GPU temporarily. Returns True if loaded."""
        if node.lfc_host_factors is None:
            return False
        gpu_factors = {}
        for layer_id, factor_tuple in node.lfc_host_factors.items():
            gpu_factors[layer_id] = tuple(
                t.to(self.device, non_blocking=False) for t in factor_tuple
            )
        node.lfc_factors = gpu_factors
        factor_bytes = self._estimate_factor_memory(gpu_factors)
        logger.warning(
            f"[LFC-LOAD] Loaded node {node.id} from host "
            f"({factor_bytes/1024/1024:.1f}MB, {len(gpu_factors)} layers)"
        )
        return True

    def _lfc_remove_host_factors(self, node: TreeNode):
        """Remove host factors and update tracking."""
        if node.lfc_host_factors is not None:
            factor_bytes = self._estimate_factor_memory(node.lfc_host_factors)
            self.lfc_host_current_memory_bytes -= factor_bytes
            node.lfc_host_factors = None

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.

        LFC Enhancement: When LFC is enabled, tracks factor chain validity to detect
        gap nodes. A gap node (no mamba_value, no lfc_factors) breaks the factor chain,
        preventing LFC reconstruction for nodes beyond the gap.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value = []
        best_value_len = 0
        best_last_node = node
        # LFC: Track if factor chain is unbroken from last mamba_value node
        lfc_enabled = is_lfc_enabled()
        lfc_chain_valid = True

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            # LFC Enhancement: Check chain validity before accepting LFC-only nodes
            if node.mamba_value is not None:
                # Has full SSM state: unconditionally accept and reset chain
                best_value_len = len(value)
                best_last_node = node
                if lfc_enabled:
                    lfc_chain_valid = True  # Reset chain validity
            elif lfc_enabled and (node.lfc_factors is not None or node.lfc_host_factors is not None) and lfc_chain_valid:
                # Has LFC factors (GPU or host) AND chain is still valid: accept
                best_value_len = len(value)
                best_last_node = node
            elif (
                lfc_enabled
                and node.lfc_factors is None
                and node.lfc_host_factors is None
                and node.mamba_value is None
                and node != self.root_node
            ):
                # Gap node (no mamba_value, no lfc_factors): chain breaks
                lfc_chain_valid = False
                logger.debug(f"[LFC-GAP] Chain broken at node {node.id}, key_len={len(node.key)}, children={len(node.children)}")
                # Do NOT update best_last_node - stop accepting LFC nodes

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        # LFC Enhancement: Apply same chain validity logic for final node
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node
        elif lfc_enabled and (node.lfc_factors is not None or node.lfc_host_factors is not None) and lfc_chain_valid:
            best_value_len = len(value)
            best_last_node = node

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        node_update = best_last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # Total key-matched tokens (attention-reusable) regardless of mamba_value
        attn_match_len = sum(len(v) for v in value)
        return value[:best_value_len], best_last_node, attn_match_len

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]

        # LFC Enhancement: Split lfc_factors if present
        # new_node gets factors for tokens [0:split_len]
        # child keeps factors for tokens [split_len:]
        if is_lfc_enabled() and child.lfc_factors is not None:
            # Remove old child from memory tracking before split
            old_bytes = self._estimate_factor_memory(child.lfc_factors)
            if child.id in self.lfc_factor_nodes:
                del self.lfc_factor_nodes[child.id]

            new_node.lfc_factors = {}
            for layer_id, (k, v, g, beta) in child.lfc_factors.items():
                # Factors are stored as [seq_len, ...], split by token index
                # new_node (prefix) gets views into the original tensors (no alloc);
                # child (suffix) gets clones so it's independent.
                # The original tensor stays alive via new_node's views until evicted.
                new_node.lfc_factors[layer_id] = (
                    k[:split_len],
                    v[:split_len],
                    g[:split_len],
                    beta[:split_len],
                )
                child.lfc_factors[layer_id] = (
                    k[split_len:].clone(),
                    v[split_len:].clone(),
                    g[split_len:].clone(),
                    beta[split_len:].clone(),
                )

            # Update memory tracking: remove old, add both new parts
            new_node_bytes = self._estimate_factor_memory(new_node.lfc_factors)
            child_bytes = self._estimate_factor_memory(child.lfc_factors)
            self.lfc_current_memory_bytes += (new_node_bytes + child_bytes - old_bytes)

            # Add both to heap
            new_node_value = self._compute_factor_value(new_node)
            heapq.heappush(self.lfc_factor_heap, (new_node_value, new_node.id, new_node))
            self.lfc_factor_nodes[new_node.id] = True

            child_value = self._compute_factor_value(child)
            heapq.heappush(self.lfc_factor_heap, (child_value, child.id, child))
            self.lfc_factor_nodes[child.id] = True

        # LFC Enhancement: Split host factors if present (CPU offloaded)
        if is_lfc_enabled() and child.lfc_host_factors is not None:
            old_host_bytes = self._estimate_factor_memory(child.lfc_host_factors)
            new_node.lfc_host_factors = {}
            for layer_id, factors in child.lfc_host_factors.items():
                new_node.lfc_host_factors[layer_id] = tuple(t[:split_len] for t in factors)
                child.lfc_host_factors[layer_id] = tuple(
                    t[split_len:].clone() for t in factors
                )
            new_host_bytes = self._estimate_factor_memory(new_node.lfc_host_factors)
            child_host_bytes = self._estimate_factor_memory(child.lfc_host_factors)
            self.lfc_host_current_memory_bytes += (new_host_bytes + child_host_bytes - old_host_bytes)

        # child time should be later than parent's time for mamba tombstone
        child.last_access_time = get_last_access_time()

        self.full_lru_list.remove_node(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        lfc_factors=None,
    ) -> Tuple[int, bool]:
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = self.get_child_key_fn(key)

        # LFC: Track factor offset for opportunistic gap filling during walk.
        # When a request re-forwards due to a tombstone gap, its factors cover
        # the full extend_input_len starting from position 0. As we walk the
        # tree matching existing nodes, factor_offset tracks which portion of
        # the factors corresponds to the current node.
        lfc_enabled = is_lfc_enabled()
        from sglang.srt.environ import envs
        break_even = envs.SGLANG_LFC_BREAK_EVEN_TOKENS.value if lfc_enabled else 0
        factor_offset = 0
        factor_len = 0
        if lfc_enabled and lfc_factors is not None:
            # Get total factor token count from any layer's first tensor
            for _layer_id, _ftuple in lfc_factors.items():
                factor_len = _ftuple[0].shape[0]
                break

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            # LFC: Opportunistic gap filling — when walking through a
            # tombstoned node (no mamba_value, no lfc_factors) whose token
            # range is covered by the current request's factors, slice and
            # store factors on it. This enables LFC reconstruction for
            # future requests that match this prefix.
            # Placed AFTER the split so we fill the clean prefix node,
            # avoiding empty-tensor factors on the suffix child.
            # Note: no break_even check here — gap filling compares factors
            # vs NO caching (tombstoned), so any factors are worthwhile.
            # Memory budget in _lfc_try_store_factors is the only constraint.
            if (lfc_enabled and lfc_factors is not None
                    and node.mamba_value is None
                    and node.lfc_factors is None
                    and node != self.root_node
                    and factor_offset + prefix_len <= factor_len):
                sliced_factors = {}
                for layer_id, factor_tuple in lfc_factors.items():
                    sliced_factors[layer_id] = tuple(
                        t[factor_offset:factor_offset + prefix_len].clone()
                        for t in factor_tuple
                    )
                self._lfc_try_store_factors(node, sliced_factors, gap_fill=True)

            factor_offset += prefix_len
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        # Adjust lfc_factors to only cover remaining (unconsumed) tokens
        if lfc_factors is not None and factor_offset > 0:
            if factor_offset >= factor_len:
                lfc_factors = None
            else:
                remaining_factors = {}
                for layer_id, factor_tuple in lfc_factors.items():
                    remaining_factors[layer_id] = tuple(
                        t[factor_offset:] for t in factor_tuple
                    )
                lfc_factors = remaining_factors

        mamba_value_exist = False
        use_factors_only = (
            lfc_enabled
            and lfc_factors is not None
            and len(key) < break_even
        )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value

            if use_factors_only:
                # Short segment: store factors, skip snapshot.
                # The factors are ~1% of snapshot size per token, so for
                # segments < break_even tokens, factors use less memory.
                new_node.mamba_value = None
                self._lfc_try_store_factors(new_node, lfc_factors)
                self.full_lru_list.insert_mru(new_node)
                # Not in mamba_lru (no mamba_value)
                node.children[child_key] = new_node
                self.full_evictable_size_ += len(value)
                # Signal caller to free the unused mamba_value
                mamba_value_exist = True
            else:
                # Long segment or no factors: store snapshot as usual
                new_node.mamba_value = mamba_value
                # Also store LFC factors alongside snapshot. When the snapshot
                # is later evicted (tombstoned) or the node is split, the
                # factors survive and enable LFC reconstruction.
                if lfc_enabled and lfc_factors is not None:
                    self._lfc_try_store_factors(new_node, lfc_factors)
                self.full_lru_list.insert_mru(new_node)
                self.mamba_lru_list.insert_mru(new_node)
                node.children[child_key] = new_node
                self.full_evictable_size_ += len(value)
                self.mamba_evictable_size_ += len(mamba_value)
        elif node.mamba_value is None:  # add for mamba tombstone
            if use_factors_only and node.lfc_factors is not None:
                # Node already has factors from before — no need to restore snapshot
                mamba_value_exist = True
                self.full_lru_list.reset_node_mru(node)
                node.last_access_time = get_last_access_time()
            elif use_factors_only:
                # Store factors instead of restoring snapshot
                self._lfc_try_store_factors(node, lfc_factors)
                mamba_value_exist = True
                self.full_lru_list.reset_node_mru(node)
                node.last_access_time = get_last_access_time()
            else:
                # Restore snapshot as usual
                node.mamba_value = mamba_value
                # Also store factors for resilience after future tombstoning
                if lfc_enabled and lfc_factors is not None and node.lfc_factors is None:
                    self._lfc_try_store_factors(node, lfc_factors)
                self.full_lru_list.reset_node_mru(node)
                self.mamba_lru_list.insert_mru(node)
                self.mamba_evictable_size_ += len(mamba_value)
                node.last_access_time = get_last_access_time()
        else:  # mamba value already exists
            mamba_value_exist = True
            # Snapshot exists → factors are redundant, don't store
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        # Clean up LFC factor memory tracking
        if is_lfc_enabled():
            self._lfc_remove_factors(node)
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        if node.mamba_value is not None:
            self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None
        # Note: node.lfc_factors is intentionally preserved when tombstoning.
        # LFC factors allow SSM state reconstruction at ~1% memory cost of full state.

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        # Free pending promotion slot if present
        pending_promo = getattr(node, '_lfc_pending_mamba_value', None)
        if pending_promo is not None:
            self.req_to_token_pool.mamba_pool.free(pending_promo)
            self._pending_promo_nodes.discard(id(node))
            del node._lfc_pending_mamba_value
        # Clean up LFC factor memory tracking
        if is_lfc_enabled():
            self._lfc_remove_factors(node)
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_leaves(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
