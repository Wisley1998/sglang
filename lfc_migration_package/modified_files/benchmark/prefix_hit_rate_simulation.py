"""
Prefix Hit Rate Simulation: Attention vs Mamba vs LFC

Simulates the MambaRadixCache prefix matching behavior to quantify
the difference in prefix reuse between:
  1. Attention (standard KV cache): any matched node is usable
  2. Mamba (cow_mamba): only nodes with mamba_value are usable
  3. LFC: nodes with mamba_value OR lfc_factors are usable

Datasets:
  - generated-shared-prefix: N groups sharing long system prompts + unique questions
  - ShareGPT: real multi-turn conversations (downloaded automatically)

Usage:
  python benchmark/prefix_hit_rate_simulation.py
  python benchmark/prefix_hit_rate_simulation.py --dataset sharegpt --num-requests 500
  python benchmark/prefix_hit_rate_simulation.py --dataset shared-prefix \
      --num-groups 16 --prompts-per-group 16 --system-prompt-len 2048
"""

import argparse
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Simulated Radix Tree ────────────────────────────────────────────────────

class SimTreeNode:
    """Simulated radix tree node mirroring MambaRadixCache.TreeNode."""

    _id_counter = 0

    def __init__(self):
        SimTreeNode._id_counter += 1
        self.id = SimTreeNode._id_counter
        self.children: Dict[int, "SimTreeNode"] = {}  # first_token → child
        self.parent: Optional["SimTreeNode"] = None
        self.key: List[int] = []  # token IDs stored in this node
        self.num_tokens: int = 0  # len(key), cached for convenience
        self.has_mamba_value: bool = False
        self.has_lfc_factors: bool = False
        self.last_access_time: float = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class SimRadixCache:
    """
    Simplified simulation of MambaRadixCache.

    Tracks three types of prefix hits:
    - attention: any node with tokens (standard KV cache behavior)
    - mamba: only nodes with has_mamba_value=True
    - lfc: nodes with has_mamba_value=True OR has_lfc_factors=True
    """

    def __init__(self, mamba_pool_size: Optional[int] = None):
        self.root = SimTreeNode()
        self.root.has_mamba_value = False  # root has no state
        self.mamba_pool_size = mamba_pool_size
        self.mamba_pool_used = 0
        self._time = 0.0
        # LRU: ordered dict mapping node.id → node, oldest first
        self.mamba_lru: OrderedDict[int, SimTreeNode] = OrderedDict()

    def _get_time(self) -> float:
        self._time += 0.001
        return self._time

    def _key_match(self, key1: List[int], key2: List[int]) -> int:
        """Return number of matching tokens at start of both keys."""
        i = 0
        for a, b in zip(key1, key2):
            if a != b:
                break
            i += 1
        return i

    def _split_node(self, child: SimTreeNode, split_len: int) -> SimTreeNode:
        """Split a node at split_len, creating a new parent."""
        new_parent = SimTreeNode()
        new_parent.key = child.key[:split_len]
        new_parent.num_tokens = split_len
        new_parent.parent = child.parent

        # SSM state cannot be split
        new_parent.has_mamba_value = False
        # LFC factors CAN be split (per-token)
        new_parent.has_lfc_factors = child.has_lfc_factors

        # Update child
        old_first_token = child.key[0] if child.key else None
        child.key = child.key[split_len:]
        child.num_tokens = len(child.key)
        child.parent = new_parent

        # Update tree structure
        new_parent.children[child.key[0]] = child
        if old_first_token is not None and new_parent.parent:
            new_parent.parent.children[new_parent.key[0]] = new_parent
        elif new_parent.parent is None:
            # parent is root
            pass

        new_parent.last_access_time = self._get_time()
        child.last_access_time = self._get_time()

        return new_parent

    def _alloc_mamba(self) -> bool:
        """Try to allocate a mamba pool slot. Returns True if successful."""
        if self.mamba_pool_size is None:
            self.mamba_pool_used += 1
            return True
        if self.mamba_pool_used < self.mamba_pool_size:
            self.mamba_pool_used += 1
            return True
        return False

    def _free_mamba(self, count: int = 1):
        self.mamba_pool_used = max(0, self.mamba_pool_used - count)

    def evict_mamba(self, count: int = 1):
        """Evict LRU mamba nodes to free pool slots."""
        evicted = 0
        to_remove = []

        for node_id, node in self.mamba_lru.items():
            if evicted >= count:
                break
            if not node.has_mamba_value:
                continue

            to_remove.append(node_id)

            if not node.is_leaf():
                # Internal node: tombstone (keep KV, remove mamba, keep LFC factors)
                node.has_mamba_value = False
                # lfc_factors preserved intentionally
                self._free_mamba(1)
                evicted += 1
            else:
                # Leaf node: fully delete
                node.has_mamba_value = False
                node.has_lfc_factors = False
                self._free_mamba(1)
                evicted += 1
                # Remove from parent
                if node.parent:
                    for k, v in list(node.parent.children.items()):
                        if v.id == node.id:
                            del node.parent.children[k]
                            break
                    # Check if parent became childless tombstone
                    self._cleanup_tombstone_leaves(node.parent)

        for nid in to_remove:
            if nid in self.mamba_lru:
                del self.mamba_lru[nid]

    def _cleanup_tombstone_leaves(self, node: SimTreeNode):
        """Iteratively delete tombstone nodes that became leaves."""
        while (node and node.parent and node != self.root
               and not node.has_mamba_value and len(node.children) == 0):
            parent = node.parent
            for k, v in list(parent.children.items()):
                if v.id == node.id:
                    del parent.children[k]
                    break
            if node.id in self.mamba_lru:
                del self.mamba_lru[node.id]
            node = parent

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, int, int]:
        """
        Simulate prefix matching and return three hit lengths:
        - attention_hit: total tokens matched (any node usable)
        - mamba_hit: tokens up to deepest node with mamba_value
        - lfc_hit: tokens up to deepest node with mamba_value or lfc_factors
        """
        node = self.root
        key = list(token_ids)

        total_matched = 0  # attention: all matched tokens
        best_mamba_len = 0
        best_lfc_len = 0
        lfc_chain_valid = True

        while key and key[0] in node.children:
            child = node.children[key[0]]

            # Check current node (before descending)
            if node != self.root:
                if node.has_mamba_value:
                    best_mamba_len = total_matched
                    best_lfc_len = total_matched
                    lfc_chain_valid = True
                elif node.has_lfc_factors and lfc_chain_valid:
                    best_lfc_len = total_matched
                elif not node.has_mamba_value and not node.has_lfc_factors:
                    lfc_chain_valid = False

            # Match child's key against remaining query
            prefix_len = self._key_match(child.key, key)

            if prefix_len < len(child.key):
                # Partial match - would trigger split in real cache
                # The split creates a parent with mamba_value=None, lfc_factors=child.has_lfc_factors
                total_matched += prefix_len
                # Check the would-be split parent
                # has_mamba_value = False (split always sets None)
                # has_lfc_factors = child.has_lfc_factors (factors can be split)
                if child.has_lfc_factors and lfc_chain_valid:
                    best_lfc_len = total_matched
                # mamba: never updated (split parent has no mamba_value)
                break
            else:
                # Full match
                total_matched += prefix_len
                node = child
                key = key[prefix_len:]

        # Final node check
        if node != self.root:
            if node.has_mamba_value:
                best_mamba_len = total_matched
                best_lfc_len = total_matched
            elif node.has_lfc_factors and lfc_chain_valid:
                best_lfc_len = total_matched

        return total_matched, best_mamba_len, best_lfc_len

    def insert(self, token_ids: List[int], has_lfc: bool = True):
        """Insert a completed request's tokens into the tree."""
        node = self.root
        key = list(token_ids)
        node.last_access_time = self._get_time()

        while key:
            if key[0] not in node.children:
                break

            child = node.children[key[0]]
            child.last_access_time = self._get_time()
            prefix_len = self._key_match(child.key, key)

            if prefix_len < len(child.key):
                # Partial match: split
                new_parent = self._split_node(child, prefix_len)
                # Fix parent's children reference
                node.children[new_parent.key[0]] = new_parent

                # If child had mamba in LRU, it stays; new_parent doesn't get mamba
                node = new_parent
                key = key[prefix_len:]
                break
            else:
                node = child
                key = key[prefix_len:]

        if key:
            # Create new leaf
            new_leaf = SimTreeNode()
            new_leaf.key = key
            new_leaf.num_tokens = len(key)
            new_leaf.parent = node
            new_leaf.has_lfc_factors = has_lfc
            new_leaf.last_access_time = self._get_time()

            # Allocate mamba pool slot
            if not self._alloc_mamba():
                self.evict_mamba(1)
                assert self._alloc_mamba(), "Failed to alloc mamba after eviction"
            new_leaf.has_mamba_value = True

            node.children[key[0]] = new_leaf
            self.mamba_lru[new_leaf.id] = new_leaf

        elif not node.has_mamba_value and node != self.root:
            # Key ended exactly at an existing tombstoned node: restore mamba
            if not self._alloc_mamba():
                self.evict_mamba(1)
                assert self._alloc_mamba()
            node.has_mamba_value = True
            if has_lfc:
                node.has_lfc_factors = True
            self.mamba_lru[node.id] = node
        else:
            # Exact match, mamba already exists: just update LRU
            if node.id in self.mamba_lru:
                self.mamba_lru.move_to_end(node.id)


# ─── Dataset Generators ──────────────────────────────────────────────────────

def generate_shared_prefix_dataset(
    num_groups: int = 16,
    prompts_per_group: int = 16,
    system_prompt_len: int = 2048,
    question_len: int = 128,
    vocab_size: int = 32000,
    seed: int = 42,
) -> List[List[int]]:
    """Generate a shared-prefix dataset (simulating sglang's generated-shared-prefix)."""
    rng = random.Random(seed)
    requests = []

    for g in range(num_groups):
        # Generate shared system prompt for this group
        system_prompt = [rng.randint(100, vocab_size - 1) for _ in range(system_prompt_len)]

        for p in range(prompts_per_group):
            # Generate unique question
            question = [rng.randint(100, vocab_size - 1) for _ in range(question_len)]
            requests.append(system_prompt + question)

    # Shuffle to simulate realistic arrival order
    rng.shuffle(requests)
    return requests


def load_sharegpt_multiturn(
    dataset_path: str = "",
    num_conversations: int = 100,
    max_turns: int = 10,
    max_tokens_per_turn: int = 512,
    seed: int = 42,
) -> List[List[int]]:
    """
    Load ShareGPT dataset and create multi-turn request sequences.

    Each conversation turn becomes a separate request:
    - Turn 1: [system + user1]
    - Turn 2: [system + user1 + assistant1 + user2]
    - Turn 3: [system + user1 + assistant1 + user2 + assistant2 + user3]
    ...

    Returns list of token_id sequences (simulated with hashes since we
    don't need a real tokenizer for hit rate simulation).
    """
    sharegpt_url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    # Download if needed
    if not dataset_path or not os.path.exists(dataset_path):
        cache_dir = os.path.expanduser("~/.cache/sglang")
        os.makedirs(cache_dir, exist_ok=True)
        dataset_path = os.path.join(cache_dir, "ShareGPT_V3_unfiltered_cleaned_split.json")

        if not os.path.exists(dataset_path):
            print(f"Downloading ShareGPT dataset to {dataset_path}...")
            import urllib.request
            urllib.request.urlretrieve(sharegpt_url, dataset_path)
            print("Download complete.")

    with open(dataset_path) as f:
        dataset = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(dataset)

    requests = []
    conv_count = 0

    for data in dataset:
        if conv_count >= num_conversations:
            break

        conversations = data.get("conversations", data.get("conversation", []))
        if len(conversations) < 2:
            continue

        conv_count += 1
        accumulated_tokens = []

        for turn_idx, turn in enumerate(conversations[:max_turns]):
            # Simulate tokenization: use hash of text to generate deterministic token IDs
            text = turn.get("value", "")
            # Simple simulation: ~4 chars per token, use character codes as token IDs
            turn_tokens = []
            for i in range(0, min(len(text), max_tokens_per_turn * 4), 4):
                chunk = text[i:i+4]
                token_id = hash(chunk) % 50000 + 100
                turn_tokens.append(token_id)

            if not turn_tokens:
                turn_tokens = [rng.randint(100, 50000) for _ in range(20)]

            accumulated_tokens = accumulated_tokens + turn_tokens

            # Each turn is a separate request with all previous context
            if len(accumulated_tokens) > 0:
                requests.append(list(accumulated_tokens))

    return requests


# ─── Simulation Runner ────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_idx: int
    total_tokens: int
    attention_hit: int
    mamba_hit: int
    lfc_hit: int

    @property
    def attention_hit_rate(self) -> float:
        return self.attention_hit / self.total_tokens if self.total_tokens > 0 else 0

    @property
    def mamba_hit_rate(self) -> float:
        return self.mamba_hit / self.total_tokens if self.total_tokens > 0 else 0

    @property
    def lfc_hit_rate(self) -> float:
        return self.lfc_hit / self.total_tokens if self.total_tokens > 0 else 0


def run_simulation(
    requests: List[List[int]],
    mamba_pool_size: Optional[int] = None,
    label: str = "",
) -> List[RequestResult]:
    """Run prefix hit rate simulation on a list of token sequences."""
    cache = SimRadixCache(mamba_pool_size=mamba_pool_size)
    results = []

    for i, token_ids in enumerate(requests):
        # 1. Match prefix (before insertion)
        attn_hit, mamba_hit, lfc_hit = cache.match_prefix(token_ids)

        results.append(RequestResult(
            request_idx=i,
            total_tokens=len(token_ids),
            attention_hit=attn_hit,
            mamba_hit=mamba_hit,
            lfc_hit=lfc_hit,
        ))

        # 2. Insert into cache (simulating cache_finished_req)
        cache.insert(token_ids, has_lfc=True)

    return results


def print_results(results: List[RequestResult], label: str):
    """Print aggregated hit rate statistics."""
    n = len(results)
    if n == 0:
        print(f"  [{label}] No results.")
        return

    # Skip first request (always cold miss)
    warm_results = [r for r in results if r.attention_hit > 0 or r.mamba_hit > 0 or r.lfc_hit > 0]

    total_tokens = sum(r.total_tokens for r in results)
    total_attn_hit = sum(r.attention_hit for r in results)
    total_mamba_hit = sum(r.mamba_hit for r in results)
    total_lfc_hit = sum(r.lfc_hit for r in results)

    attn_rates = [r.attention_hit_rate for r in results]
    mamba_rates = [r.mamba_hit_rate for r in results]
    lfc_rates = [r.lfc_hit_rate for r in results]

    # Count requests with any hit
    attn_any_hit = sum(1 for r in results if r.attention_hit > 0)
    mamba_any_hit = sum(1 for r in results if r.mamba_hit > 0)
    lfc_any_hit = sum(1 for r in results if r.lfc_hit > 0)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total requests: {n}")
    print(f"  Total tokens:   {total_tokens}")
    print()

    # Token-level hit rates
    print(f"  {'Metric':<35} {'Attention':>12} {'Mamba':>12} {'LFC':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Total prefix tokens hit':<35} {total_attn_hit:>12,} {total_mamba_hit:>12,} {total_lfc_hit:>12,}")
    print(f"  {'Token-level hit rate':<35} {total_attn_hit/total_tokens:>11.1%} {total_mamba_hit/total_tokens:>11.1%} {total_lfc_hit/total_tokens:>11.1%}")
    print(f"  {'Requests with any hit':<35} {attn_any_hit:>12} {mamba_any_hit:>12} {lfc_any_hit:>12}")
    print(f"  {'Request hit fraction':<35} {attn_any_hit/n:>11.1%} {mamba_any_hit/n:>11.1%} {lfc_any_hit/n:>11.1%}")
    print()

    # Per-request hit rate statistics
    import numpy as np
    attn_arr = np.array(attn_rates)
    mamba_arr = np.array(mamba_rates)
    lfc_arr = np.array(lfc_rates)

    print(f"  {'Per-request hit rate (mean)':<35} {np.mean(attn_arr):>11.1%} {np.mean(mamba_arr):>11.1%} {np.mean(lfc_arr):>11.1%}")
    print(f"  {'Per-request hit rate (median)':<35} {np.median(attn_arr):>11.1%} {np.median(mamba_arr):>11.1%} {np.median(lfc_arr):>11.1%}")
    print()

    # Average hit length
    avg_attn_len = np.mean([r.attention_hit for r in results])
    avg_mamba_len = np.mean([r.mamba_hit for r in results])
    avg_lfc_len = np.mean([r.lfc_hit for r in results])
    print(f"  {'Avg prefix hit length (tokens)':<35} {avg_attn_len:>11.1f} {avg_mamba_len:>11.1f} {avg_lfc_len:>11.1f}")

    # Gap analysis
    if total_attn_hit > 0:
        mamba_gap = (total_attn_hit - total_mamba_hit) / total_attn_hit * 100
        lfc_gap = (total_attn_hit - total_lfc_hit) / total_attn_hit * 100
        print()
        print(f"  Gap vs Attention (lower = better):")
        print(f"    Mamba: {mamba_gap:.1f}% of attention's prefix tokens are LOST (split/tombstone)")
        print(f"    LFC:   {lfc_gap:.1f}% of attention's prefix tokens are LOST")
        if total_mamba_hit < total_attn_hit:
            lfc_recovery = (total_lfc_hit - total_mamba_hit) / (total_attn_hit - total_mamba_hit) * 100
            print(f"    LFC recovers {lfc_recovery:.1f}% of mamba's lost prefix tokens")

    print(f"{'='*70}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prefix Hit Rate Simulation")
    parser.add_argument("--dataset", choices=["shared-prefix", "sharegpt", "both"],
                        default="both", help="Dataset to simulate")
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--prompts-per-group", type=int, default=16)
    parser.add_argument("--system-prompt-len", type=int, default=2048)
    parser.add_argument("--question-len", type=int, default=128)
    parser.add_argument("--num-conversations", type=int, default=200,
                        help="Number of ShareGPT conversations")
    parser.add_argument("--sharegpt-path", type=str, default="")
    parser.add_argument("--mamba-pool-size", type=int, default=None,
                        help="Mamba pool size limit (None = unlimited)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  Prefix Hit Rate Simulation")
    print("  Comparing: Attention (KV cache) vs Mamba (cow_mamba) vs LFC")
    print("=" * 70)

    # ── Shared Prefix Dataset ──────────────────────────────────────────────
    if args.dataset in ("shared-prefix", "both"):
        print(f"\n  Generating shared-prefix dataset:")
        print(f"    Groups: {args.num_groups}, Prompts/group: {args.prompts_per_group}")
        print(f"    System prompt: {args.system_prompt_len} tokens, Question: {args.question_len} tokens")

        requests = generate_shared_prefix_dataset(
            num_groups=args.num_groups,
            prompts_per_group=args.prompts_per_group,
            system_prompt_len=args.system_prompt_len,
            question_len=args.question_len,
            seed=args.seed,
        )
        print(f"    Total requests: {len(requests)}")
        print(f"    Avg tokens/request: {sum(len(r) for r in requests) / len(requests):.0f}")

        results = run_simulation(requests, mamba_pool_size=args.mamba_pool_size,
                                 label="shared-prefix")
        print_results(results,
                      f"Shared-Prefix Dataset ({args.num_groups} groups × {args.prompts_per_group} prompts, "
                      f"prefix={args.system_prompt_len} tokens)")

        # Also run with limited mamba pool to show eviction effects
        if args.mamba_pool_size is None:
            for pool_size in [32, 64, 128]:
                results_limited = run_simulation(
                    requests, mamba_pool_size=pool_size,
                    label=f"shared-prefix (pool={pool_size})"
                )
                print_results(results_limited,
                              f"Shared-Prefix Dataset (pool_size={pool_size})")

    # ── ShareGPT Dataset ──────────────────────────────────────────────────
    if args.dataset in ("sharegpt", "both"):
        print(f"\n  Loading ShareGPT dataset ({args.num_conversations} conversations)...")
        requests = load_sharegpt_multiturn(
            dataset_path=args.sharegpt_path,
            num_conversations=args.num_conversations,
            seed=args.seed,
        )
        print(f"    Total requests (turns): {len(requests)}")
        if requests:
            print(f"    Avg tokens/request: {sum(len(r) for r in requests) / len(requests):.0f}")

        results = run_simulation(requests, mamba_pool_size=args.mamba_pool_size,
                                 label="sharegpt")
        print_results(results,
                      f"ShareGPT Multi-turn ({args.num_conversations} conversations)")

        # With limited mamba pool
        if args.mamba_pool_size is None and requests:
            for pool_size in [64, 128]:
                results_limited = run_simulation(
                    requests, mamba_pool_size=pool_size,
                    label=f"sharegpt (pool={pool_size})"
                )
                print_results(results_limited,
                              f"ShareGPT Multi-turn (pool_size={pool_size})")


if __name__ == "__main__":
    main()
