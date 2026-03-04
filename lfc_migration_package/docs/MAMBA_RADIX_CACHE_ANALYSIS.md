# 为什么 SGLang 的 Mamba Radix Cache 在共享前缀场景下 Prefix Hit Rate 为 0%

## 摘要

SGLang 为 Mamba2（SSM）模型提供了专门的 `MambaRadixCache`，在 radix tree 节点上存储 SSM state 实现 prefix caching。但在最常见的共享前缀（shared-prefix）工作负载下——即多个请求共享相同的 system prompt、但拥有不同的用户输入——这个 cache **完全失效**，prefix hit rate 为 0%。

本文将从源代码层面完整追踪这个问题的因果链，解释为什么会出现这个现象、为什么它是 SSM state 不可拆分性的必然结果、以及为什么 SGLang 最终选择在生产环境中主动禁用 Mamba 模型的 radix cache。

---

## 1. 前置知识：Transformer KV Cache vs SSM State

要理解这个问题，首先需要理解两种缓存的数据结构差异：

| 属性 | Transformer KV Cache | SSM State (Mamba2) |
|------|---------------------|-------------------|
| **粒度** | Per-token: 每个 token 独立存储一对 K/V 向量 | Per-sequence: 整个序列压缩为一个固定大小的状态矩阵 |
| **Shape** | `[num_tokens, num_heads, head_dim]` | `[num_heads, head_dim, state_dim]` |
| **语义** | Token i 的 key 和 value | 处理完 token 0, 1, ..., N 后的累积摘要 |
| **可拆分性** | ✓ 取前 k 个 token 的 KV = 简单切片 `cache[:k]` | ✗ 无法从最终 state 中提取"只处理前 k 个 token 后的 state" |

**关键差异：KV Cache 是一个 token 列表，可以按位置任意切分；SSM State 是一个不可逆的压缩摘要，无法回退到中间状态。**

SSM 的状态更新方程说明了为什么不可逆：
```
state_{t+1} = state_t × decay(g_t) + input(k_t, v_t, beta_t)
```
这是一个递归过程。`state_N` 包含了 token 0..N 的所有信息，但无法从 `state_N` 中分离出 `state_k` (k < N)。

---

## 2. Radix Tree 的 Split 操作：问题的根源

### 2.1 什么时候会触发 Split

Radix Tree（前缀树）的核心操作之一是 **split**：当两个序列共享一段前缀但后缀不同时，需要将一个节点拆分为两个。

**典型场景：多个请求共享 system prompt**

```
请求 1: [system_prompt] + [question_0]
请求 2: [system_prompt] + [question_1]
请求 3: [system_prompt] + [question_2]
...
```

这是 LLM serving 中最常见的 prefix caching 使用场景。

### 2.2 Split 的具体过程（以两个请求为例）

**Step 1：请求 1 完成，将完整序列插入 radix tree**

```
树结构:
Root ──→ [system_prompt + question_0]
          mamba_value = state(全序列)  ✓
          value = KV_cache(全序列)     ✓
```

节点存储了处理完整个 `system_prompt + question_0` 后的 SSM state。

**Step 2：请求 2 到来，进行 prefix matching**

请求 2 的 token 序列 `[system_prompt + question_1]` 与树中已有的 `[system_prompt + question_0]` 进行匹配。在 `system_prompt` 的最后一个 token 处，两者开始分叉。此时触发 `_split_node()`。

**Split 后的树结构：**

```
Root ──→ [system_prompt]                    ← 新创建的父节点
          mamba_value = ???
          value = KV_cache[:len(SP)]   ✓    ← KV 可以按 token 切片
          │
          ├──→ [question_0]                 ← 原节点变为子节点
          │     mamba_value = state(SP+Q0)  ✓
          │     value = KV_cache[len(SP):]  ✓
          │
          └──→ [question_1]                 ← 请求 2 创建的新节点
                mamba_value = state(SP+Q1)  ✓
                value = KV_cache(Q1)        ✓
```

**核心问题**：父节点 `[system_prompt]` 的 `mamba_value` 应该存什么？

它应该存储 "只处理了 system_prompt 这些 token 后的 SSM state"。但我们手头只有 "处理完 system_prompt + question_0 全部 token 后的 SSM state"。如前所述，SSM state 不可逆——**无法从 `state(SP+Q0)` 中提取 `state(SP)`**。

### 2.3 SGLang 的处理：设置 `mamba_value = None`

源代码 `mamba_radix_cache.py` 第 1019-1028 行：

```python
def _split_node(self, key, child, split_len):
    new_node = TreeNode()
    new_node.children = {self.get_child_key_fn(key[split_len:]): child}
    new_node.parent = child.parent
    new_node.mamba_value = None          # ← 关键行：SSM state 无法拆分
    new_node.full_lock_ref = child.full_lock_ref
    new_node.mamba_lock_ref = 0
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len]  # KV cache 可以正常切片
```

注意对比：
- `new_node.value = child.value[:split_len]` — KV Cache **可以** 按 token 位置切片 ✓
- `new_node.mamba_value = None` — SSM State **无法** 拆分，只能设为 None ✗

这不是一个 bug，而是 SSM 数据结构的必然结果。

---

## 3. `mamba_value = None` 如何导致 Prefix Matching 完全失败

### 3.1 `_match_prefix_helper()` 的匹配逻辑

当请求 3 `[system_prompt + question_2]` 到来时，radix tree 尝试进行 prefix matching。匹配函数 `_match_prefix_helper()` 遍历树节点，寻找可复用的最深节点。

源代码第 940-1017 行的核心逻辑（简化版）：

```python
def _match_prefix_helper(self, key):
    node = self.root_node
    best_last_node = node       # 初始化为 root（无缓存）
    best_value_len = 0          # 初始化为 0（无匹配）

    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]

        # 只有当节点有 SSM state 时，才更新 best
        if node.mamba_value is not None:       # ← 检查点
            best_value_len = len(value)
            best_last_node = node
        # (LFC 相关逻辑暂时忽略)

        # 继续向下遍历...
        prefix_len = self.key_match_fn(child.key, key)
        ...

    # 最终节点也检查一次
    if node.mamba_value is not None:
        best_value_len = len(value)
        best_last_node = node

    return value[:best_value_len], best_last_node
```

**关键逻辑**：`best_last_node` 只在 `node.mamba_value is not None` 时更新。如果一个节点的 `mamba_value` 是 `None`，它不会被视为有效匹配点。

### 3.2 具体追踪：请求 3 的匹配过程

树结构：
```
Root ──→ [system_prompt] (mamba_value=None)
          ├──→ [question_0] (mamba_value=✓)
          └──→ [question_1] (mamba_value=✓)
```

请求 3 的 token: `[system_prompt + question_2]`

```
遍历步骤：
1. 从 Root 开始
   - best_last_node = Root, best_value_len = 0
   - Root 的 children 中有 [system_prompt] 节点，匹配成功

2. 进入 [system_prompt] 节点
   - 检查 node (= Root): mamba_value? → Root 没有 mamba_value → 不更新 best
   - 完整匹配 [system_prompt] 的所有 token → 继续

3. 检查 [system_prompt] 的 children
   - 没有以 question_2 开头的子节点 → 循环结束

4. 检查最终节点 [system_prompt]
   - mamba_value is None → 不更新 best

5. 返回结果
   - best_last_node = Root
   - best_value_len = 0
   - 返回: value = [], last_node = Root
```

**结果：尽管 radix tree 中存储了完整的 `[system_prompt]` 的 KV Cache（`value` 字段），但因为该节点 `mamba_value = None`，prefix matching 返回空结果，就好像这个前缀从未被缓存过一样。**

### 3.3 为什么 KV Cache 也没有被复用

这是一个容易被忽视但非常重要的细节。Split 后的 `[system_prompt]` 节点中，KV Cache（`value` 字段）实际上是完好的——它正确存储了 system_prompt 每个 token 的 K/V 向量。

但 `_match_prefix_helper()` 的逻辑是：如果 `mamba_value = None`，则不更新 `best_last_node`，导致 `best_value_len` 停留在 0。最终 `match_prefix()` 返回 `value[:0] = []`，即空的 prefix_indices。

这意味着对于 hybrid Mamba 模型（同时包含 Attention 层和 Mamba 层）：
- **Mamba 层**：无法复用 → 需要从零计算 ✗
- **Attention 层**：KV Cache 物理上存在，但因为 prefix_indices 为空，**也无法复用** ✗

两层的缓存都浪费了。

### 3.4 为什么不能只复用 KV Cache 而跳过 Mamba

一个自然的问题是：为什么不能只用 attention 层的 KV Cache prefix，而让 Mamba 层从零计算？

原因是 SGLang 的调度系统对一个请求只生成一个 `prefix_indices` / `extend_input_len`，**所有层共享同一个 prefix 范围**。如果 prefix_indices 包含了前 2048 个 token，那么所有层——包括 Attention 和 Mamba——都只处理第 2049 个 token 开始的后缀。

源代码 `schedule_batch.py` 第 1231 行：
```python
input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
```

如果让 prefix_indices 包含 system_prompt 的 token：
- Attention 层：有 KV Cache，可以直接跳过 → ✓
- Mamba 层：没有 SSM state（`mamba_value=None`），**无法计算后续 token** → ✗

Mamba 层的计算是递归的：`state_t` 依赖 `state_{t-1}`。如果跳过了前 2048 个 token，Mamba 层没有 `state_{2048}` 作为起始状态，后续 token 的结果将完全错误。

**因此，当 Mamba 层的缓存不可用时，必须保守地将所有层的 prefix 都设为 0，即全部重新计算。**

---

## 4. 第二层防线：SGLang 主动禁用 Mamba 模型的 Radix Cache

正是因为认识到上述问题，SGLang 在初始化阶段直接禁用了 Mamba2 模型的 radix cache。

源代码 `model_runner.py` 第 416-430 行：

```python
if config := self.mamba2_config:
    class_name = config.__class__.__name__
    if envs.SGLANG_LFC_ENABLED.value:
        logger.warning(
            f"{class_name} model detected, but LFC is enabled — "
            f"keeping radix cache active for LFC reconstruction"
        )
    elif envs.SGLANG_FORCE_RADIX_CACHE.value:
        logger.warning(
            f"{class_name} model detected, but SGLANG_FORCE_RADIX_CACHE=1 — "
            f"keeping radix cache active (benchmark mode)"
        )
    else:
        logger.warning(f"{class_name} model detected, disable radix cache")
        self.server_args.disable_radix_cache = True  # ← 主动禁用
```

禁用后，`match_prefix()` 在入口处直接返回空结果（第 394-403 行）：

```python
if self.disable or len(key) == 0:
    return MatchResult(
        device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
        last_device_node=self.root_node,
        last_host_node=self.root_node,
    )
```

**这不是 bug，而是有意为之的工程决策**：既然 cow_mamba 在共享前缀场景下注定失效，不如直接禁用，避免白白维护 tree 结构和 SSM state 复制的开销。

---

## 5. 强制启用 Radix Cache 后的表现

通过设置 `SGLANG_FORCE_RADIX_CACHE=1`，可以绕过上述禁用逻辑，强制保留 radix cache。这让我们能够验证第 3 节的分析是否正确。

### 5.1 实验验证

我们编写了仿真脚本 `benchmark/prefix_hit_rate_simulation.py`，模拟了 radix tree 的插入和匹配行为，追踪三种 prefix matching 模式的 hit rate：

**共享前缀数据集**（16 groups × 16 prompts, system_prompt=2048 tokens, question=128 tokens）：

| 模式 | Token-level Hit Rate | 命中请求数 |
|------|---------------------|-----------|
| **Attention** (任何节点都可匹配) | 88.2% | 240/256 |
| **Mamba cow_mamba** (只匹配 `mamba_value≠None` 的节点) | **0.0%** | **0/256** |
| **LFC** (匹配有 `mamba_value` 或 `lfc_factors` 的节点) | 88.2% | 240/256 |

**结果完全印证了分析**：Mamba 的 prefix hit rate 为 **0%**。

仿真中的详细追踪显示，每次 split 后创建的 `[system_prompt]` 父节点都有 `mamba_value=None`，因此所有后续匹配到该节点的请求都无法获得 SSM state，被退回到 root。

### 5.2 对照实验：多轮对话（ShareGPT）

在多轮对话场景中（ShareGPT 数据集，200 conversations，1233 requests），结果截然不同：

| 模式 | Token-level Hit Rate | 命中请求数 |
|------|---------------------|-----------|
| **Attention** | 76.5% | 1117/1233 |
| **Mamba cow_mamba** | **76.5%** | 1034/1233 |
| **LFC** | 76.5% | 1117/1233 |

Mamba 的 hit rate 与 Attention 完全相同（76.5%）！

原因是多轮对话是 **append-only** 模式：

```
Turn 1: [msg1]
Turn 2: [msg1 + resp1 + msg2]          ← 包含 Turn 1 的全部 token + 新内容
Turn 3: [msg1 + resp1 + msg2 + resp2 + msg3]
```

每一轮都是前一轮的完整延续，不会出现 "共享前缀但后缀不同" 的情况。因此不触发 split，`mamba_value` 始终完整保留。

### 5.3 结论

| 工作负载类型 | 是否触发 Split | Mamba Hit Rate | 说明 |
|-------------|--------------|---------------|------|
| **共享 System Prompt** | ✓ 频繁 | **0%** | Split 后 `mamba_value=None`，所有后续请求无法匹配 |
| **多轮对话** | ✗ 几乎不触发 | **76.5%** | Append-only，无 split，cow_mamba 正常工作 |
| **完全相同请求** | ✗ 不触发 | **~100%** | 完全匹配叶子节点，cow_mamba 正常工作 |

**cow_mamba 只在不触发 split 的场景下有效；而最常见、最有价值的 prefix caching 场景——共享 system prompt——恰恰是必定触发 split 的。**

---

## 6. 为什么这对 SGLang 影响重大

### 6.1 共享 System Prompt 是最核心的 Prefix Caching 场景

在实际生产环境中，几乎所有 LLM 应用都使用 system prompt：

- **ChatGPT-style 应用**：所有用户共享同一个 system prompt（通常 500-4000 tokens）
- **RAG 系统**：相同文档片段作为上下文前缀
- **Few-shot Prompting**：相同的 few-shot examples 前缀
- **Tool Use / Function Calling**：相同的工具描述前缀

这些场景的共同特征就是：多个请求共享长前缀 + 不同的用户输入后缀。对于 Transformer 模型，这正是 radix cache 发挥最大价值的场景。

### 6.2 实际影响：TTFT（Time-to-First-Token）

我们使用 `bench_serving.py` 对 `tiiuae/falcon-h1-1.5b-instruct`（Hybrid Mamba2 + Attention 模型）进行了真实 benchmark：

**数据集**：8 groups × 16 prompts/group，system_prompt=1024 tokens，question=128 tokens

| 负载 | 无 Cache (Baseline) | 有 Cache (LFC) | TTFT 改善 |
|------|-------------------|----------------|----------|
| 低 (2 req/s) | 123.2ms | 139.9ms | +13.5% (开销) |
| 中 (4 req/s) | 1339.9ms | 230.3ms | **-82.8%** |
| 高 (8 req/s) | 1341.0ms | 419.3ms | **-68.7%** |

在中等负载下，TTFT 从 1.34 秒降低到 230 毫秒——**改善超过 80%**。这个差距完全来自于 prefix caching 是否可用。

---

## 7. 问题总结与因果链

```
SSM state 是不可逆的压缩摘要
    ↓
radix tree split 时，无法从 state(prefix+suffix) 中提取 state(prefix)
    ↓
_split_node() 只能设置 new_node.mamba_value = None
    ↓
_match_prefix_helper() 不接受 mamba_value=None 的节点
    ↓
best_last_node 退回到 root，返回空的 prefix_indices
    ↓
所有层（包括 Attention 的 KV Cache）都无法复用
    ↓
每个请求都必须从零 prefill 全部 token
    ↓
prefix hit rate = 0%
    ↓
SGLang 认识到这个问题后，主动禁用了 Mamba 模型的 radix cache
```

这不是一个 implementation bug，而是 **SSM 数据结构的固有限制**与 **radix tree prefix caching 机制** 之间的根本不兼容。Transformer 的 KV Cache 天然是 per-token、可拆分的，因此完美适配 radix tree；SSM State 是 per-sequence、不可拆分的，因此在 radix tree 需要 split 时必然失效。

---

## 附录 A：相关源代码位置

| 文件 | 行号 | 功能 |
|------|------|------|
| `mem_cache/mamba_radix_cache.py` | 1024 | `_split_node()` 设置 `mamba_value = None` |
| `mem_cache/mamba_radix_cache.py` | 966-978 | `_match_prefix_helper()` gap node 检测逻辑 |
| `mem_cache/mamba_radix_cache.py` | 394-403 | `match_prefix()` 禁用时直接返回空结果 |
| `mem_cache/mamba_radix_cache.py` | 414-436 | `match_prefix()` cow_mamba 状态复制逻辑 |
| `mem_cache/mamba_radix_cache.py` | 1088-1158 | `_insert_helper()` 插入节点时设置 mamba_value |
| `mem_cache/mamba_radix_cache.py` | 1197-1202 | `_tombstone_internal_node()` 驱逐时设置 mamba_value=None |
| `model_executor/model_runner.py` | 416-430 | 检测 Mamba2 模型并禁用 radix cache |
| `managers/schedule_batch.py` | 1231 | `prefix_indices` 决定 input_ids 的截断位置 |
| `managers/schedule_batch.py` | 737-751 | 从 `match_prefix()` 获取 prefix_indices |

## 附录 B：仿真工具

仿真脚本 `benchmark/prefix_hit_rate_simulation.py` 可用于复现上述 hit rate 数据：

```bash
# 共享前缀场景（验证 0% hit rate）
python benchmark/prefix_hit_rate_simulation.py \
    --dataset shared-prefix \
    --num-groups 16 --prompts-per-group 16 \
    --system-prompt-len 2048 --question-len 128

# 多轮对话场景（对照实验）
python benchmark/prefix_hit_rate_simulation.py \
    --dataset sharegpt \
    --num-conversations 200
```
