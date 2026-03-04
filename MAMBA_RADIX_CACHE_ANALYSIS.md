# SGLang Mamba Radix Cache 在共享前缀场景下的 Prefix Hit Rate 分析

## 摘要

SGLang 为 Mamba2（SSM）模型提供了专门的 `MambaRadixCache`，在 radix tree 节点上存储 SSM state 实现 prefix caching。从理论上分析，在共享前缀（shared-prefix）工作负载下——即多个请求共享相同的 system prompt、但拥有不同的用户输入——radix tree 的 split 操作会导致中间节点 `mamba_value = None`，使得纯前缀级别的 mamba 匹配失效。

然而，**实际端到端实验表明，整体 mamba hit rate 与 attention hit rate 几乎一致（~78%，差距 < 0.1%）**。这是因为请求在实际运行中主要匹配的是叶节点（完整序列），而非中间节点（纯前缀），且 mamba cache pool（672 slots）为叶节点提供了充足的 mamba state 存储。

本文从源代码层面追踪 split 导致 `mamba_value = None` 的因果链，并通过端到端实验验证其实际影响。

> **重要更正**：本文早期版本声称"SGLang 主动禁用 Mamba 模型的 radix cache"并引用了 `model_runner.py` 第 416-430 行的代码。经验证，**该代码在当前代码库（v0.5.6.post2）中不存在**。Qwen3-Next 等混合架构模型的 radix cache 默认启用且正常工作。详见第 8 节。

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

## 4. ~~第二层防线：SGLang 主动禁用 Mamba 模型的 Radix Cache~~ [已更正]

> **更正说明**：本节早期版本声称 SGLang 在 `model_runner.py` 第 416-430 行主动禁用 Mamba 模型的 radix cache，并引用了 `SGLANG_FORCE_RADIX_CACHE` 和 `SGLANG_LFC_ENABLED` 两个环境变量。**经实际代码验证（v0.5.6.post2），上述代码和环境变量均不存在于当前代码库中。**

### 4.1 实际代码行为

当前 `model_runner.py` 中 Mamba 相关的配置属性：

```python
# model_runner.py, line 1495-1520
@property
def hybrid_gdn_config(self):
    config = self.model_config.hf_config
    if isinstance(config, Qwen3NextConfig | JetNemotronConfig | JetVLMConfig):
        return config      # ← Qwen3-Next 走这条路径
    return None

@property
def mamba2_config(self):
    config = self.model_config.hf_config
    if isinstance(config, FalconH1Config | NemotronHConfig):
        return config      # ← 只覆盖 FalconH1 和 NemotronH
    return None

@property
def mambaish_config(self):
    return self.mamba2_config or self.hybrid_gdn_config or self.kimi_linear_config
```

关键发现：
- **没有**任何自动禁用 radix cache 的逻辑
- `Qwen3NextConfig` 属于 `hybrid_gdn_config`，不属于 `mamba2_config`
- Radix cache 对所有 Mamba 混合架构模型**默认启用**

### 4.2 Mamba Cache 自动分配

当 radix cache 启用且 `mamba_cache_per_req > 0` 时，`handle_max_mamba_cache()` 根据 `mamba_full_memory_ratio`（默认 0.9）自动分配 mamba cache pool：

```python
# model_runner.py, line 1466-1481
mamba_state_memory_raw = (
    total_rest_memory * server_args.mamba_full_memory_ratio
    / (1 + server_args.mamba_full_memory_ratio)
)
server_args.max_mamba_cache_size = int(
    (mamba_state_memory_raw * (1 << 30))
    // config.mamba2_cache_params.mamba_cache_per_req
)
```

在 Qwen3-Next-80B-A3B-Instruct + 4×A100 80GB 环境下，自动分配了 **672 个 mamba cache slots**，远超典型工作负载需要的数量。

---

## 5. 仿真验证与局限性

> **注意**：本节的仿真分析仅考虑了"中间节点"（纯前缀）的匹配，忽略了请求完成后叶节点会携带有效 `mamba_value` 的事实。仿真得出的 "0% mamba hit rate" 与端到端实验结果（~78%）不符。详见第 8 节的实验验证。

### 5.1 仿真实验

仿真脚本 `benchmark/prefix_hit_rate_simulation.py` 模拟了 radix tree 的插入和匹配行为：

**共享前缀数据集**（16 groups × 16 prompts, system_prompt=2048 tokens, question=128 tokens）：

| 模式 | Token-level Hit Rate | 命中请求数 |
|------|---------------------|-----------|
| **Attention** (任何节点都可匹配) | 88.2% | 240/256 |
| **Mamba cow_mamba** (只匹配 `mamba_value≠None` 的节点) | **0.0%** | **0/256** |
| **LFC** (匹配有 `mamba_value` 或 `lfc_factors` 的节点) | 88.2% | 240/256 |

### 5.2 仿真的关键局限性

仿真得出 "0% mamba hit rate" 的原因是：**它只检查了请求能否匹配 `[system_prompt]` 中间节点**。但在真实服务器中，请求的匹配行为远比这复杂：

1. **叶节点匹配**：同一 group 内的不同请求虽然不能匹配 `[system_prompt]` 中间节点（`mamba_value=None`），但同一 group 内**完全相同的请求序列**（system_prompt + question）可以匹配叶节点，而叶节点的 `mamba_value` 是有效的。

2. **请求完成后的树状态**：每个请求完成后，其完整序列（包括生成的 output tokens）会被插入 radix tree。后续请求如果具有相同前缀，会匹配到已完成请求的叶节点路径上，这些节点保留了有效的 `mamba_value`。

3. **仿真忽略了 output token 的积累**：在真实 benchmark 中，已完成请求的 output 部分也被缓存在树中，扩展了可匹配的叶节点路径。

### 5.3 对照实验：多轮对话（ShareGPT）

在多轮对话场景中（ShareGPT 数据集，200 conversations，1233 requests），结果截然不同：

| 模式 | Token-level Hit Rate | 命中请求数 |
|------|---------------------|-----------|
| **Attention** | 76.5% | 1117/1233 |
| **Mamba cow_mamba** | **76.5%** | 1034/1233 |
| **LFC** | 76.5% | 1117/1233 |

Mamba 的 hit rate 与 Attention 完全相同（76.5%）。多轮对话是 append-only 模式，不触发 split，`mamba_value` 始终完整保留。

### 5.4 仿真结论（需要端到端实验修正）

仿真正确地揭示了 split 导致中间节点 `mamba_value=None` 的机制，但其 "0% hit rate" 的结论**不适用于端到端实际服务场景**，原因是忽略了叶节点匹配和 mamba cache pool 的作用。

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

## 7. 理论因果链（部分成立）

```
SSM state 是不可逆的压缩摘要
    ↓
radix tree split 时，无法从 state(prefix+suffix) 中提取 state(prefix)
    ↓
_split_node() 只能设置 new_node.mamba_value = None
    ↓
_match_prefix_helper() 不接受 mamba_value=None 的节点
    ↓
仅匹配 system_prompt 中间节点时，best_last_node 退回到 root
    ↓
[理论推断] prefix hit rate = 0% → [实际结果] ✗ 不成立
```

上述因果链在"中间节点匹配"层面是正确的。但最终的 "hit rate = 0%" 结论不成立，因为：
1. **请求并非只匹配中间节点**：同一 group 的后续请求会匹配到已完成请求的叶节点，这些叶节点有有效的 `mamba_value`
2. **SGLang 并未禁用 Mamba 模型的 radix cache**：当前代码中没有自动禁用逻辑
3. **Mamba cache pool 容量充足**：672 slots 远超典型工作负载所需

---

## 8. 端到端实验验证 [NEW]

为验证上述理论分析的实际影响，我们实施了端到端的 prefix hit rate 埋点实验。

### 8.1 实验方法

在 SGLang pipeline 中新增两个 per-request 指标：

- **`attn_potential_hit_tokens`**：radix tree 纯 key 匹配（不检查 `mamba_value`）的 token 总数，代表 attention KV cache 的理论可复用量
- **`mamba_hit_tokens`**：radix tree 中 `mamba_value` 有效的最深匹配节点处的 token 总数，代表实际 mamba 状态可复用量

两个指标之差即为 "split 导致 `mamba_value=None` 的实际损失"。

**实现方式**：修改 `_match_prefix_helper` 返回 3-tuple，增加 `attn_match_len`；通过 output pipeline 传播到 API 的 `prompt_tokens_details` 字段。

### 8.2 实验环境

| 组件 | 规格 |
|------|------|
| 模型 | Qwen/Qwen3-Next-80B-A3B-Instruct (80B MoE, 3B active) |
| 架构 | Qwen3NextForCausalLM (Mamba + Attention hybrid) |
| GPU | 4x NVIDIA A100 80GB PCIe |
| TP | 4 |
| SGLang | v0.5.6.post2 |
| Mamba Cache | 672 slots (auto-allocated, ~12GB/GPU) |
| KV Cache | ~1.17M tokens |

### 8.3 Sanity Check：确认埋点正确检测到差距

两个连续请求共享 system prompt：

```
Request 1: prompt_tokens_details = null (cold cache)
Request 2: prompt_tokens_details = {
    "attn_potential_hit_tokens": 21,
    "mamba_hit_tokens": 0
}
```

第二个请求中 attention 匹配到 21 个 token（system prompt），但 mamba 匹配为 0 — **正是因为 split 后中间节点 `mamba_value = None`**。这确认了第 2-3 节描述的机制确实存在。

### 8.4 主要结果：Mamba 与 Attention Hit Rate 几乎一致

使用 `generated-shared-prefix` 数据集，8 groups × 8 prompts/group，prefix=2048 tokens：

| 配置 | 请求速率 | Mean TTFT (ms) | Attn Hit Rate | Mamba Hit Rate | 差距 |
|------|---------|---------------|--------------|---------------|------|
| **Radix ON** | **1 req/s** | **329.6** | **0.7824** | **0.7824** | **0.000062** |
| Baseline (no cache) | 1 req/s | 1873.0 | 0.0000 | 0.0000 | - |
| **Radix ON** | **4 req/s** | **409.4** | **0.7824** | **0.7824** | **0.000050** |
| Baseline (no cache) | 4 req/s | 3139.7 | 0.0000 | 0.0000 | - |
| **Radix ON** | **8 req/s** | **683.3** | **0.7824** | **0.7824** | **0.000062** |
| Baseline (no cache) | 8 req/s | 3677.8 | 0.0000 | 0.0000 | - |

**核心发现：attention 与 mamba hit rate 的差距可忽略不计（<0.01%）。**

### 8.5 Cache 压力分析

| 配置 | Groups×Per | Prefix | Rate | Attn HR | Mamba HR | 差距 | TTFT (ms) |
|------|-----------|--------|------|---------|---------|------|-----------|
| Standard | 8×8 | 2048 | 4 | 0.7824 | 0.7824 | 0.000050 | 409.4 |
| High Pressure | 32×4 | 4096 | 4 | 0.6034 | 0.6032 | 0.000138 | 1746.0 |
| Few Groups | 4×16 | 2048 | 4 | 0.8851 | 0.8851 | 0.000007 | 520.9 |
| **Many Groups** | **64×2** | **2048** | **8** | **0.3395** | **0.3388** | **0.000760** | **1645.3** |
| Long Prefix | 8×8 | 8192 | 4 | 0.7790 | 0.7790 | 0.000008 | 3003.6 |

观察：
- 最大差距（0.076%）出现在 **64 groups、rate=8** 的极端压力场景
- 即使在此最坏情况下，差距也 < 0.1%
- Group 数越少，hit rate 越高（4×16 达 88.5%）
- Group 数越多，hit rate 越低（64×2 仅 34%），但这是因为 radix tree 中不同 group 的叶节点相互竞争，与 mamba 无关

### 8.6 TTFT 加速效果

| Rate (req/s) | Baseline TTFT (ms) | Radix ON TTFT (ms) | 加速比 |
|------|-------------------|-------------------|---------:|
| 1 | 1873.0 | 329.6 | **5.68x** |
| 4 | 3139.7 | 409.4 | **7.67x** |
| 8 | 3677.8 | 683.3 | **5.38x** |

Radix cache 在共享前缀工作负载上提供 **5-8x TTFT 改善**。

### 8.7 为什么实验结果与仿真预测不同

仿真预测 "mamba hit rate = 0%"，但实验显示 ~78%。根本原因是**仿真只考虑了中间节点匹配，忽略了叶节点匹配**。

真实服务器中的请求匹配流程：

```
请求 1: [SP + Q0] → cold miss, prefill all → insert leaf with mamba_value ✓
请求 2: [SP + Q1] → split: [SP] node mamba_value=None
                   → 但 Q1 是新 suffix, 无法匹配任何叶节点 → miss
                   → prefill all → insert leaf [Q1] with mamba_value ✓
请求 3: [SP + Q0] → 匹配叶节点 [SP+Q0+output0] → 该叶节点 mamba_value ✓ → HIT!
请求 4: [SP + Q1] → 匹配叶节点 [SP+Q1+output1] → 该叶节点 mamba_value ✓ → HIT!
```

**关键洞察**：

1. **首次 miss 后，后续相同序列的请求命中叶节点**。叶节点在请求完成后被 `_insert_helper()` 写入，携带完整的 `mamba_value`。

2. **中间节点 `mamba_value=None` 的影响仅限于"首次出现新 suffix"的请求**。一旦该 suffix 的完整序列被缓存为叶节点，后续请求不再需要匹配中间节点。

3. **Mamba cache pool（672 slots）远大于所需**。即使 64 个 group 也只需 64 个 slot，远未耗尽 672 的容量。

### 8.8 何时会出现显著差距

根据实验和分析，mamba-attention 差距会在以下条件下增大：

| 条件 | 当前实验 | 触发阈值 |
|------|---------|---------|
| 唯一前缀数 >> mamba cache slots | 64 << 672 | >672 个不同前缀 |
| mamba_full_memory_ratio 过低 | 0.9（默认） | 降低后 pool 缩小 |
| 高频新前缀到达 | 固定 group 集合 | 持续流入全新前缀 |
| 极端 eviction 压力 | 未触发 | mamba pool 满 + 频繁 LRU 驱逐 |

---

## 9. 修正后的结论

### 9.1 原文正确的部分

- **Split 机制**（第 2 节）：`_split_node()` 设 `mamba_value = None` 是 SSM 不可拆分性的必然结果 ✓
- **中间节点匹配失败**（第 3 节）：`_match_prefix_helper()` 确实跳过 `mamba_value=None` 的节点 ✓
- **所有层共享 prefix_indices**（第 3.4 节）：Mamba 和 Attention 无法独立设置不同的 prefix 长度 ✓
- **多轮对话不触发 split**（第 5.3 节）：append-only 模式下 cow_mamba 正常工作 ✓

### 9.2 原文需要修正的部分

| 原文主张 | 修正 |
|---------|------|
| "SGLang 主动禁用 Mamba 模型的 radix cache"（第 4 节） | **代码不存在**。当前 v0.5.6.post2 中没有自动禁用逻辑，Qwen3-Next 的 radix cache 默认启用 |
| "共享前缀场景下 prefix hit rate 为 0%"（标题、摘要） | **端到端实验显示 ~78%**。0% 仅适用于中间节点匹配，叶节点匹配提供了高 hit rate |
| "cow_mamba 只在不触发 split 的场景下有效"（第 5.3 节） | **过于绝对**。叶节点匹配在共享前缀场景下同样有效，hit rate 与 attention 几乎一致 |
| "prefix hit rate = 0% → SGLang 主动禁用"（第 7 节因果链） | **因果链不完整**。实际 hit rate 远非 0%，且 SGLang 并未禁用 |

### 9.3 综合结论

1. **Split 导致中间节点 `mamba_value=None` 的问题确实存在**，但其实际影响可忽略不计（<0.1%），因为请求主要通过叶节点匹配获得 mamba state。

2. **MambaRadixCache 在共享前缀场景下工作良好**：mamba hit rate 与 attention hit rate 几乎一致（~78%），radix cache 提供 5-8x TTFT 加速。

3. **潜在风险**：当唯一前缀数超过 mamba cache pool 容量（当前 672 slots）时，差距可能显著增大。建议在高多样性流量下监控 `attn_potential_hit_tokens` 与 `mamba_hit_tokens` 的差异。

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
| `model_executor/model_runner.py` | 1495-1520 | `mambaish_config` 属性（**无**自动禁用逻辑） |
| `model_executor/model_runner.py` | 1446-1493 | `handle_max_mamba_cache()` mamba cache 自动分配 |
| `managers/schedule_batch.py` | 1231 | `prefix_indices` 决定 input_ids 的截断位置 |
| `managers/schedule_batch.py` | 737-751 | 从 `match_prefix()` 获取 prefix_indices |

## 附录 B：仿真工具

仿真脚本 `benchmark/prefix_hit_rate_simulation.py` 可用于验证中间节点匹配行为（注意：仿真结果不代表端到端 hit rate）：

```bash
# 共享前缀场景（验证中间节点 mamba_value=None）
python benchmark/prefix_hit_rate_simulation.py \
    --dataset shared-prefix \
    --num-groups 16 --prompts-per-group 16 \
    --system-prompt-len 2048 --question-len 128

# 多轮对话场景（对照实验）
python benchmark/prefix_hit_rate_simulation.py \
    --dataset sharegpt \
    --num-conversations 200
```

## 附录 C：端到端实验工具

端到端 hit rate 验证实验的工具链：

```bash
# 启动服务
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tp 4 --host 0.0.0.0 --port 30000 \
    --enable-cache-report

# 运行 benchmark（共享前缀场景）
python benchmark/hicache/bench_serving.py \
    --backend sglang --base-url http://127.0.0.1:30000 \
    --dataset-name generated-shared-prefix \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --num-prompts 64 \
    --gsp-num-groups 8 --gsp-prompts-per-group 8 \
    --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 128 \
    --request-rate 4 \
    --output-file results/gsp_radix_rr4.jsonl

# 分析结果
python benchmark/hicache/analyze_hit_rate.py \
    --results-dir results/ --output-dir analysis/
```

输出中 `prompt_tokens_details` 包含 `attn_potential_hit_tokens` 和 `mamba_hit_tokens` 两个指标，两者之差即为 mamba split 造成的实际损失。
