# LFC (Linear Factor Caching) 迁移包

## 概述

LFC 是针对 Hybrid Mamba2 模型（如 FalconH1）的 prefix caching 优化。SGLang 原生的 `MambaRadixCache` 在共享前缀（shared-prefix）场景下 prefix hit rate 为 0%，因为 SSM state 不可拆分，radix tree 的 split 操作会导致父节点丧失 SSM state。LFC 通过保存 per-token 的计算因子（factors），在需要时从 factors 重建 SSM state，使 prefix caching 在 Mamba 模型上恢复正常工作。

**核心效果**：
- 共享前缀 Prefix Hit Rate: 0% → 88.2%（与 Transformer 模型相同）
- TTFT（中等负载）: 1340ms → 230ms（**-83%**）
- TTFT（高负载）: 1341ms → 419ms（**-69%**）

## 基线版本

- **SGLang commit**: `0648eb482` (`[Profiler] Add SGLANG_PROFILE_RECORD_SHAPES for recording shapes when profiling (#11641)`)
- **分支**: `main`
- **修改统计**: 10 个文件修改 (+768/-13 行)，2 个新文件 (+804 行)，1 个 benchmark 脚本 (+608 行)

---

## 迁移方式

### 方式一：使用 patch 文件（推荐）

确保目标集群的 SGLang 版本基于相同的 commit（`0648eb482`），然后：

```bash
cd /path/to/sglang
git apply lfc_full.patch
```

如果目标版本有差异，可能需要手动解决冲突。patch 文件位于 `lfc_full.patch`。

### 方式二：手动替换文件

将 `modified_files/` 下的文件逐个复制到目标 SGLang 项目的对应路径。每个文件的完整项目路径见下方文件清单。

> **注意**：方式二会替换整个文件，如果目标 SGLang 版本与基线不同，可能引入兼容性问题。建议先 diff 对比。

---

## 文件清单

### 修改的文件（10 个）

| # | 文件路径 | 修改量 | 修改内容摘要 |
|---|---------|-------|-------------|
| 1 | `python/sglang/srt/environ.py` | +8 行 | 新增 7 个 LFC 相关环境变量 |
| 2 | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | +91 行 | GDN/GLA 层的 factor capture 和 reconstruction |
| 3 | `python/sglang/srt/layers/attention/mamba/mamba.py` | +120 行 | Mamba2 层的 factor capture 和 reconstruction |
| 4 | `python/sglang/srt/managers/schedule_batch.py` | +55 行 | Req 新增 LFC 字段；factor 收集与传递 |
| 5 | `python/sglang/srt/managers/scheduler.py` | +3/-2 行 | 修复 `is_hybrid_gdn` 检测逻辑以覆盖 Mamba2 |
| 6 | `python/sglang/srt/mem_cache/mamba_radix_cache.py` | +280 行 | 核心：TreeNode LFC 字段、match_prefix LFC 路径、split factor 处理、显存预算 |
| 7 | `python/sglang/srt/mem_cache/memory_pool.py` | +147 行 | MambaPool 新增 LFC factor pool 管理方法 |
| 8 | `python/sglang/srt/model_executor/forward_batch_info.py` | +9 行 | ForwardBatch 新增 `lfc_reconstruction_factors` 和 `reqs` 字段 |
| 9 | `python/sglang/srt/model_executor/model_runner.py` | +65 行 | LFC 启用时保留 radix cache；修复 `--max-mamba-cache-size` 覆盖 bug |
| 10 | `python/sglang/srt/models/falcon_h1.py` | +1 行 | 传递 `forward_batch` 参数给 Mamba2 层 |

### 新增的文件（2 个）

| # | 文件路径 | 行数 | 用途 |
|---|---------|------|------|
| 11 | `python/sglang/srt/layers/attention/fla/lfc_reconstruct.py` | 409 行 | GDN/GLA 模型的 LFC 状态重建（Triton kernel + PyTorch fallback） |
| 12 | `python/sglang/srt/utils/catchup_timing.py` | 395 行 | `is_lfc_enabled()` 全局开关、timing collector 工具 |

### Benchmark 脚本（1 个）

| # | 文件路径 | 行数 | 用途 |
|---|---------|------|------|
| 13 | `benchmark/prefix_hit_rate_simulation.py` | 608 行 | Prefix hit rate 仿真：对比 Attention / Mamba / LFC 三种模式 |

### 文档（2 个，仅供参考，不需要复制到项目中）

| # | 文件 | 内容 |
|---|------|------|
| 14 | `docs/LFC_DESIGN_AND_ANALYSIS.md` | LFC 完整设计文档：问题背景、核心思想、实现架构、实验验证 |
| 15 | `docs/MAMBA_RADIX_CACHE_ANALYSIS.md` | Mamba Radix Cache 0% hit rate 问题的深度分析 |

---

## 各文件修改详情

### 1. `python/sglang/srt/environ.py`

新增 7 个环境变量：

```python
SGLANG_CATCHUP_TIMING = EnvBool(False)          # 启用 timing 收集
SGLANG_CATCHUP_TIMING_DIR = EnvStr("/tmp/sglang_catchup")  # timing 输出目录
SGLANG_LFC_ENABLED = EnvBool(False)             # 启用 LFC 模式（主开关）
SGLANG_SNAPSHOT_INTERVAL = EnvInt(500)          # snapshot 间隔
SGLANG_LFC_MEMORY_BUDGET_GB = EnvFloat(2.0)    # LFC factors GPU 显存预算
SGLANG_FORCE_RADIX_CACHE = EnvBool(False)       # 强制启用 radix cache（benchmark 用）
```

### 2. `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

**Factor Capture**（+~40 行）：在 GDN/GLA 层的 forward 中，提取 per-token 的 `(k, v, g, beta)` 因子，保存到 `req.pending_lfc_factors[layer_id]`。

**State Reconstruction**（+~50 行）：在 forward 开始时，检查 `lfc_reconstruction_factors`。如有，调用 `lfc_reconstruct_state()` 从 factors 重建 SSM state，写入 `initial_states`。

**Bug Fix**：消除了循环内的 `.item()` CUDA 同步调用，改为 `.tolist()` 一次性转换。

### 3. `python/sglang/srt/layers/attention/mamba/mamba.py`

**Factor Capture**（+~50 行）：在 Mamba2 层的 forward 中，提取 `(hidden_states, B, C, dt)` 因子，保存到 `req.pending_lfc_factors[layer_id]`。

**State Reconstruction**（+~70 行）：使用 `mamba_chunk_scan_combined()` kernel 从 factors 重建 SSM state。需传递 `seq_idx`、`chunk_indices`、`chunk_offsets`（Triton kernel 约束）和 `state_dtype=torch.float32`（指针类型匹配约束）。

### 4. `python/sglang/srt/managers/schedule_batch.py`

**Req 类新增字段**（+11 行）：
- `pending_lfc_factors`: forward → tree 方向，存储待缓存的 factors
- `lfc_reconstruction_factors`: tree → forward 方向，存储重建用的 factors
- `_needs_factor_capture`: 标记是否需要 factor capture

**Factor 收集**（+23 行）：
- `_collect_lfc_factors()` 方法：从 batch 中所有请求收集 reconstruction factors，传递给 ForwardBatch
- LFC 指标日志记录

**ModelWorkerBatch**（+7 行）：新增 `lfc_reconstruction_factors` 和 `reqs` 字段。

### 5. `python/sglang/srt/managers/scheduler.py`

修复 `is_hybrid_gdn` 检测：原逻辑只检测 `hybrid_gdn_config`，FalconH1（使用 `mamba2_config`）未被识别。修改后同时检测两种 config。

### 6. `python/sglang/srt/mem_cache/mamba_radix_cache.py`

这是 LFC 最核心的修改文件（+280 行）：

**TreeNode 扩展**：
- 新增 `lfc_factors` 字段：`Dict[layer_id, (k, v, g, beta)]`

**`_match_prefix_helper()` 增强**：
- 新增 LFC chain validity 追踪
- gap node 检测（`mamba_value=None` 且 `lfc_factors=None`）
- 有 `lfc_factors` 的节点可作为有效匹配点

**`match_prefix()` LFC 路径**：
- 当匹配到有 `lfc_factors` 但无 `mamba_value` 的节点时，触发 LFC 重建
- 向上遍历找最近的有 `mamba_value` 的 ancestor
- 收集 factor chain 并预合并（concat）为单次 kernel 调用的输入
- 存储到 `req.lfc_reconstruction_factors`

**`_split_node()` Factor 分割**：
- Split 时正确按 token 位置分割 `lfc_factors`
- 父节点获得前半部分，子节点保留后半部分
- 更新显存跟踪

**`_tombstone_internal_node()` Factor 保留**：
- 驱逐 SSM state 时保留 `lfc_factors`（内存占用远小于 SSM state）

**GPU 显存预算管理**：
- `_lfc_try_store_factors()`: 基于节点价值的选择性存储
- `_estimate_factor_memory()`: 估算 factor 内存占用
- `_compute_factor_value()`: 节点价值 = hit_count × num_children / key_len
- `_lfc_remove_factors()`: 清理 factor 内存跟踪
- 最小堆管理，懒删除策略

### 7. `python/sglang/srt/mem_cache/memory_pool.py`

**MambaPool LFC 方法**（+147 行）：
- Factor pool 的分配和管理
- 与现有 MambaPool 集成

### 8. `python/sglang/srt/model_executor/forward_batch_info.py`

**ForwardBatch 新增字段**（+9 行）：
- `lfc_reconstruction_factors: Optional[dict]` — 重建因子
- `reqs: Optional[List]` — 请求引用，用于 factor capture 后写回

### 9. `python/sglang/srt/model_executor/model_runner.py`

**Radix Cache 保留逻辑**（+15 行）：
- 检测 `SGLANG_LFC_ENABLED`：LFC 启用时不禁用 radix cache
- 检测 `SGLANG_FORCE_RADIX_CACHE`：benchmark 模式强制保留

**Bug Fix: `--max-mamba-cache-size` 覆盖**（+5 行）：
- 原代码中自动计算会覆盖用户显式传入的值
- 修复：添加 `if server_args.max_mamba_cache_size is None:` 守护

**LFC 内存初始化**（+~45 行）：
- LFC factor pool 的 GPU 显存分配
- 与现有 mamba memory 初始化集成

### 10. `python/sglang/srt/models/falcon_h1.py`

传递 `forward_batch` 参数给 Mamba2 层（+1 行）：

```python
# 修改前:
hidden_states * self.ssm_in_multiplier, mamba_hidden_states, layer_id=self.layer_id, mup_vector=...

# 修改后:
hidden_states * self.ssm_in_multiplier, mamba_hidden_states, layer_id=self.layer_id, forward_batch=forward_batch, mup_vector=...
```

这使得 Mamba2 层可以访问 `forward_batch.reqs` 进行 factor capture。

### 11. `python/sglang/srt/layers/attention/fla/lfc_reconstruct.py`（新文件）

GDN/GLA 模型的 LFC 状态重建模块（409 行）：
- `lfc_reconstruct_state()`: 主入口函数
- Triton kernel 实现（`_lfc_reconstruct_kernel`）：高效的 GPU 并行重建
- PyTorch fallback 实现：当 Triton 不可用时的回退
- 状态更新公式: `state = state * exp(g) + beta * outer(k, v)`

### 12. `python/sglang/srt/utils/catchup_timing.py`（新文件）

LFC 工具函数（395 行）：
- `is_lfc_enabled()`: 全局 LFC 开关查询
- `is_timing_enabled()`: timing 收集开关
- `TimingCollector`: per-request timing 收集器（prefix_match、ssm_state_load、lfc_reconstruction 各阶段耗时）
- `get_timing_collector()` / `reset_timing_collector()`: 线程局部存储管理

### 13. `benchmark/prefix_hit_rate_simulation.py`（新文件）

Prefix hit rate 仿真工具（608 行）：
- `SimTreeNode` / `SimRadixCache`: 模拟 radix tree 行为
- 三种匹配模式：Attention（任意节点）、Mamba（仅有 mamba_value）、LFC（有 mamba_value 或 lfc_factors）
- 支持两种数据集：`shared-prefix`（合成）和 `sharegpt`（真实多轮对话）
- 支持 mamba pool 大小限制、驱逐模拟

---

## 使用方式

### 启动 LFC 服务

```bash
# 基本启动
SGLANG_LFC_ENABLED=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
    python -m sglang.launch_server \
    --model-path tiiuae/falcon-h1-1.5b-instruct \
    --port 30000

# 带 timing 诊断
SGLANG_LFC_ENABLED=1 SGLANG_CATCHUP_TIMING=true FLASHINFER_DISABLE_VERSION_CHECK=1 \
    python -m sglang.launch_server \
    --model-path tiiuae/falcon-h1-1.5b-instruct \
    --port 30000

# 自定义显存预算
SGLANG_LFC_ENABLED=1 SGLANG_LFC_MEMORY_BUDGET_GB=4.0 FLASHINFER_DISABLE_VERSION_CHECK=1 \
    python -m sglang.launch_server \
    --model-path tiiuae/falcon-h1-1.5b-instruct \
    --port 30000
```

### 环境变量说明

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `SGLANG_LFC_ENABLED` | `false` | **主开关**。启用 LFC factor capture 和 reconstruction |
| `SGLANG_LFC_MEMORY_BUDGET_GB` | `2.0` | LFC factors 的 GPU 显存预算（GB） |
| `SGLANG_SNAPSHOT_INTERVAL` | `500` | SSM state snapshot 间隔（token 数） |
| `SGLANG_CATCHUP_TIMING` | `false` | 启用各阶段耗时收集 |
| `SGLANG_CATCHUP_TIMING_DIR` | `/tmp/sglang_catchup` | timing 数据输出目录 |
| `SGLANG_FORCE_RADIX_CACHE` | `false` | 强制保留 radix cache（benchmark 用，无需 LFC） |

### Benchmark 命令

```bash
# 真实 benchmark
python -m sglang.bench_serving \
    --backend sglang --port 30000 \
    --num-prompts 128 \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 8 --gsp-prompts-per-group 16 \
    --gsp-system-prompt-len 1024 --gsp-question-len 128 --gsp-output-len 16 \
    --request-rate 4 --disable-stream

# Prefix hit rate 仿真
python benchmark/prefix_hit_rate_simulation.py \
    --dataset shared-prefix \
    --num-groups 16 --prompts-per-group 16 \
    --system-prompt-len 2048 --question-len 128
```

---

## 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                Factor Capture (forward → tree)               │
│                                                              │
│  forward pass (mamba.py / hybrid_linear_attn_backend.py)     │
│      ↓  提取 per-token 因子 (k, v, g, beta) / (hidden, B, C, dt) │
│  req.pending_lfc_factors[layer_id] = (factors...)            │
│      ↓  请求完成后                                             │
│  cache_finished_req() → insert() → TreeNode.lfc_factors      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Reconstruction (tree → forward)                  │
│                                                              │
│  match_prefix() 匹配到有 lfc_factors 但无 mamba_value 的节点    │
│      ↓                                                       │
│  向上遍历找 ancestor (有 mamba_value 或 root)                   │
│  收集 factor_chain → 预合并为单次 kernel 调用的输入               │
│      ↓                                                       │
│  req.lfc_reconstruction_factors[layer_id] = merged_factors   │
│      ↓                                                       │
│  forward pass 中逐层重建 SSM state                             │
│  (mamba.py: mamba_chunk_scan_combined)                        │
│  (hybrid_linear_attn_backend.py: lfc_reconstruct_state)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
lfc_migration_package/
├── README.md                          ← 本文件
├── lfc_full.patch                     ← 完整 git diff patch（推荐使用此方式迁移）
├── modified_files/                    ← 所有修改/新增的完整文件
│   ├── python/sglang/srt/
│   │   ├── environ.py
│   │   ├── layers/attention/
│   │   │   ├── hybrid_linear_attn_backend.py
│   │   │   ├── mamba/mamba.py
│   │   │   └── fla/lfc_reconstruct.py        ← 新文件
│   │   ├── managers/
│   │   │   ├── schedule_batch.py
│   │   │   └── scheduler.py
│   │   ├── mem_cache/
│   │   │   ├── mamba_radix_cache.py
│   │   │   └── memory_pool.py
│   │   ├── model_executor/
│   │   │   ├── forward_batch_info.py
│   │   │   └── model_runner.py
│   │   ├── models/falcon_h1.py
│   │   └── utils/catchup_timing.py           ← 新文件
│   └── benchmark/
│       └── prefix_hit_rate_simulation.py     ← 新文件（benchmark 脚本）
└── docs/
    ├── LFC_DESIGN_AND_ANALYSIS.md            ← 完整设计与实验文档
    └── MAMBA_RADIX_CACHE_ANALYSIS.md         ← Mamba cache 0% hit rate 分析
```
