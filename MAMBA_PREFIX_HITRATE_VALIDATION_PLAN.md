# Mamba/Attention Prefix Hit Rate 完整验证实验方案（Qwen3-Next-80B-A3B-Instruct）

## 1. 目标与要验证的结论

我们希望**完整验证**以下命题是否成立：

1. 在“共享长前缀 + 分叉后缀”（典型 shared system prompt）场景中，
   - Attention 机制存在显著可复用前缀；
   - Mamba（SSM）机制可复用前缀显著更低（可能接近 0）。
2. 在“append-only 多轮对话”场景中，Mamba 前缀复用能力显著改善，并接近 Attention。
3. 二者差异会反映到端到端性能（TTFT / 吞吐）上。

主实验模型：`Qwen/Qwen3-Next-80B-A3B-Instruct`

---

## 2. 实验总体设计

采用三层验证：

- **L1 黑盒验证（无需改代码）**：通过 API `cached_tokens`、TTFT、吞吐验证“整体缓存效果是否存在/消失”。
- **L2 白盒机制验证（最关键）**：增加最小埋点，分别统计 `Attention 可命中 token` 与 `Mamba 实际命中 token`。
- **L3 稳健性验证**：多数据集、多负载点、多次重复，给出置信区间与显著性判断。

> 说明：仅靠 `cached_tokens` 无法区分 Attention 与 Mamba 的贡献，必须做 L2 埋点才能“完全验证”。

---

## 3. 数据集与负载场景

### 3.1 场景 A：共享前缀（核心场景）

目标：验证“同一系统提示词 + 不同用户问题”时的分层 hit rate。

建议两种数据：

1. **generated-shared-prefix（可控基线）**
   - 使用仓库现有 `benchmark/hicache/bench_serving.py` 的 `generated-shared-prefix`。
   - 控制变量方便，便于先确认趋势。
2. **真实语料共享前缀（生产近似）**
   - 从真实数据集抽用户问题（如 ShareGPT/UltraChat 首轮 user turn）。
   - 给所有样本统一加同一 system prompt（长度建议 1k~4k token）。
   - 形成“真实用户分布 + 共享系统前缀”的 workload。

### 3.2 场景 B：append-only 多轮对话（对照场景）

- 使用 `sharegpt` + `--enable-multiturn`。
- 该场景用于验证：当请求主要是“历史会话追加”而非“共享前缀分叉”时，Mamba hit 是否恢复。

### 3.3 负载点

每个场景至少跑 3 个请求速率点：

- 低负载：2 req/s
- 中负载：4 req/s
- 高负载：8 req/s

每个点重复至少 3 次（推荐 5 次）用于置信区间估计。

---

## 4. 指标定义（必须统一口径）

对每个请求 i：

- `eligible_prefix_i`：可用于 prefix matching 的 token 数（通常 `prompt_tokens - 1`，与调度逻辑一致）。
- `hit_mamba_i`：实际被调度复用的前缀 token（等价于 `len(prefix_indices)`，也是当前 `cached_tokens` 的主要来源）。
- `hit_attn_i`：若忽略 Mamba 状态约束，Attention/KV 理论上可复用的最长前缀 token。

聚合指标：

- `Mamba Token Hit Rate = Σ hit_mamba_i / Σ eligible_prefix_i`
- `Attention Token Hit Rate = Σ hit_attn_i / Σ eligible_prefix_i`
- `Request Hit Rate(mech) = 命中请求数 / 总请求数`（命中定义：`hit_x_i > 0`）

同时记录：TTFT、TPOT、总吞吐、失败率。

---

## 5. 关键埋点方案（L2，最小改造）

## 5.1 需要新增/导出的字段

在每个请求上输出以下统计到 JSONL（或 metrics）：

- `eligible_prefix_tokens`
- `mamba_hit_tokens`（当前真实命中，来自 `prefix_indices`）
- `attn_potential_hit_tokens`（忽略 mamba_value 约束时的最长 key match）
- `dataset_name`, `scenario`, `request_id`, `group_id`（可选）

## 5.2 推荐实现点

1. `python/sglang/srt/mem_cache/mamba_radix_cache.py`
   - 在 `match_prefix()` 中，除现有返回外，补充 `attn_potential_hit_tokens`。
   - 计算方式：基于当前匹配路径得到“纯 key 匹配长度”（不要求 `mamba_value != None`）。
2. `python/sglang/srt/managers/schedule_batch.py`
   - 在请求对象上保存上述字段（`Req` 级别）。
3. 输出路径（二选一）
   - A: 写入 benchmark 脚本结果 JSONL（推荐，便于分析）；
   - B: 暴露到 `meta_info`（便于在线采集）。

> 注意：`MatchResult` 已有 `mamba_branching_seqlen` 字段预留，可直接复用语义，避免引入新结构。

---

## 6. 实验矩阵（最少）

固定模型：`Qwen/Qwen3-Next-80B-A3B-Instruct`

每个场景跑以下组合：

1. **Baseline**：`--disable-radix-cache`
2. **Radix On**：默认 radix cache（MambaRadixCache 生效）

可选补充：

3. **Instrumentation only**：与 Radix On 同配置，仅增加埋点，不改调度行为。

每组记录：

- 黑盒：`cached_tokens`, TTFT, throughput
- 白盒：`mamba_hit_tokens`, `attn_potential_hit_tokens`

---

## 7. 可执行命令模板

## 7.1 启动服务（示例）

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 8 \
  --host 0.0.0.0 --port 30000
```

Baseline（关闭 radix）:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 8 \
  --disable-radix-cache \
  --host 0.0.0.0 --port 30000
```

## 7.2 场景 A：generated shared prefix（可控）

```bash
python benchmark/hicache/bench_serving.py \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name generated-shared-prefix \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --num-prompts 256 \
  --gsp-num-groups 16 \
  --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 \
  --gsp-question-len 128 \
  --gsp-output-len 128 \
  --request-rate 4 \
  --output-file results_gsp_rr4.jsonl
```

## 7.3 场景 B：ShareGPT 多轮（append-only 对照）

```bash
python benchmark/hicache/bench_serving.py \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name sharegpt \
  --dataset-path /path/to/sharegpt.json \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --enable-multiturn \
  --num-prompts 200 \
  --fixed-output-len 128 \
  --request-rate 4 \
  --output-file results_sharegpt_rr4.jsonl
```

> 对 2/4/8 req/s 分别执行；Baseline 与 Radix On 各跑一遍。

---

## 8. 数据分析与验收标准

## 8.1 主结论判定

在共享前缀场景（A）满足以下条件则认为“结论成立”：

1. `Attention Token Hit Rate` 显著大于 0（例如 > 50%，具体阈值按数据长度而定）；
2. `Mamba Token Hit Rate` 显著低于 Attention（理想情况下接近 0）；
3. Radix On 相比 Baseline 的 TTFT 改善与 `mamba_hit_tokens` 变化一致。

在 append-only 场景（B）应观察到：

4. `Mamba Token Hit Rate` 明显高于场景 A，且接近 Attention。

## 8.2 统计方法

- 对每个场景/负载点，做 3~5 次重复，报告 mean ± 95% CI。
- 对 `Attention Hit Rate - Mamba Hit Rate` 做 bootstrap CI。
- 若 CI 全部大于 0，则可认定差异稳定存在。

---

## 9. 风险与控制变量

- 固定 `seed`、`fixed_output_len`、`request-rate`，减少抖动。
- 关闭与本实验无关的变量（如 speculative decoding、不同采样参数）。
- 每次正式计时前先 warmup 一轮。
- 明确记录硬件、驱动、commit hash、启动参数。

---

## 10. 交付物清单

1. 实验配置清单（模型、参数、数据集版本、commit hash）
2. 原始结果 JSONL（每请求）
3. 汇总表（按场景/负载）
4. 两张核心图：
   - `Attention vs Mamba Token Hit Rate`（分场景）
   - `TTFT`（Baseline vs Radix On，分场景）
5. 结论文档（是否验证通过 + 反例/边界条件）

---

## 11. 推荐执行顺序（实操）

1. 先跑场景 A 的 `generated-shared-prefix`（最快发现机制差异）。
2. 再跑场景 B 的 `sharegpt --enable-multiturn`（验证对照）。
3. 最后替换成“真实语料共享前缀”数据，复现实验结论并给出生产相关性。

这样可以先用可控场景验证机理，再用真实场景验证外部有效性。