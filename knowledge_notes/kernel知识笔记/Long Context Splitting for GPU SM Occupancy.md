## Long Context Splitting for GPU SM Occupancy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Long Context Splitting 是 FastTree 中解决 GPU SM 欠饱和问题的 runtime 优化技术。在 tree-structured attention kernel 执行中，可能出现两种 GPU 利用率不足的情况：(1) group 级并行度不足——context-queries groups 数量少于 GPU 可容纳的 block 数，部分 SM 空闲；(2) tail effect——部分 node 的 context length 极长（如 root node），对应 block 的执行时间远长于其他 block，最后几波执行中仅少数 block 活跃。Long context splitting 通过将超长 context node 沿 context dimension 切分为多个子 context，增加 group 数量和 block parallelism，使 GPU SM 充分填充。虽然 context splitting 会引入 intermediate result reduction overhead（与 FlashAttention 的 split-KV 模式类似），但实验证明当 GPU SM 欠饱和时，occupancy 改善带来的加速完全覆盖 reduction overhead。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Long context splitting 的决策与执行流程：

```
// === Problem Detection ===
Input: grouping_plan, GPU_SM_count, max_blocks_per_SM

total_blocks = 0
max_context_len = 0
for each group g in grouping_plan:
    total_blocks += ceil(g.nQ / T_q)      // blocks per group
    max_context_len = max(max_context_len, len(g.ctx))

max_concurrent_blocks = SM_count * max_blocks_per_SM

// Case 1: Insufficient group-level parallelism
if total_blocks < max_concurrent_blocks:
    need_split = true

// Case 2: Tail effect (long context dominates)
if max_context_len > threshold_long:
    // 少数 blocks 处理极长 context → tail waves 中仅少数 blocks 活跃
    need_split = true

// === Splitting Execution ===
if need_split:
    for each group g with len(g.ctx) > threshold:
        // Split context along context dimension
        n_splits = ceil(len(g.ctx) / max_context_per_split)
        for i in 0..n_splits:
            ctx_i = g.ctx[i*max: (i+1)*max]
            // Each split becomes separate group with same queries
            new_groups.append((ctx_i, g.queries))

    // Re-launch attention kernel with more groups
    // → More blocks → higher SM occupancy
    // → Reduction kernel combines results from splits
```

Timeline diagram (Mermaid Gantt) — before vs after splitting:

```
Before splitting (N=[1,10], C=[4000,400]):
  SM0:  [==================  Group 0 (ctx 4000) ===============]
  SM1:  [== G1 ==]
  SM2:  [== G2 ==]
  ...   (SMs 3-131 idle)
  SM132:[== G10 ==]
  → Tail effect: last waves only 1 block active

After splitting context 4000 into 4×1000:
  SM0:  [==== G0a(1000) ====][==== G0c(1000) ====]
  SM1:  [==== G0b(1000) ====][==== G0d(1000) ====]
  SM2:  [== G1 ==]
  ...
  → All SMs utilized across all waves
  → Reduction overhead << occupancy improvement
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FastTree 在 runtime 阶段执行 long context splitting——在 greedy heuristic 生成 grouping plan 后，检查 total_blocks 和 max_context_len，若触发 split 条件则修改 grouping plan 后重新 launch kernel。Split 阈值通过 profiling 确定（与 GPU 特定的 SM count、shared memory size 和 max blocks/SM 相关）。在 N=[1,10], C=[4000,400] 等配置下，splitting 带来 up to 1.9× speedup。该技术是 FlashAttention split-KV 思想的 tree-aware 泛化——FlashAttention 中 split-KV 用于处理超长 single sequence，FastTree 将其扩展为处理 tree 中个别超长 node。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference
