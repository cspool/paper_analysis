## Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

- baseline方法是什么？
  - Baseline 是标准 top-k MoE 路由（如 Qwen3 默认的 top-8 out of 128 experts），在 decode 阶段每个 token 独立激活 k=8 个 router 得分最高的专家。
  - 全栈执行例子（Qwen3-30B，B=16，decode 阶段）：
    - **算法层**：Router R 对每 token 输出 N=128 维 softmax 分布，取 top-8 索引。batch 中 16 个 token 各选 8 个专家，理论上 T（唯一激活专家数）在 8~128 之间，期望值约 82 个（公式 N(1-(1-k/N)^B)）。
    - **Serving框架层**：SGLang 将 16 个 token 的 decode batch 送入 MoE 层。Router 评分后，每个 token 的 top-8 专家索引确定。SGLang 聚合所有被选中的专家权重，调用 Grouped GEMM kernel 加载权重到 SRAM 并计算。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：Grouped GEMM（cuBLAS）批量执行不同专家的 (2048×768) 矩阵乘法。对于 memory-bound 的 decode，每个专家权重加载（b 项）主导总延迟（b·T + a·Bk），T≈82 时延迟 ~175μs。
    - **硬件架构层**：H100 HBM→SRAM 带宽是瓶颈。每个专家 3 个权重矩阵（SwiGLU 的 3 个 2048×768 GEMM），加载 82 个专家的权重耗时远大于计算时间。
  - Baseline 缺陷：在中等 batch size（如 16）下，MoE 层处于 memory-bound 状态。因为每个 token 仅激活 k=8/N=128 个专家（稀疏因子 16×），平均每专家负载仅 B·k/N=1 token，远低于 compute-bound 所需的大量 tokens。延迟被"加载所有被激活专家的权重"主导（T 项），而非计算量（Bk 项）。T 随 batch size 快速增长（batch=1 时 T=8，batch=16 时期望 T=82），导致 decode 延迟恶化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：OEA（Opportunistic Expert Activation），一种无需重新训练的 batch-aware 动态路由算法，通过两阶段策略最小化 batch 内唯一激活专家数 T。
    - **Phase 1（Baseline Expert Selection）**：每 token 仅激活 top-k0（k0 < k）个最关键的专家，构成 S_base，保证每 token 的独立质量底线。关键洞察：top 排名的专家对输出质量最为关键（Gupta et al., 2024）。
    - **Phase 2（Opportunistic Piggybacking）**：每 token 从 S_base（Phase 1 中所有 token 已激活的专家集合）中寻找其 k0+1 到 k 位的"次优先"专家，若能找到则免费附加上去（因为这些专家权重已在 SRAM 中）。这保持了 T = |S_base| 不变，仅增加计算量（可忽略，因 memory-bound）。
  - **对比 baseline 全栈执行例子**：
    - **算法层**：OEA 替换 top-8 为 top-k0 + piggybacking（k0=5 时）。Phase 1 每 token 只保证 5 个基线专家，Phase 2 有机会用已在 S_base 的专家补到 8 个。每 token 最终仍激活约 8 个专家（质量不降），但 T = |S_base| ≈ 35（vs. baseline 约 48）——仅 Phase 1 控制 T 的规模。
    - **Serving框架层**：在 SGLang 的 MoE decode 路径中插入 OEA 路由器。OEA 先统计 S_base 再分配 piggybacking。路由仅在 decode 阶段使用（prefill 已足够 compute-bound）。额外修改：捕获 CUDA Graph 到 batch size 16 以避免 SGLang padding 引入无用专家。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：Grouped GEMM 加载 T≈35 而非 T≈48 个专家的权重。b 主导延迟，T 降低 27% 意味着 ~23% 的延迟降低（k0=5 时 175.7→136.0μs）。
    - **硬件架构层**：HBM 带宽压力降低 ~27%。更少的权重加载意味着每个 decode step 更快完成，latency 从 175.7μs 降至 136.0μs（k0=5, 23% reduction）和 106.8μs（k0=3, 39% reduction），且准确率无统计显著退化。
