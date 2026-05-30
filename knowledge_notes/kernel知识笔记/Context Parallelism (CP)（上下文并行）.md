## Context Parallelism (CP)（上下文并行）

术语是什么？
Context Parallelism (CP) 是一种将长序列的 token 沿序列维度切分到多个 GPU 上的并行策略，使每个 GPU 仅处理序列的一个子段。CP 的核心通信原语是 All-to-All 风格的序列重分布（如 DeepSpeed-Ulysses 的 all-to-all 或 Ring Attention 的 P2P 环形传递）。在 MoE 模型训练中，CP 与 EP 的组合尤为关键——当 CP×EP group 超过 NVLink 域时，通信开销会急剧增加。

从 kernel 调度角度拆解术语：
以 DeepSpeed-Ulysses 风格的 Sequence Parallelism 为例（在 Attention 层替代 TP）：

1. **输入切分**：将长度为 S 的序列沿序列维度切分为 cp_size 份，每 GPU 处理 S/cp_size 个 token
2. **All-to-All**：将 (S/cp_size, d) 的输入从序列分片转为 head 分片 → (S, d/cp_size)
3. **Attention 计算**：各 GPU 在 head 子集上独立执行 QKV projection + attention
4. **All-to-All**：将 head 分片转回序列分片 → (S/cp_size, d)
5. **输出 MLP**：各 GPU 计算其子序列的 output projection

CP 的通信量 = 2 × bsh (n-1)/n × (2 + 2/m)/n（m 为 GQA group 数），当 m=4 时约为 TP 的 1/4。

在 MoE Parallel Folding 中的关键作用：
- CP 切分的序列在进入 MoE 层前通过 reshape 展平为 token batch（零通信开销）
- MoE Parallel Folding 允许 CP 和 EP 组折叠在一起，使 EP 的 All-to-All 优先使用 NVLink 而非 InfiniBand
- 当 CP×EP > 8 时，无 Folding 的 EP A2A 走跨节点 InfiniBand，延迟显著上升；Folding 后保持稳定

术语一般如何实现？如何使用？
- 适用于序列长度 > 8192 tokens 的场景（论文中测试至 128K）
- Megatron-Core 中通过 context_parallel_size 参数配置
- 与 TP/EP 组合使用时需注意通信域重叠：理想情况下 CP×EP ≤ node_size（保持在 NVLink 域内）

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
