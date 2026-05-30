## Fused Gate / Gate Fusion（融合门控）

术语是什么？
Fused Gate（融合门控）是 MoEsaic 在 multi-tenant MoE serving 场景中提出的技术：将多个 model instance 的独立 gating network 合并为单一 fused gate kernel，单次 CUDA kernel 调用完成所有 model instance 的 routing 计算。当同时服务 N 个 model instance 时，separate gate 需逐 model 串行调用 N 次 CUDA kernel，路由开销随 N 线性增长；fused gate 将所有 routing 请求合并为一次批量调用，实现接近常量级的 routing 延迟。

从系统架构角度拆解术语：
在 MoEsaic 的 vLLM 推理循环中（每层 MoE layer），Fused Gate 的工作流程：

1. **请求抵达**：每层 MoE layer 前，MoEsaic 收集所有请求的 hidden states X ∈ ℝ^(B×H) 和对应的 model_id 列表。
2. **Batched Gate Computation**：Fused gate 在一次 CUDA kernel 调用中对所有 model instance 执行 gating 计算。内部组织为 per-model 的 batch matrix multiplication：对每个 model instance i，执行 Softmax(W_gate^i · X[model_i]) → TopK 选择 experts。
3. **Gate Mapping Table**：MoEsaic 维护每个 model instance 的 gate mapping 表——将原始 expert ID 映射到去重后的 merged expert ID。fused gate 的输出经 mapping 表转换后，所有请求被路由到正确的共享专家。
4. **Token-to-Expert Dispatch**：路由结果按 merged expert ID 将 batch 中所有 token 分配至对应 expert。

图 3（论文 Figure 3）展示了 2 model instances 下 separate gate vs fused gate 的对比：separate gate 中每个 model 独立调用 gate → 各选出 top-2 experts；fused gate 中 2 个 gate 合并为 1 次调用 → 批量输出所有 routing 结果 → gate mapping 映射到 merged experts。

术语一般如何实现？如何使用？
- 在 vLLM 中实现为 per-layer 的 fused gate module。每个 MoE layer 中，所有 model instance 的 gate weight 矩阵 W_gate^i 被组织为 batched linear 操作。
- 对小型模型（Mixtral-4x1B，expert 计算时间短）的收益最显著——separate gate 路由延迟每增加一个 model 增长 8%，fused gate 降至 4%。
- 对大型模型（Mixtral-4x7B，expert 计算占主导），fused gate 的增量收益较小（路由时间远小于 expert 计算时间），但仍保持约 8% 常数开销 vs baseline 单模型。
- Fused gate 与 vLLM 的 FusedMoE layer 概念不同：vLLM 的 FusedMoE 是指将多个 expert 的 FFN 计算融合为 single Triton kernel；MoEsaic 的 Fused Gate 是指将多个 model 的 gating 计算融合为 single CUDA kernel 调用。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
