## Unbalanced Pipeline Parallelism（非均衡流水线并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Unbalanced Pipeline Parallelism 是 Skywork-MoE 提出的一种 Pipeline Parallelism (PP) 优化策略，打破传统的均匀层分割（每 PP stage 分配相同数量的 transformer 层），采用非均匀分割以减少 pipeline bubble time。其核心动机：由于最后一个 PP stage 除了正常的 transformer 层计算外还需要处理 loss calculation（包括 logits projection、cross-entropy 计算等），导致该 stage 成为计算瓶颈，增加 bubble time。通过将最后一 stage 的 transformer 层数减少（例如从 6 层减少到 4 层），可以补偿 loss calculation 的额外计算，实现更好的 stage 间负载均衡。Skywork-MoE 实验显示，将 24 层模型从均匀 4-stage [6,6,6,6] 改为非均匀 5-stage [5,5,5,5,4] 可以减少 pipeline bubble time 约 10%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 24 层 transformer, 4-stage PP 对比 5-stage unbalanced PP 为例：

```
# === 均匀 4-stage PP: [6, 6, 6, 6] ===
# Stage 0: layers 0-5   (6 layers)
# Stage 1: layers 6-11  (6 layers)
# Stage 2: layers 12-17 (6 layers)
# Stage 3: layers 18-23 + Loss (6 layers + loss calc)
# 问题: Stage 3 的 loss calculation 使其成为 bottleneck
# Bubble time 较大

# === 非均匀 5-stage PP: [5, 5, 5, 5, 4] ===
# Stage 0: layers 0-4   (5 layers)
# Stage 1: layers 5-9   (5 layers)
# Stage 2: layers 10-14 (5 layers)
# Stage 3: layers 15-19 (5 layers)
# Stage 4: layers 20-23 + Loss (4 layers + loss calc)
# 优势: Stage 4 少 1 层 → 节省的时间补偿 loss calculation
# Bubble time 减少约 10%

# 配合差异化梯度重计算 (Gradient Checkpointing):
# Stage 0-3: 正常 checkpointing 配置 (buffer 较小)
# Stage 4: 减少 checkpointing (buffer 较大, 因层数少)
# 进一步平衡各 stage 的显存使用
```

Skywork-MoE 在 146B/52 层模型上使用 12-way unbalanced PP。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 Skywork-Megatron 框架：(1) 在 PP 配置中指定非均匀的层分割方案；(2) 为每个 PP stage 差异化配置 gradient checkpointing（activation recomputation）策略，buffer 大的 stage 减少 checkpointing 以平衡显存；(3) 根据实际 profiling 结果调整各 stage 的层数分配。该技术是 PP 的通用优化，可应用于任何使用 PP 的 transformer 训练，不仅仅是 MoE 模型。

涉及论文标题：
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models
