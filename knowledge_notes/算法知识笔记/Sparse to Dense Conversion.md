## Sparse to Dense Conversion

术语解释
稀疏到稠密转换是将MoE稀疏模型转换为同等结构的稠密模型的技术，以消除动态路由开销和expert管理的复杂性，在推理时将MoE模型"压缩"为高效的稠密模型。

术语是什么？
稀疏到稠密转换适用于稠密模型部署更优的场景（如缺少MoE优化框架支持的设备）：
- **XFT**：生成sparse-upcycled MoE模型，再通过可学习的合并机制转换回同等大小和结构的稠密LLM
- **Switch Transformers**：蒸馏将稀疏模型转稠密，97%参数压缩后保留30%+性能
- **OneS**：Knowledge Gathering（求和/平均/top-k/SVD聚合）→ Knowledge Distillation两阶段
- **EWA**：训练时用MoE替代FFN，推理时恢复为稠密ViT
- **AdaMoLE**：结合LoRA结构的专用网络，根据不同任务复杂度调整激活阈值

从算法pipeline角度拆解术语。
```
# Sparse → Dense Pipeline
# Stage 1: Knowledge Gathering
W_dense = aggregate_experts_to_dense(MoE_model)
# 聚合方式：
# - Sum: W_dense = Σ_i W_i
# - Average: W_dense = mean(W_i)
# - Top-K: W_dense = Σ_i topk_weight(i) * W_i
# - SVD: W_dense = SVD_reconstruct([W_1, ..., W_N])

# Stage 2: Knowledge Distillation
for x in dataset:
    y_moe = MoE(x)      # 原MoE教师
    y_dense = Dense(x)  # 聚合后的稠密学生
    loss = KL_div(y_moe, y_dense) + task_loss(y_dense, labels)
    update(Dense)
```
转换后的稠密模型参数量更小（相当于原MoE的激活参数量级），推理路径更简单（无router、无expert选择开销）。

术语一般如何实现？如何使用？
- Knowledge Gathering阶段将专家知识合并到单一FFN
- 蒸馏阶段微调稠密模型以恢复MoE性能
- 适用于需要简单部署的场景（移动端、边缘设备）
- 牺牲了MoE的"无限"扩展能力，但获得了部署简便性

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models

---
