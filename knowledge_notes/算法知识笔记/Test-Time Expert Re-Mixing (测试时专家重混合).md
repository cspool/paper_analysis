## Test-Time Expert Re-Mixing (测试时专家重混合)

术语解释
Test-Time Expert Re-Mixing 是 C3PO 提出的 MoE LLM 测试时自适应范式：在推理阶段，不修改任何模型参数，仅通过优化 expert routing weights（专家路由权重）来重新混合各层的 expert 贡献比例，使每个测试样本获得定制化的 expert pathway。核心优势：优化变量极少（几十到几百个 routing weights vs prompt tuning 的数千维 token embeddings），且部分方法无需反向传播。

术语是什么？
给定预训练的 MoE LLM，其每层有一个 router（gate）计算 expert 的选择概率。Test-Time Expert Re-Mixing 在推理时将 routing weights 视为可优化变量 ω ∈ R^{L×E}（L 层数，E 专家数），通过最小化 surrogate objective 来调整 ω，使得模型在测试样本 x 上的输出 f(x, ω) 更准确。

C3PO 发现预训练的 end-to-end router 存在严重的次优性（sub-optimality）：base model 与 Oracle（使用 ground truth 找到的最优 routing）之间存在 10-20% accuracy gap。这表明仅靠 pretraining 的 router 无法为每个样本找到最优 expert 组合，尤其是对于困难样本或分布外样本。

三种 Re-Mixing 方法：
1. **Mode Finding (Meanshift)**: 在 pathway 权重空间中找到邻居样本 pathway 的最密集区域，将当前 ω 向该区域移动
2. **Kernel Regression**: 用邻居样本 pathway 的核加权平均作为估计值，与原始 ω 插值
3. **Neighborhood Gradient Descent (NGD)**: 用邻居样本 loss 的加权平均作为 surrogate objective，梯度下降优化 ω（性能最强）

从算法pipeline角度拆解术语：
Test-Time Expert Re-Mixing 的完整流程（以 NGD 为例）：

```
# 输入: 测试样本 x, MoE模型 f, 参考集 {(x_i, y_i, ω_i)}
# 超参数: k=3 (邻居数), steps=10, lr cosine 1e-2→1e-5, Gaussian kernel

# Phase 1: 嵌入与检索
emb_x = NV-Embed-V2(task_description(x))     # 用任务描述获取嵌入
emb_ref = {NV-Embed-V2(task_description(x_i))} # 参考集样本嵌入
N = kNN(emb_x, emb_ref, k=3)                 # 检索 k=3 个最近邻

# Phase 2: 初始 pathway 提取
ω_0 = f.get_routing_weights(x)               # shape: [L, E] 或 [L, E, T]
ω = ω_0[last_5_layers, top_20_experts]       # Critical-Layer + Core-Expert 裁剪

# Phase 3: NGD 迭代优化
for step in range(10):
    total_loss = 0
    total_weight = 0
    for i in N:
        K_val = exp(-||emb_x - emb_i||^2 / (2 * σ^2))  # Gaussian kernel
        logits_i = f.forward(x_i, routing_override=ω)   # 替换 routing weights
        loss_i = cross_entropy(logits_i, y_i)
        total_loss += K_val * loss_i
        total_weight += K_val
    
    surrogate_loss = total_loss / total_weight
    grad = ∇_ω surrogate_loss
    lr = cosine_schedule(step, 10, 1e-2, 1e-5)
    ω = ω - lr * grad

# Phase 4: 推理
output = f.forward(x, routing_override=ω)
```

关键设计选择：
- 仅优化最后 1 个 token 的 routing weights（而非所有 token），因最后一个 token 承载最多的任务决策信息
- 仅优化最后 5 层的 routing weights（而非全部 16 层），深层负责任务特定的精炼
- 仅优化 top-20 experts 的 routing weights（而非全部 64 个），覆盖最终激活的 top-8 experts 的 99.8%

术语一般如何实现？如何使用？
- **实现**: 替换 HuggingFace transformers 中的 `olmoe_modeling.py`，在 MoE 层的 forward 中注入优化后的 routing weights。主优化逻辑在 `olmoe_optimizer.py` 中。
- **依赖**: PyTorch, CUDA 12.3, Python 3.10, NV-Embed-V2 (embedding model)
- **参考集构建**: 需要为每个 benchmark 准备参考集——收集模型输出正确的样本及其对应的 expert pathway。参考集与 benchmark 不同但领域相关（如 MMLU 用 BIG-Bench + SuperGLUE 作参考集）
- **适用场景**: 任何 MoE LLM（OLMoE, DeepSeekMoE 等），只需替换 routing weights 的注入逻辑
- **开源**: https://github.com/tianyi-lab/C3PO (Apache-2.0, COLM 2025)

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
