## CLIP-MoE (Vision Encoder MoE)

术语解释
CLIP-MoE 是 CuMo 论文提出的将 Top-K 稀疏门控 MoE 块集成到 CLIP ViT（视觉编码器）每个 transformer encoder 层的设计。具体是将 CLIP ViT-L 中每层的 dense MLP（两个线性层 + GELU）替换为 Top-2-in-4 稀疏 MoE 块，保持 skip connection 不变。

术语是什么？
CLIP ViT-L 的标准结构为交替的 Multi-Head Self-Attention + dense MLP blocks。CLIP-MoE 仅替换 MLP blocks：每个 MLP 变为 1 个 Router（线性层 → Softmax → Top-K）+ 4 个 expert MLP（每个与原始 MLP 结构相同）。Router 对每个 visual token 选择 Top-2 experts，输出为 2 个选中 expert 输出的加权和。

```
# CLIP-MoE 单层 forward（替换原来 ViT 的 MLP block）
def clip_moe_layer(x):  # x: [N, d] visual tokens
    # 1. Multi-Head Self-Attention（保持不变）
    x = x + MHSA(LayerNorm(x))

    # 2. MoE block 替换 dense MLP（核心改动）
    residual = x
    x = LayerNorm(x)
    W = Softmax(Linear_router(x))           # [N, 4]
    W_K_values, W_K_indices = TopK(W, K=2) # 选 top-2 experts
    W_K = Softmax(W_K_values)              # [N, 2]

    out = zeros_like(x)
    for i in range(2):
        expert_idx = W_K_indices[:, i]
        expert_weight = W_K[:, i:i+1]
        out += expert_weight * ExpertMLP_i(x)  # 仅通过选中的 2/4 experts

    x = residual + out
    return x
```

从算法pipeline角度拆解术语：
CLIP-MoE 位于多模态 LLM pipeline 的**视觉编码阶段**。输入图像 → CLIP-MoE（每层 top-2 routing 处理 visual tokens）→ 输出 visual tokens → MLP-MoE 连接器 → LLM。CLIP-MoE 激活 0.50B / 总 0.91B 参数（仅激活 2/4 experts）。论文发现增加 experts 数量到 8（Top-2-in-8）反而略有性能下降（Table 4），推测是有限的视觉指令微调数据不足以训练 8 个鲁棒且均衡的 experts。

术语一般如何实现？如何使用？
- CLIP-MoE experts 使用 Co-Upcycling 初始化（从预微调后的 dense CLIP MLP 权重复制）
- 随机初始化 CLIP-MoE 专家 → 模型不收敛
- 训练时 unfreeze CLIP（预训练阶段 freeze）+ 降低学习率至 2e-5
- 配合 bzloss 维持 expert 负载均衡
- 推理时 expert distribution 在各层间均匀分布（Figure 5 验证）

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

---
