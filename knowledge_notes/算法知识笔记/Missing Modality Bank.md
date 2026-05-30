## Missing Modality Bank

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Missing Modality Bank 是 Flex-MoE 提出的用于多模态 missing modality 场景的 learnable embedding bank。给定模态集合 $\mathcal{M}$，缺失模态的所有可能组合数为 $2^{|\mathcal{M}|}-1$（不包括全模态组合）。bank $\mathbf{B} \in \mathbb{R}^{(2^{|\mathcal{M}|}-1) \times |\mathcal{M}| \times d}$ 为每种 observed modality combination 下的每个 missing modality 存储一个可学习的 embedding（维度为 d）。

其核心设计原则是：缺失模态的 embedding 不应是全局统一的 learnable vector，而应**依赖于当前样本有哪些模态被观测到**。例如，同一个缺失的 biospecimen 模态，在样本有 {Image, Clinical} 时的补充 embedding 与样本有 {Image, Genetic, Clinical} 时应不同——因为观测组合提供的上下文信息不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 输入: modality_set M = {I, C, B, G}, hidden_dim d=128
# Missing modality bank 初始化:
B = torch.randn(2^|M|-1, |M|, d)   # (15, 4, 128) for |M|=4

# MC_index: 将观测模态组合映射到 bank 行索引
# e.g., MC_index(I=1, C=1, B=0, G=0) → "IC" → index 6
#       MC_index(I=1, G=1, B=1, C=0) → "IGB" → index 3

def forward(sample_i, observed_modalities, encoders):
    embeddings = []
    mc_idx = MC_index(observed_modalities)  # 观测组合 → bank 行
    for m in M:
        if m in observed_modalities:
            e_i^m = encoders[m](sample_i[m])     # 使用真实数据编码
        else:
            e_i^m = B[mc_idx][m]                 # 从bank查找缺失embedding
        embeddings.append(e_i^m)
    return concat(embeddings)                    # (4*d,) 的完整多模态表示
```

Flex-MoE 在 ADNI 数据集上验证了 bank 的有效性：cosine similarity 分析显示"共享更多观测模态的组合有更相似的缺失 embedding"——full "ICBG" 与 "ICB" 相似度 0.56，与 "IC" 仅 0.46。去除 embedding bank 使得 ACC 从 66.11 降至 63.87。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Bank 是一个 PyTorch `nn.Parameter` 张量，随模型端到端训练。bank 索引使用位掩码：将 modality combination 编码为位掩码（如 [I=1, G=1, C=0, B=0] → 二进制 1100），转为整数索引。编码器**仅用对应 modality 被 observed 的样本训练**——避免了 traditional zero-padding/imputation 对 encoder 训练质量的破坏。bank 的参数随下游任务 loss 一起优化，学习"当特定模态缺失时，基于已有观测信息应该补充什么"。

涉及论文标题：
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
