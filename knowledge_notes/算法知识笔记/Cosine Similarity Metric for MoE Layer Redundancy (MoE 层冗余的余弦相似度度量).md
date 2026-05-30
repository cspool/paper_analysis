## Cosine Similarity Metric for MoE Layer Redundancy (MoE 层冗余的余弦相似度度量)

术语解释
Cosine Similarity Metric for MoE Layer Redundancy 是一种无训练的度量方法，通过计算 Transformer block 的输入输出隐状态之间的 cosine similarity 来评估层的冗余程度，进而决定哪些层/块可以被安全移除。由 He et al. (2025) 提出，用于 MoE 模型的 Layer Drop 和 Block Drop。

术语是什么？
核心思想：如果一层的输出与输入高度相似（cosine similarity ≈ 1），则该层对 token hidden state 的"变换"很小，可能是冗余的。具体定义为：

- S^(M) = (x · y) / (||x|| · ||y||)，where y = MoE(x)：仅 MoE 层的输入输出相似度
- S^(NM) = (x' · y') / (||x'|| · ||y'||)，where y' = x' + MoE(Norm(x'))：含 Norm + MoE + residual connection 的完整效果

论文发现 S^(NM) 比 S^(M) 更能准确反映层冗余度，因为在 Transformer 中 Norm 和 residual connection 是关键组件。单独的 S^(M) 相似度低（移除仅 MoE 不可行），但 S^(NM) 相似度高（Norm+MoE+Residual 整体冗余，可移除）。

从算法pipeline角度拆解术语：
```
# 计算流程
def compute_layer_similarity(model, calibration_data, device):
    for each layer l in range(L):
        similarities = []
        for each batch x in calibration_data (128 samples, seq_len=2048):
            # 记录 block 输入
            x_input = x.clone()  # 残差连接前的 hidden state

            # Forward through Norm + MoE + Residual
            x_norm = layer_norm[l](x)
            x_moe = moe_layer[l](x_norm)
            x_output = x_input + x_moe  # residual connection

            # 计算相似度
            sim = cosine_similarity(x_input.flatten(), x_output.flatten())
            similarities.append(sim)

        S_l = mean(similarities)  # 层的平均冗余度

    # 按 S 降序排列 → 高相似度层优先被 drop
    drop_order = argsort(S, descending=True)
    return drop_order

# cosine_similarity(a, b) = (a·b) / (||a||·||b||)
```

鲁棒性验证：
- 样本数：32→128→1024 samples from C4，相似度模式稳定
- 数据集：C4 (pretraining)、Lima (instruction tuning)、MetaMathQA (math) 三种分布的相似度模式一致 → 度量对数据分布鲁棒

术语一般如何实现？如何使用？
- Calibration: 128 random samples from C4, sequence length=2048，sufficient for stable similarity estimation
- 应用流程: 加载模型 → forward calib data 收集每层 hidden states → 计算每层平均 S^(NM) → 按 S^(NM) 排序 → 移除 Top-K 层/块
- 与 MoE routing 的关系：S^(NM) 仅依赖 hidden states，不依赖 router 决策；drop 后剩余层的 router 无需修改
- Drop 模式发现：深层 layers/blocks 优先被 drop（与 Xu et al., ShortGPT, Men et al., 2024 一致），因为深层主要负责高层语义抽象，冗余度更高
- 拓展：可推广到 dense 模型，但 MoE 模型 dropping 后性能衰减更小（same depth: MoE -7.0 vs Dense -24.3 MMLU at 8 layers dropped）

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework
