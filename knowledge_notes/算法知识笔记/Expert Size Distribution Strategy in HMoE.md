## Expert Size Distribution Strategy in HMoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Size Distribution Strategy 是 HMoE 中定义各 expert 的 hidden dimension 相对大小的策略，直接决定异构 MoE 中 expert 之间的容量差异程度。HMoE 提出三种分布策略：

1. **Arithmetic Strategy（算术级数）**：expert 大小按等差序列分布，如相对比例 {9, 11, 13, 15, 17, 19, 21, 23}。特点：相邻 expert 容量差恒定，总差异相对温和（最大/最小≈2.5×），小 expert 仍有足够能力参与训练。HMoE 主实验采用此策略，训练最稳定。

2. **Geometric Strategy（几何级数）**：expert 大小按等比序列分布，如 {1, 2, 4, 8, 16, 32, 64, 128}。特点：容量差异极大（最大/最小=128×），突出关键 expert 的作用。但实验表现最差——极小 expert 缺乏足够容量，即使 P-Penalty loss 也无法充分激活它们（Figure 8 right）。

3. **Hybrid Strategy（混合策略）**：结合同构与异构，如 {1, 1, 1, 1, 2, 2, 4, 4}。特点：部分 expert 共享相同大小（形成"功能组"），组间有容量差异。假设某些场景需要多个相似能力的 expert 协同工作。实验表现优于 arithmetic（Figure 8 left），说明适量的"相似 expert 组"结合"组间异构"可能是最优方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

三种策略的 expert hidden dim 计算（以 HMoE-3B 为例，总 hidden=32768，8 experts）：

```python
# 归一化比例 → 实际 hidden dim 映射
total_sum = sum(ratio_list)  # e.g., arithmetic: 9+11+...+23 = 128
scale = total_hidden_dim / total_sum  # 32768 / 128 = 256
expert_dims = [r * scale for r in ratio_list]

# Arithmetic Strategy: 等差
# {9,11,13,15,17,19,21,23} × 256
# = {2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888}

# Geometric Strategy: 等比 (ratio=2)
# {1,2,4,8,16,32,64,128} × (32768/255)
# ≈ {128, 257, 514, 1028, 2056, 4112, 8224, 16448}
# 问题：expert_0 (128 dim) 容量过小，几乎无建模能力

# Hybrid Strategy: 分组
# {1,1,1,1,2,2,4,4} × (32768/16) = 
# {2048, 2048, 2048, 2048, 4096, 4096, 8192, 8192}
```

HMoE 进一步实验了 arithmetic 策略在不同方差下的表现（Figure 11）：改变最大/最小 expert dim ratio 从 1:1（完全同构）到约 2.5:1（主实验设置）再到更大的比例，发现 loss 先上升后下降，存在一个最优异构度——ratio 约 2.5:1 时 loss 最低，验证了适度异构优于极端异构和完全同构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Expert 大小分布在模型初始化时设定，通过定义各 expert 的 FFN 权重矩阵维度（W_g, W_p 的 out_dim, W_o 的 in_dim）实现。HMoE 基于 LLaMA 架构，每层 8 个 expert。使用时根据任务场景选择策略：(1) 需要最大化大 expert 能力且能接受训练不稳定 → geometric；(2) 需要稳定训练且平衡所有 expert 的参与度 → arithmetic；(3) 需要"功能组"协作（部分 expert 冗余）→ hybrid。未来方向可能包括可学习的异构度（训练中自适应调整 expert 大小）。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
