## Function Preserving Initialization (FPI)

术语解释
FPI 是一种神经网络权重扩展方法，在扩展模型宽度（hidden dims）时使大模型的输出与输入的关系与小模型完全一致，从而保留小模型学到的知识。最初由 Net2Net (Chen et al., ICLR 2016) 提出，bert2BERT (ACL 2022) 将其扩展到 Transformer 语言模型的训练。

术语是什么？
FPI 的核心思想：扩展 MLP 层 y = U^T · W^T · x 的维度时，将新添加的神经元复制已有神经元的值，通过缩放保持输出等值。当输入 dim 从 2→3、intermediate dim 从 3→4、输出 dim 从 2→3 时：(1) Input Dim Expansion: 复制输入神经元，拆分权重；(2) Output Dim Expansion: 复制 hidden 神经元；(3) MLP Expansion: 复制输出神经元。

但 FPI 存在内在缺陷：复制操作导致对称权重（W'_1 = W'_2），梯度在训练中始终相同，有效参数量减半。具体来说，如果 y = w1x + w2x 且 w1=w2 初始化后，w1 和 w2 的梯度完全相同，永远无法分化。

从算法pipeline角度拆解术语：
```
# FPI 宽度扩展（MLP y = U^T · W^T · x）
# 源模型：d_in=2, d_inter=3, d_out=2
# 目标模型：d_in=3, d_inter=4, d_out=3

# Step 1: Input Dim Expansion
W_new = FPI_expand(W, d_in_new=3)
# w'_1 = w_1/2,  w'_2 = w_2/2,  w'_3 = w_1/2  # 复制并缩放

# Step 2: Output Dim Expansion (upsampling linear)
U_new = FPI_expand(U, d_inter_new=4)
# u'_1 = u_1/2,  u'_2 = u_2/2,  u'_3 = u_3/2,  u'_4 = u_1/2  # 复制

# 结果: 大模型 = 小模型在相同输入下有相同输出
# 问题: 对称权重 → 梯度永远相同 → 有效参数减半
```

术语一般如何实现？如何使用？
- 适用于 Transformer 架构的宽度扩展：Embedding layers、QKV projections、MLP 等
- 对 MHA (Multi-Head Attention) 将每个 attention head 视为一个"神经元"
- 不能扩展深度——bert2BERT 使用 StackBERT 的 stacking 方法扩展层数
- LN (Layer Normalization) 在非整数倍扩展时输出不完全相同，但论文指出这对最终 loss 影响不大
- 在 AquilaMoE 中 FPI 作为对比 baseline：FPI-Stacking validation loss 4.30 vs FPI-Interpolation 3.31

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---
