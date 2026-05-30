## Wanda (Pruning by Weights and Activations)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（Pruning by Weights and Activations）是 Sun et al. (ICLR 2024) 提出的一种简单且高效的 LLM 后训练剪枝方法。其核心思想是：对线性层权重 W ∈ R^{d_row×d_col}，用一小批校准数据前向传播得到输入激活 X ∈ R^{b×d_col}，定义每个权值的剪枝重要性度量为 S_ij = |W_ij| * ||X_j||_2（即权重绝对值乘以对应输入维度的 L2 范数），然后在每个输出神经元内比较，保留重要性最高的 (1-p%) 个权值，其余置零。Wanda 不需要权重更新（no weight update），不需要计算 Hessian 逆矩阵（不像 SparseGPT），计算复杂度为 O(d_hidden²)，仅需一次前向传播。其度量近似来源于 OBS 框架的简化：丢弃 Hessian 非对角项，用激活协方差对角近似替代完整 Hessian 逆。

从算法pipeline角度拆解术语：
```
# Wanda 逐层剪枝
# 输入: 预训练 LLM, 校准数据 (128 seqs), 目标稀疏度 p%
for layer in model.layers:
    X = forward_until(layer, X_calib)       # 收集该层输入激活
    for W_name in [W_gate, W_up, W_down]:
        W = layer.W_name                     # W ∈ R^{d_row × d_col}
        col_norm = ||X||_2 along dim=0       # [d_col] 每维 L2 范数
        S = |W| * col_norm.unsqueeze(0)      # [d_row × d_col] 重要性
        for row in range(d_row):
            thresh = top_k(S[row,:], k=d_col*(1-p%))
            mask[row,:] = (S[row,:] >= thresh)
        W = W * mask                         # 不重要权值置零
    X = forward_layer(layer, X)              # 传递到下一层
```
Wanda vs SparseGPT：(1) 无需 O(d_hidden³) Hessian 逆计算；(2) 不做权重更新补偿重构误差；(3) 小校准集下更鲁棒。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch forward hook 捕获激活 → 列 L2 norm → sort → mask。官方代码：https://github.com/locuslab/wanda。支持 LLaMA、OPT 等 50% unstructured 和 2:4 semi-structured。
- 局限：对 MoE 缺乏 router 感知——所有 expert 用相同度量。MoE-Pruner 增加 router 权重项（S = |W_ij| * ||X_j * Gate_j||）改进此局限。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
