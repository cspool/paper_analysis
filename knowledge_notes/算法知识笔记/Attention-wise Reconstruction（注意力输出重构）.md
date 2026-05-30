## Attention-wise Reconstruction（注意力输出重构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention-wise Reconstruction 是 aespa 论文提出的量化重构目标范式：以 attention 模块的输出 `SA(Q, K, V) = softmax(QK^T/√d)V` 为重构目标，而非以单个 linear 层的输出或整个 Transformer block 输出为目标。其定位介于 layer-wise reconstruction（仅最小化各层输出误差）和 block-wise reconstruction（最小化整个 block 输出误差）之间。核心动机：Q、K、V 投影之间存在强跨层依赖——Q 和 K 共同决定 attention map A，A 又与 V 相乘得到最终 attention 输出——而 layer-wise 方法假设层间独立，忽略了这种依赖。通过以 attention 输出为重构目标，可以在逐层量化的前提下引入跨投影依赖信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种重构目标的数学对比：
```
# Layer-wise Reconstruction (AdaRound, OPTQ)
# 目标: 最小化各层独立输出误差
for each linear layer W:
    minimize E[||Q(W)X - WX||^2]          # 不考虑其他层

# Block-wise Reconstruction (BRECQ)  
# 目标: 最小化整个 attention block 输出误差
minimize E[||SA(Q_hat, K_hat, V_hat) - SA(Q, K, V)||^2]
# 所有 Q/K/V 联合优化 → 每轮需完整 attention forward → O(B·L·d_h·max(d,L))

# Attention-wise Reconstruction (aespa)
# 目标: 最小化 attention 输出误差，但逐层单独量化
for W_V (fixed Q, K):
    minimize E[||A·V_hat - A·V||^2]       # ΔSA_V, Equation (9)
    = minimize E[||ΔW_V·X·A^T||^2]       # 展开后 Q/K 信息通过 A 耦合
for W_Q (fixed K, V):
    minimize E[||SA(Q_hat,K,V) - SA(Q,K,V)||^2]  # Equation (11)
    ≈ minimize E[||K·ΔW_Q·X||^2]                # Equation (15), 通过上界近似
for W_K (fixed Q, V):
    minimize E[||SA(Q,K_hat,V) - SA(Q,K,V)||^2]  # Equation (25)
    ≈ minimize E[||Q·ΔW_K·X||^2]                # Equation (16)
```
有效性验证（Table 5）：在 OPT-125M 上，同时使用 layer-wise 量化粒度 + attention-wise 重构目标，INT2 PPL=69.23，显著优于 layer-wise 粒度 + layer-wise 重构的 AdaRound（PPL=160.7），接近 block-wise 联合量化的 BRECQ（PPL=60.38）。INT3/4 下与 BRECQ 性能几乎持平。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 attention-wise reconstruction 需设计精炼量化损失函数。aespa 的核心实现技巧：(1) 对 W_V：利用 Q/K 固定时 A=softmax(QK^T/√d) 不变，将目标简化为 `E[||ΔW_V·X·A^T||^2]`；(2) 对 W_Q/W_K：通过 first-order Taylor 展开近似 softmax 变化，再构造上界避开 Jacobian 存储（L³ 元素）。然后通过预计算 `E[XA^TAX^T]`、`E[K^TK]`、`E[Q^TQ]` 将这些统计量"冻结"，后续迭代无需执行 attention forward。这一策略不仅降低了计算复杂度，还使每轮 loss 计算等价于在整个校准集上评估（batch size = 全部校准数据量），梯度估计更准确，2000 轮即可收敛。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
