## ASVD (Activation-aware Singular Value Decomposition, 激活感知奇异值分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

ASVD 是标准 SVD 低秩分解的激活感知扩展。标准 SVD 仅基于权重矩阵 $W$ 的奇异值做低秩近似，忽略激活值分布——某些输出通道激活值远大于其他通道，在 MSE 中贡献更大。ASVD 引入缩放矩阵 $S = \operatorname{diag}(|\bar{X}|^\alpha)$（$\bar{X}$ 为标定数据平均激活值，$\alpha=0.5$），对缩放后权重 $W_s = W S$ 做 SVD，使低秩分解重点保留高激活值通道的信息。CSKV 中使用 Absolute Mean Value 方法计算 S，从 256 个标定样本收集激活值。

从算法pipeline角度拆解术语。

```
// 1. 收集标定激活 (256 samples)
calib_X = collect_activations(model, calib_dataset)
// 2. 计算 S
S_diag = mean(|calib_X|, dim=0)^α  // α=0.5
S = diag(S_diag)
// 3. 对缩放后权重 SVD
U_s, Σ_s, V_s_T = SVD(W @ S)
// 4. 构造低秩分解
A_K = inv(S) @ U_s[:, :hcomp] @ sqrt(Σ_s[:hcomp, :hcomp])
B_K = sqrt(Σ_s[:hcomp, :hcomp]) @ V_s_T[:hcomp, :] @ inv(S)
```

术语一般如何实现？如何使用？

CSKV 消融实验证明 SVD-based 初始化的必要性：随机初始化训练完全无法收敛（Avg.Acc=0.00），ASVD 初始化后 Loss 从 ~5.5 迅速收敛到 ~4.0。ASVD 初始化在各压缩率下均优于标准 SVD（80% 压缩：ASVD 0.92 vs SVD 0.87 vs Random 0.00）。仅对 W_K, W_V 做 ASVD（非所有权重）。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
