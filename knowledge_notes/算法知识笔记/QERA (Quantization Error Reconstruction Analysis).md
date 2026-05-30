## QERA (Quantization Error Reconstruction Analysis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QERA (Quantization Error Reconstruction Analysis) 是 Imperial College London 提出的量化误差重建分析框架，给出 QER 问题（最小化层输出误差）的闭式解。核心贡献：(1) 理论证明：QER 问题中最小化权重逼近误差（Problem 1）与最小化模型输出误差（Problem 2）不等价——LoftQ 迭代增加权重误差单调降但模型输出误差可能增加（Figure 1）；(2) 两个闭式解：QERA-exact (Theorem 1) 使用输入自相关矩阵 R_XX = E[x^T x] 的矩阵平方根进行标度化，与 CALDERA Lemma 4.2 等价但证明路径不同；QERA-approx (Theorem 2) 在"不同嵌入维度不相关"假设（Assumption 1, E[x_i x_j]=0 for i≠j）下将 R_XX 简化为对角矩阵 S² = diag(E[x_i²])；(3) 实践验证：QERA-exact 在 LLM 上的 Assumption 1 测试（超过 60% 层满足、MLP 层尤佳）；QERA-approx 解释了 LQER 启发式标度的成功并解决其校准不稳定问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QERA 的完整数学推导链：

```
问题: min_{C_k} E[||x(W̃ + C_k) − xW||²]

定义 P = W̃ + C_k − W, p_i = P 的第 i 行
目标 = E[||Σ_i x_i p_i||²]
    = E[Σ_i Σ_j x_i x_j p_i p_j^T]
    = Tr(E[x^T x] · P P^T)                (Hadamard product 性质)
    = Tr(R_XX · P P^T)                     (R_XX = E[x^T x])
    = ||R_XX^{1/2} P||_F²                 (R_XX 正半定，有唯一对称平方根)

→ 等价于: min_{C_k} ||R_XX^{1/2}(W̃ + C_k − W)||_F²

令 Q = R_XX^{1/2}(W − W̃), Q_k = R_XX^{1/2}C_k
→ min_{Q_k} ||Q_k − Q||_F²
→ Q_k = SVD_k(Q)  (Eckart-Young-Mirsky 定理)
→ C_k = (R_XX^{1/2})^{-1} · SVD_k(R_XX^{1/2}(W − W̃))

在 QERA-approx 下: 假设 E[x_i x_j]=0 for i≠j
→ R_XX = diag(E[x_1²], ..., E[x_m²]) = S²
→ C_k = S^{-1} · SVD_k(S · (W − W̃))
其中 S = diag(√E[x_1²], ..., √E[x_m²])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PyTorch + Transformers + PEFT + SciPy（blocked Schur algorithm for matrix sqrt, CPU FP64）。关键技术细节：(1) R_XX 外积在 FP32 累积→FP64 精度矩阵平方根以确保数值稳定；(2) block size=32 MXINT 作为量化格式（也可用任意其他量化方法）；(3) 逐层独立计算可并行化。QPEFT 场景推荐 QERA-approx（初始化时间 21s-30min vs QERA-exact 的 1.6min-4.9h，微调可补偿近似误差）；PTQ 场景推荐 QERA-exact（离线预处理，精度最高，推理零额外开销）。QERA-exact 的瓶颈在矩阵平方根的 CPU 计算，GPU 加速矩阵平方根是关键优化方向。开源：https://github.com/ChengZhang-98/QERA。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction

---
