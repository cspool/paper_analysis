## Quantization Error Reconstruction (QER)（量化误差重建）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Error Reconstruction (QER) 是指在模型量化后，用额外的低秩、高精度矩阵 C_k = A_k B_k（rank k ≪ min(m, n)）来补偿/重建量化误差的技术。给定线性层 y = xW 和量化后的近似 W̃ = dq(q(W))，QER 将输出重写为 ỹ = x(W̃ + C_k)。QER 问题的核心挑战在于：给定 C_k 的秩约束，应以什么优化目标求解 A_k 和 B_k。传统方法（ZeroQuant-V2, LoftQ）通过截断 SVD 最小化权重逼近误差 ||W − W̃ − C_k||_F（Problem 1，Eckart-Young-Mirsky 最优解）。QERA 论文证明该目标不能保证降低模型输出误差，转而最小化层输出误差 E[||x(W̃ + C_k) − xW||²]（Problem 2），并给出闭式解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QER 在模型推理pipeline中的位置：模型权重加载 → 逐层应用 q(·) 量化 → dq(·) 反量化 → 计算权重量化误差 E = W − W̃ → 求解 A_k, B_k（离线预处理）→ 推理时 y = xW̃ + x(A_k B_k)。两种求解范式的对比伪代码：

```
# === Problem 1: Minimize Weight Error (ZeroQuant-V2 / LoftQ) ===
W_tilde = dq(q(W))
E = W - W_tilde         # weight quantization error
U, Sigma, Vt = SVD(E)   # full SVD
C_k = U_{:,:k} @ diag(Sigma_{:k}) @ Vt_{:k,:}  # truncated SVD (Eckart-Young-Mirsky optimal)
# 前向: y = x @ W_tilde + x @ C_k
# 问题：最小化的是权重误差，不保证输出误差降低 (Section 4.2, Figure 1)

# === Problem 2: Minimize Layer Output Error (QERA) ===
W_tilde = dq(q(W))
# Calibration: 对校准集 X 中的输入向量 x 累积统计量
# QERA-exact:
R = 1/N * sum_i x_i^T x_i           # 自相关矩阵 [m, m], FP64累积
R_sqrt = matrix_sqrt(R)              # blocked Schur algorithm (CPU, FP64)
Q = R_sqrt @ (W - W_tilde)          # 标度化误差
U, Sigma, Vt = SVD(Q)
A_k = inverse(R_sqrt) @ U_{:,:k}    # 反标度化: [m, k]
B_k = diag(Sigma_{:k}) @ Vt_{:k,:}  # [k, n]

# QERA-approx (假设 E[x_i x_j]=0 for i≠j):
s = [sqrt(E[x_1^2]), ..., sqrt(E[x_m^2])]  # 对角激活统计
S = diag(s)
Q = S @ (W - W_tilde)
U, Sigma, Vt = SVD(Q)
A_k = inverse(S) @ U_{:,:k}
B_k = diag(Sigma_{:k}) @ Vt_{:k,:}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QER 实现基于 PyTorch + SciPy（矩阵平方根使用 blocked Schur algorithm, CPU 执行）。应用于两个场景：(1) QPEFT 中作为 LoRA 初始化——QERA-approx（21s-30min）替代 QLoRA 的随机初始化，使微调起点更接近全精度模型，2-bit RoBERTa @ GLUE 平均 Δacc = +6.05% vs LoftQ；(2) PTQ 中作为离线误差补偿——QERA-exact 或 QERA-approx 预计算低秩项并合并入 W̃，推理时零额外开销。QERA 对量化函数 q(·) 无约束，兼容任意量化方法（Uniform, NF4, MXINT 等）。QERA-approx 的正确标度 S = diag(√E[x_i²]) 从理论上解释了 LQER 启发式标度（使用 E[|x_i|]）在更多校准样本下性能不稳定的原因。开源：https://github.com/ChengZhang-98/QERA。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction

---
