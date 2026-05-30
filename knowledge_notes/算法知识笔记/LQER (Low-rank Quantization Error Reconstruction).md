## LQER (Low-rank Quantization Error Reconstruction)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LQER (Low-rank Quantization Error Reconstruction) 是 Zhang et al. (2024a, Imperial College London) 提出的 PTQ 方法，在权重量化后使用激活引导的启发式标度矩阵 S 对权重量化误差做截断 SVD 来求解低秩补偿项 A_k, B_k。LQER 的标度矩阵 S 通过对校准集输入 x 的每个维度累积平均绝对值来构建：s_i = E[|x_i|]，S = diag(s_1, ..., s_m)（归一化后）。关键步骤：对标度化后的权重量化误差 S(W − W̃) 做 SVD → 取前 k 个奇异值/向量 → A_k = S^{-1}U_{:,:k}, B_k = Σ_{:k,:k}V_{:k,:}^T。LQER 观察到标度后某些层的奇异值衰减更快（"更理想"的奇异值分布）。

从算法pipeline角度拆解术语：QERA-approx 从理论上揭示了 LQER 的本质——当 Assumption 1 (E[x_i x_j]=0 for i≠j) 成立时，最小化层输出误差的最优标度应为 S = diag(√E[x_i²])，而非 LQER 的 diag(E[|x_i|])。这解释了：(1) LQER 通常优于 ZeroQuant-V2（S=I），因为标度方向近似正确；(2) LQER 性能随校准样本数增加而随机波动（Figure 3 purple curve）——基于绝对值的启发式统计量不收敛到理论最优解。QERA-approx 使用正确的二阶矩标度后，性能随校准样本数单调提升至收敛（Figure 3 green curve）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LQER 的算法流程（Algorithm 2）：初始化 s ← 0 → 对校准集 X 中每个样本 x: s += abs(x) → S = (1/N) diag(s) → 量化 W_q = q(W) → 反量化 W̃ = dq(W_q) → SVD(S(W − W̃)) → A_k = S^{-1}U_{:,:k}, B_k = Σ_{:k,:k}V_{:k,:}^T。LQER 是纯 PTQ 方法，不涉及训练，低秩项离线预计算并在推理时合并入 W̃。ZeroQuant-V2 可视为 LQER 在 S=I 时的特例。LQER 论文的局限：启发式标度不保证最优性，校准样本数与性能关系不稳定。QERA（同研究组后续工作）将 LQER 理论化并给出了正确的解析解。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction

---
