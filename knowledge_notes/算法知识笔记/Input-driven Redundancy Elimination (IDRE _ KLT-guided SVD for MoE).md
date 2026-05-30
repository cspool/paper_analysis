## Input-driven Redundancy Elimination (IDRE / KLT-guided SVD for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IDRE（Input-driven Redundancy Elimination）是 KBVQ-MoE 框架的前处理模块，通过 KLT 引导的 SVD 消除 MoE expert 间的冗余表示。核心思想：MoE 中不同 expert 对相同输入常产生高度相似输出（Fig. 2a），VQ 重复量化相似表示浪费 codebook 容量。IDRE 分三步：(1) 对输入激活 X 做 KLT 分解构建输入相干基 `U_X = U_KLT Λ_KLT^{1/2}`，将输入按能量方向正交化；(2) 将所有 expert 权重投影到输入相干空间形成统一表示 W̄；(3) 对 W̄ 做截断 SVD 提取 top-k 主导共享分量（保留 FP16），剩余作为 expert-specific 残差交 VQ 量化。KLT 的关键作用：使 SVD 的 Gram 矩阵 `S = W̄^T W̄` 的频谱同时反映输入能量（通过 Λ_X）和跨 expert 权重能量（通过 Σ_i W^(i)T W^(i)），确保提取的共享方向在"输入高能量"和"跨 expert 高使用率"两个维度上同时主导。理论保证（Appendix A.2）：在输入相干空间中，输出 MSE 等价于权重误差的 Frobenius 范数 `L = Σ_i ||(W̃^(i) - W^(i)) U_X||_F^2`，因此所有最小化输出失真的提取/量化操作应在此空间执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# IDRE 核心计算 (Algorithm 1 pre-process 部分)
输入: expert 权重 {W^(1),...,W^(n)} ∈ R^{oc×ic}, 校准激活 X ∈ R^{B×ic}
输出: U_share, {W_share^(i)}, {W_quant^(i)}

# Step 1: KLT 分解输入激活
C_X = (X^T X) / (B-1)                           # ic×ic 输入协方差
U_KLT, Λ_KLT = eig(C_X)                         # C_X = U_KLT Λ_KLT U_KLT^T
U_X = U_KLT @ sqrt(Λ_KLT)                       # ic×ic 输入相干基

# Step 2: 投影 + 堆叠所有 expert
W̄ = zeros((n * oc, ic))
for i in 1..n:
    W̃^(i) = W^(i) @ U_X                         # oc×ic 投影到相干空间
    W̄[i*oc:(i+1)*oc, :] = W̃^(i)                 # 纵向堆叠

# Step 3: 截断 SVD 提取共享子空间
U, Σ, V = SVD(W̄)                                # W̄ 的 SVD
k = ic // 128                                   # 经验截断秩
U_k = U[:, :k]                                  # ic×k 共享左奇异向量
V_k_full = V[:, :k] @ diag(Σ[:k])              # (n·oc)×k 加权右奇异向量

# Step 4: 分解共享/残差
U_share = inv(U_X) @ U_k                        # ic×k 原始空间共享映射
for i in 1..n:
    V_k_i = V_k_full[i*oc:(i+1)*oc, :]          # oc×k expert i 的右奇异分量
    W_share^(i) = (U_share @ V_k_i^T)^T          # oc×ic 共享分量 (保留 FP16)
    W_quant^(i) = W^(i) - W_share^(i)            # oc×ic expert-specific 残差
```

关键性质：(1) 共享子空间通过 Ky Fan 定理最优：`max_{U^T U=I} ||W̄ U||_F^2`，保留了跨 expert 的最大能量方向；(2) 截断误差由尾部奇异值精确控制：`Σ_{j>k} σ_j^2`；(3) 冗余消除率 `ρ_k = Σ_{j=1}^k σ_j^2 / Σ_{j=1}^{ic} σ_j^2`，k=ic/128 时典型 ρ_k ≈ 0.6-0.8（功率律衰减 `σ_j^2 ∝ j^{-α}, α>1`）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IDRE 应用于 MoE 层所有 expert（shared + routing）的 MLP 权重。仅离线执行一次（开销约等于 1 次前向传播）。消融验证：KLT+SVD 优于纯 SVD（Table 3: Qwen1.5-MoE-A2.7B 2-bit W2 PPL 从 14.03 降至 11.87）；k=ic/128 为经验最优平衡点（Table 4: k=1/128 时 PPL 11.87, k=1/32 时 PPL 11.01 但 bit-width 从 2.08 升至 2.20）。IDRE 可解耦为独立 pre-processing 模块，与不同 VQ 方法组合（GPTVQ, VPTQ, PCDVQ 等）。

涉及论文标题：
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---
