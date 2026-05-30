## Randomized Hadamard Transform (RHT) for Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Randomized Hadamard Transform (RHT) 在 QuIP# 中特指用于量化非相干处理的随机化正交变换：x → V S x，其中 V ∈ R^{n×n} 为正交的 scaled Hadamard 矩阵（V^T V = I），S = diag(s), s ∼ U{±1}^n 为随机符号对角矩阵。RHT 最早由 Halko et al. (2011) 在随机数值线性代数中引入，用于近似矩阵分解。QuIP# 首次将 RHT 应用于 LLM 量化，替代 QuIP (Chee et al., 2023) 的 2-factor Kronecker 积正交矩阵构造。RHT 的核心性质：(1) 将任意向量与 Hadamard 矩阵相乘使输出向量在所有坐标上"扩散"——对标准基向量 e_i，输出 H e_i 的所有元素幅值为 1/√n（完全非相干）；(2) 随机符号翻转提供了随机化保证（Lemma 3.1），将矩阵变为 μ-incoherent；(3) 通过 Fast Walsh-Hadamard Transform (FWHT) 实现 O(n log n) 计算，仅需加减法无需浮点乘法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuIP# 中 RHT 在量化前/推理时的完整计算流程：
```
# === 量化前 (Algorithm 3: IP-RHT) ===
# W ∈ R^{m×n}, H ∈ R^{n×n}
# 采样随机符号: S_U ∼ {±1}^m, S_V ∼ {±1}^n

# RHT 应用于权重矩阵 W (双边):
# Ŵ = Had(diag(S_U) × Had(diag(S_V) × W^T)^T)
#    = H_m × S_U × W × S_V × H_n^T
# 其中 Had(·) 为 FWHT (O(n log n))

# RHT 应用于 Hessian H (双边):
# Ĥ = H_n × S_V × H × S_V × H_n^T
#    = Had(diag(S_V) × Had(diag(S_V) × H)^T)

# === 推理时 (Algorithm 2) ===
# y = Had(S_U ⊙ (quantized_Ŵ × Had(S_V ⊙ x)))
#   = H_m × S_U × (quantized_Ŵ) × S_V × H_n^T × x

# 缩放因子: ρ|W|_F 乘以权重矩阵使其幅值匹配 E8P 码书覆盖范围
```
QuIP# 的 key insight：RHT 将权重分布转换为近似球状亚高斯分布（中心极限定理效应），这为后续 E8P 球状格基码书的向量量化创造了理想条件——"先让数据变高斯，再用球状码书去量化高斯数据"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RHT 的实现关键点：(1) FWHT kernel 使用 Tri Dao 的 CUDA 实现（https://github.com/Dao-AILab/fast-hadamard-transform），通过 in-place butterfly 运算实现，log₂n 层，每层 n/2 对 (a+b, a-b) 操作；(2) 符号向量 S_U, S_V 在无微调时以 1-bit/元素存储（±1），微调时以 FP16 存储以允许梯度优化；(3) 存储开销极低：对 m×n 矩阵，16(n+m)/(nm) bits/weight（微调时）或 (n+m)/(nm)（无微调），对于 4096×4096 的最小矩阵仍 <0.01 bits/weight；(4) 维度非 2 的幂时使用 Kronecker 分解 H = H_p ⊗ H_q，复杂度 O(q²p log p)；(5) 备选方案 RFFT：使用 FFT + 随机复数相位，仅需 n 为偶数，适合无 Hadamard 矩阵的高速硬件（如 DSP/FPGA）。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---
