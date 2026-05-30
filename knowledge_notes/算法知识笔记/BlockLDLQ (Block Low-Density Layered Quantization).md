## BlockLDLQ (Block Low-Density Layered Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BlockLDLQ 是 QuIP# 提出的一种向量量化兼容的自适应舍入算法，是对 QuIP (Chee et al., 2023) 标量 LDLQ 的块级推广。传统 LDLQ 基于 Hessian H 的 LDL 分解（H = L^T D L），设置反馈矩阵 U = L^T - I，逐列迭代舍入 Ŵ_k = Q(W_k + (W_{:k-1} - Ŵ_{:k-1})a_k)，其中 Q 为标量量化器。BlockLDLQ 的改进：(1) 将列级舍入推广到列块级——以 g 列（QuIP# 中 g=8，匹配 E8P 维度）为一块；(2) 基于 g-block LDL 分解 H = L^T D L，其中 L 为单位块下三角矩阵（n/g 个对角块均为 I_{g×g}），D 为块对角矩阵；(3) 反馈矩阵 U = L^T - I；(4) 量化步骤：Ŵ_k = Q(W_k + (W_{:k-1} - Ŵ_{:k-1})A_k)，其中 A_k ∈ R^{n×g} 为 U 的第 k 个 g 列块，Q 为向量量化器（E8P）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BlockLDLQ 的完整计算流程（以 Llama 2 线性层 W ∈ R^{4096×4096}, g=8 为例）：
```
# 输入: W_hat ∈ R^{m×n} (RHT 变换后), H_hat ∈ R^{n×n} (RHT 变换后)
# 参数: g = 8 (block size), codebook C ∈ R^{256×8} (E8P)

# Step 1: g-block LDL 分解
# H_hat = L^T @ D @ L
# L: unit block lower triangular (512×512 blocks of 8×8 each)
# D: block diagonal (512 blocks of 8×8 each)
# 实现: 从 Cholesky 分解 H_hat = G^T G 导出 L 和 D

# Step 2: 设置反馈矩阵
# U = L^T - I  (unit block upper triangular)

# Step 3: 逐块自适应舍入 (n/g = 512 blocks)
for k from 1 to 512:
    # 当前块的列范围: [(k-1)*8 : k*8]
    A_k = U[:, (k-1)*8 : k*8]     # 反馈向量, size: n × 8
    
    # 之前已量化块的误差
    past_error = W_hat[:, 0:(k-1)*8] - W_quantized[:, 0:(k-1)*8]
    
    # 带反馈的预舍入值
    W_pre = W_hat[:, (k-1)*8 : k*8] + past_error @ A_k
    
    # 对 8 列中的每行独立做 E8P VQ
    for row in 0..m-1:
        W_quantized[row, (k-1)*8 : k*8] = e8p_quantize(W_pre[row, :])
```

误差理论界（Theorem 4.1）：若 H 是 μ-incoherent 且 Q 满足 E[(Q(x)-x)(Q(x)-x)^T] ≤ σ²I，则 E[tr((Ŵ-W)H(Ŵ-W)^T)] ≤ gmμ²σ²/n · tr(H^{1/2})²。与独立块舍入相比，BlockLDLQ 的界从 tr(H) 改善到 tr(H^{1/2})²/n（继承 QuIP 标量 LDLQ 的相同改进因子）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BlockLDLQ 的实现关键：(1) g-block LDL 分解通过对 H 的 Cholesky 分解 H = G^T G 做块划分导出——对角块 D_k = G_{kk}^T G_{kk}，下三角块 L_{jk} = G_{jj}^{-T} G_{jk}^T；(2) 在 QuIP# 中 g=8，与 E8P 的 8 维向量量化完全匹配；(3) 对 m≫n 的大矩阵（LLM 线性层），内循环近似 O(m·2^{kd}·g)（每行搜索 2^{16} 条目），E8P 的 256× 压缩将搜索空间从 2^{16} 降到实际的 256 次查表；(4) 开源：https://github.com/Cornell-RelaxML/quip-sharp，BlockLDLQ 在 Python 层面实现，E8P 量化器在 C++/CUDA 层面实现。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---
