## FP4 Microscaling Quantization (NVFP4/MXFP4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FP4 Microscaling Quantization 是一种基于 GPU Tensor Core 原生支持的 4-bit 浮点量化方案。每个元素使用 FP4 格式（1-bit sign + 2-bit exponent + 1-bit mantissa, E2M1），仅可表示 16 个值（含符号），但通过将数据分组为固定大小的 block，每组共享一个 FP8 格式的 scale factor，实现"微缩放"（microscaling）——scale factor 在计算时动态恢复值域。NVFP4 是 NVIDIA Blackwell 架构的原生实现，量化 block 大小为 1×16，scale factor 为 E4M3 FP8 格式。MXFP4 是 OCP 开放标准的对应格式，block 大小为 1×32，scale factor 为 E8M0 格式。两者均通过 Blackwell 的 FP4MMA 指令直接执行硬件加速的矩阵乘法，无需软件反量化。SageAttention3 对比后选择 NVFP4，因为其在 attention 量化中精度显著优于 MXFP4（更小的 block 粒度 + E4M3 scale 提供更多有效量化级别）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FP4 microscaling attention 的 pipeline（SageAttention3 Algorithm 1 简化）：
```
输入: Q, K, V ∈ FP16, shape N×d
分块: Q → {Q_i} (B_q×d), K → {K_j} (B_kv×d), V → {V_j} (B_kv×d)

// Smoothing Q (SageAttention2 technique)
q̄_i = mean(Q_i)
s_Q, Q̂_i = φ(Q_i - q̄_i)  // φ: NVFP4 microscaling quant, 1×16 block

for j in range(T_n):
    s_K, K̂_j = φ(K_j^T)    // NVFP4 quant, K needs transpose
    s_V, V̂_j = φ(V_j)      // NVFP4 quant
    
    // QK^T in FP4
    S_ij = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T)
    
    // Online softmax
    m_ij = max(m_{i,j-1}, rowmax(S_ij))
    P̃_ij = exp(S_ij - m_ij)
    l_ij = e^{m_{i,j-1}-m_ij} * l_{i,j-1} + rowsum(P̃_ij)
    
    // Two-level quantization for P
    s_P1 = rowmax(P̃_ij) / (448×6)
    P̃_ij = P̃_ij / s_P1
    s_P2, P̂_ij = φ(P̃_ij)
    
    // PV in FP4
    O_ij = diag(e^{m_{i,j-1}-m_ij}) * O_{i,j-1} 
           + FP4MM(P̂_ij, s_P2, V̂_j, s_V) × s_P1

O_i = diag(l_{i,T_n})^{-1} * O_{i,T_n}
```
量化函数 φ(X)：将 X ∈ R^{N×d} 分为 1×16 块 X_ij，s_ij = max(|X_ij|)/6，X̂_ij = ⌈X_ij/s_ij⌋（FP4 rounding），scale s_ij ∈ E4M3 FP8。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：使用 NVIDIA CUTLASS 3.x + CUDA，调用 Blackwell FP4MMA PTX 指令（mmasm4或等价指令）。数据布局要求：(1) FP4 数据 packed 为 INT32（每 8 个 FP4 元素占 1 个 INT32）；(2) scale factor 按 FP8 (E4M3) 格式排列，每个 1×16 块对应 1 个 scale；(3) accumulator 为 FP32 布局，但与 operand A 寄存器布局不匹配时需 permutation 优化（SageAttention3 通过重排 accumulator 布局 + fuse K 列重排到量化 kernel 解决）。使用场景：所有需要 attention 计算的 Transformer 模型推理，Blackwell 架构 GPU（RTX5090/B200/B300），plug-and-play 替换现有 attention 实现。开源参考：https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
