## Hadamard Transformation for LLM Quantization（LLM 量化中的 Hadamard 变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hadamard Transformation for LLM Quantization 是 QuaRot (Ashkboos et al. 2024) 引入的预量化变换技术。Hadamard 矩阵 H ∈ {+1,−1}^{n×n} 是一个正交矩阵（H^T H = I），其元素仅含 +1 和 −1，无需浮点乘法即可实现（仅需加减）。在 LLM 量化中，对激活和权重分别应用 H 变换：Y = XW^T = (XH)(H^T W^T)，利用矩阵乘法的正交等价性。Hadamard 变换将单个通道的离群值通过旋转分散到所有通道，消除极端离群值峰值。优点：(1) 快速——H 的元素为 ±1，可用快速 Walsh-Hadamard Transform (FWHT) 在 O(n log n) 时间内计算；(2) 全局通用——所有层共享同一 Hadamard 矩阵；(3) 无需学习。局限：(1) 不考虑逐层特性差异，某些层的分布仍呈现 steep envelopes；(2) 对 pivot tokens 的大量离群值效果有限；(3) 修改 LayerNorm 为 RMSNorm 导致全局变换共享受限。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuaRot 中 Hadamard 变换的量化流程（LLaMA-2-7B, W4A4）：

```
// 离线阶段：权重变换
H_n = hadamard_matrix(n)                   // n=4096, H∈{+1,-1}^{4096×4096}
W'_qkv = H_n @ W_qkv @ H_n^T               // QKV 投影权重融合
W'_o = H_n @ W_o @ H_n^T                   // 输出投影
// ... 所有线性层同理
W_q = RTN_quantize(W')                      // 量化

// 在线推理阶段：激活变换
X' = fast_walsh_hadamard_transform(X)       // O(n log n), 仅加减运算
X_q = per_token_quantize(X')               // INT4
Y_q = INT4_matmul(X_q, W_q)                // CUTLASS kernel
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuaRot 中的 Hadamard 变换使用在线 FWHT 实现，带来约 0.26× 端到端减速（3 次在线变换）。SpinQuant 用学习到的正交旋转矩阵替代固定 Hadamard 矩阵以提升表达力。FlatQuant 则完全放弃 Hadamard 变换，转而使用可学习的 Kronecker 仿射变换，在逐层定制化和推理开销间取得更好平衡。QTIP 使用 Random Hadamard Transform (RHT) 作为 incoherence processing 的核心：W̃ ← V_m S_m W S_n V_n^T，其中 V_k 为 Hadamard 矩阵、S_k 为随机符号向量。RHT 以概率 ≥1-δ 使 Ŵ 的 incoherence μ_Ŵ = 2log(4mn/δ)，意味权重近似 i.i.d. 高斯分布——恰好是 TCQ 对 i.i.d. 高斯源高效量化的前提。QTIP 中的 RHT 是离线预处理（无需在线推理开销），Hadamard 矩阵来自 Neil Sloane 网站 (http://neilsloane.com/hadamard/)。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- QTIP: Quantization with Trellises and Incoherence Processing
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

Quamba2 中 Hadamard 变换的使用：(1) offline fusion——将 H 矩阵 offline 融合到 input/output projection 权重（$W_{out}^H = H W_{out} H^T$, $W_{in}^H = W_{in} H^T$），配合 online FWHT 实现 compute-invariance，避免半精度激活中的 outlier 放大 4-bit 权重的量化误差；(2) 在 output proj input 上应用（与 Quamba/MambaQuant 一致），消除 outlier 以提升 4-bit weight 的量化效果——W4A16 ablation 中 Hadamard + PerG 从 64.7% 提升到 69.6% LAMBADA accuracy。
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

SDP4Bit 中 Hadamard 变换的新用途——梯度平滑通信压缩：(1) 在线 32×32 Walsh-Hadamard 变换应用于梯度张量（而非权重/激活），在 INT4 量化前平滑梯度 outlier，使量化误差大幅减少；(2) 两步 Hadamard：intra-node all-to-all 量化前一次 + inter-node all-to-all 量化前一次；(3) 利用 H·H=I 和 ΣHg=HΣg 的数学性质裁剪冗余 transform（6次→2次）；(4) Hadamard 与 (de)quantization 融合为单个 CUDA kernel，要求 group_size 能被 H size 整除（32），确保 fused kernel 内存局部性。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- QTIP: Quantization with Trellises and Incoherence Processing
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals


ResQ 中 Hadamard 的独特用法——U_D FFN 内投影：ResQ 在 FFN block 内部使用 U_D 投影 down_proj 的激活。因 SiLU/GELU 激活函数隔断导致 U_D 无法融入前一层权重，而 d_FFN（通常为 d_hidden 的 3-4 倍）上的直接矩阵乘法开销很大，ResQ 将 U_D 选择为 Hadamard 矩阵，利用快速 Hadamard 变换实现 O(d log d) 的运行时计算。Hadamard 矩阵不存在于特定维度时回退为随机正交旋转。
