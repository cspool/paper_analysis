## Block Quantization（块量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Quantization（块量化）是针对FP8等低精度格式的一种量化策略：将tensor划分为多个block（如$B_r \times d$或$B_c \times d$大小的子矩阵），每个block独立计算并保存一个scaling factor（通常为block内元素绝对值的最大值），量化时block内所有元素除以该block的scaling factor后映射到FP8表示范围，反量化时乘以对应的scaling factor恢复。与per-tensor quantization（整个tensor共享一个scalar scale）相比，block quantization提供更细粒度的动态范围适配——每个block独立伸缩，大幅减小outlier elements对量化精度的破坏。FlashAttention-3中，Q、K、V分别在进入attention kernel前进行block quantization，scaling factor可以fuse到前序操作（如rotary embedding，本身是memory-bound，无额外开销）。由于FlashAttention的tiled算法自然按block操作，block-wise scaling可以在$S_{ij}=Q_i K_j^T$计算时以零成本整合：$\tilde{S}_{ij} = \text{scale}_Q(i) \cdot \text{scale}_K(j) \cdot (Q_i K_j^T)$，scale因子仅需逐block相乘一次。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FP8 FlashAttention-3前向的block quantization pipeline：
```
输入: Q, K, V ∈ R^{N×d}, block sizes B_r, B_c
1. 将Q按行划分为T_r个blocks Q_i ∈ R^{B_r×d}
   将K按行划分为T_c个blocks K_j ∈ R^{B_c×d}
   将V按行划分为T_c个blocks V_j ∈ R^{B_c×d}
2. 对每个Q_i计算scale: scale_Q[i] = max(|Q_i|) / max_FP8
   对每个K_j计算scale: scale_K[j] = max(|K_j|) / max_FP8
   对每个V_j计算scale: scale_V[j] = max(|V_j|) / max_FP8
3. 量化（可fuse到rotary embedding）:
   Q_i_FP8 = quantize_FP8(Q_i / scale_Q[i])
   K_j_FP8 = quantize_FP8(K_j / scale_K[j])
   V_j_FP8 = quantize_FP8(V_j / scale_V[j])
4. Tiled attention主循环:
   for j in 0..T_c-1:
       S_ij = FP8_GEMM(Q_i_FP8, K_j_FP8^T)       // FP8 tensor core
       S_ij *= scale_Q[i] * scale_K[j]             // rescale before softmax
       P_ij = softmax(S_ij)
       O_i = FP8_GEMM(P_ij, V_j_FP8) * scale_V[j] // FP8 tensor core
```
关键：scale因子仅在softmax前和PV累加后各乘一次，不引入每元素scale开销——因为scale对于整个block是常数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block quantization在FlashAttention-3中通过CUTLASS的FP8 WGMMA primitives实现。量化本身在kernel外部或fuse到rotary embedding kernel中完成（rotary embedding是memory-bound，fuse不增加延迟）。Kernel内部通过WGMMA的FP8模式直接使用量化后的Q_i_FP8, K_j_FP8, V_j_FP8。Scale因子存储为per-block FP32标量数组，在kernel内通过寄存器传递。Block quantization同样适用于KV cache quantization（如KIVi, KVQuant）和weight quantization（如LLM.int8()），是一种通用的细粒度量化策略。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
