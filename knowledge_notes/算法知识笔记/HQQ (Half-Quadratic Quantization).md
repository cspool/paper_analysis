## HQQ (Half-Quadratic Quantization)

术语是什么？
HQQ (Half-Quadratic Quantization, Badri & Shaji 2024) 是一种无需校准数据的 LLM 权重量化方法，通过半二次优化（half-quadratic optimization）直接在权重空间上交替优化求解量化参数（scale s、zero-point z）和量化权重 W_q。与传统 PTQ（需校准数据）不同，HQQ 完全不需要任何校准数据或前向推理。支持 1-8 bit 量化。在 MC-MoE 中，HQQ 被用于两方面：(1) 混合精度权重存储（compact bit-packed storage），将 PMQ 量化后的 1/2/3/4-bit 权重紧凑位压缩保存；(2) 提供 CUDA kernel 执行反量化（dequantization）+ 矩阵乘法，对 1-bit 权重有专门位运算加速。

从算法pipeline角度拆解术语：
```
// 推理路径
W_packed = load_compact_bits(memory, bit_width)  // 位压缩加载
W_dequant = W_packed * scale + zero_point         // 反量化到 FP16
Y = X @ W_dequant                                 // GEMM

// MC-MoE 1-bit 特化优化（位变换）:
// 存储: B̃ = (sign(W)+1)/2 ∈ {0,1}  // ±1 → 0/1
// 推理: s·xB = s(Σ_{B̃_{ij}=1} x_j - Σ_{B̃_{ij}=0} x_j)
// MACs: m (仅 scaling) vs FP16 dm 次乘法
```

术语一般如何实现？如何使用？
- 开源：https://github.com/mobiusml/hqq
- 优势：无需校准数据，推理 CUDA kernel 现成，位压缩存储紧凑
- MC-MoE 中的角色：作为推理部署工具，GPTQ 负责量化、HQQ 负责存储/反量化/内存管理
- 局限：HQQ 自身量化精度通常不如 GPTQ/Omniquant（无校准数据优化），但在极低位宽下差异缩小

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
