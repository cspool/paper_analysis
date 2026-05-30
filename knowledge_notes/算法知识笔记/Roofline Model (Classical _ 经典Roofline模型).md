## Roofline Model (Classical / 经典Roofline模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Roofline Model 是 Williams, Waterman, Patterson (2009) 提出的可视化性能分析模型，用于评估给定应用在特定硬件上的性能上限和优化方向。模型核心关联两个参数：Operational Intensity I（FLOPs/Byte，即计算量/内存访问量，作为 X 轴）和 Achievable Performance P（FLOPs/sec，作为 Y 轴）。模型画出两条"屋顶"：(1) Memory Roof——斜线 P ≤ B_peak × I，由峰值内存带宽 B_peak 决定，表示数据供给速率对性能的上限；(2) Compute Roof——水平线 P ≤ P_peak，由处理器峰值算力决定，表示计算能力对性能的上限。两条屋顶的交点称为 Ridge Point，对应的 operational intensity 为 critical intensity Ī = P_peak / B_peak。应用若 I ≥ Ī → compute-bound（黄色区域）；若 I < Ī → memory-bound（蓝色区域）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 LLM 推理场景中，不同算子的 operational intensity 差异巨大：(1) Attention softmax：I ≈ 1-2 FLOPs/Byte（对每个 KV pair 仅做乘加和 exp），memory-bound——受限于 HBM BW；(2) Linear projection / FFN GEMM：I ≈ 100-500 FLOPs/Byte（大矩阵乘法，数据复用率高），compute-bound——受限于 GPU FLOPS。以 A100 为例，B_peak ≈ 2TB/s, P_peak ≈ 312 TFLOPS (FP16)，Ī = 312T/2T = 156 FLOPs/Byte。Attention 的 I 远低于 156 → memory-bound；FFN GEMM 的 I 可在 100-500 → 可能 compute-bound。

Roofline 分析流程：
```
1. Profile or calculate: Ops = total FLOPs of kernel, Bytes = total DRAM bytes accessed
2. I = Ops / Bytes
3. If I >= P_peak / B_peak: compute-bound → optimize via better algorithm, mixed precision
   Else: memory-bound → optimize via kernel fusion, data reuse, quantization
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 经典 Roofline 分析工具：Intel Advisor、NVIDIA Nsight Compute（自动计算 kernel 的 I 并绘制 roofline chart）、Empirical Roofline Toolkit (ERT)。
- MoE-Lightning 将 Roofline Model 扩展为 Hierarchical Roofline Model (HRM)，引入多层内存层次（CPU DRAM、GPU HBM、PCIe）和多处理器 compute roof，用于指导 CPU-GPU 混合推理的 operator placement 和 resource allocation。
- 局限：经典 Roofline 仅适合同构单处理器场景——对于 CPU-GPU 混合系统，需要 HRM 等扩展来建模跨层数据传输。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

**MoESD Roofline 应用**：MoESD 将 Roofline Model 应用于 SD speedup 性能建模（Algorithm 1）。核心设计 G(t; λRP, s) 函数——将 Ridge Point 的过渡区域建模为指数增长段后接线性段，λ<1 修正实际内存带宽利用率。MoE 专家部分的 modeling 引入 N(t)（激活专家数）和 Texp(t;ρ)（每专家平均 token 数）两个因子：N(t) 控制参数加载时间（memory access volume），Texp(t;ρ) 替代原始 t 作为 G() 的输入（因为每个 expert 仅处理分配到的 tokens 子集）。该设计解释了为何稀疏度 ρ 越小→Texp 越小→系统更 memory-bound→SD 加速窗口更宽。

**Amortization Roofline 变体**：MoE-SpeQ 将 Roofline 思想扩展到 speculative offloading 场景。X 轴改为 Amortization Intensity I_amort(k) = E[Accepted Tokens] / E[Synchronous I/O Bytes]（有用工作每字节同步 I/O），Y 轴为 Effective Throughput Θ(k)。两 Roof：Compute Roof（I/O 完美隐藏时上限）和 I/O Roof（斜率=B_PCIe）。在线 argmax_k Θ(k) 确定最优 draft length，受离线 SLO 约束 k_SLO。与 HRM 区别：HRM 指导 operator placement；Amortization Roofline 指导 speculation degree。
