## Systolic-array Accelerator (for VLM Inference)

术语解释
Systolic-array Accelerator是Focus论文的baseline计算架构，也是Focus Unit所嵌入的宿主加速器。它采用weight stationary dataflow的32×32 PE array，每个PE执行FP16 multiply和FP32 accumulate。

术语是什么？通过联网搜索让回答具体和精准。
Systolic Array是一种通过2D PE grid做rhythmic data movement完成矩阵乘法的硬件架构，广泛应用于Google TPU和各类AI加速器。在Focus中，systolic array配置为32×32 PE array、FP16 multiply/FP32 accumulate、weight stationary dataflow（weights预加载到PE中，inputs和outputs流经array）。GEMM以tile方式执行：输入tile m×K (m=1024)、weight tile K×n (n=32)、输出tile m×n。此架构是Focus的baseline computing fabric，Focus Unit作为modular add-on嵌在array的memory interface侧。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Focus baseline systolic array的GEMM tiling执行：
```
GEMM: Input (M×K) × Weight (K×N) → Output (M×N)
Tiling:
- Input tile: m×K (m=1024) streamed from input buffer (128KB)
- Weight tile: K×n (n=32) preloaded from weight buffer (78KB)
- Output tile: m×n accumulated in output buffer (512KB)
- K dimension: iterated in k=32 sub-tiles

Inner loop (weight stationary):
  PE[i][j] accumulates: sum_k input[i][k] * weight[k][j]

Outer loop (output stationary):
  Output tile stays on-chip while K iterations complete
```
Focus论文中的systolic array和所有baseline accelerator (vanilla SA, AdapTiV, CMC)的核心逻辑均用SystemVerilog实现，用相同技术节点(TSMC 28nm)、频率(1.32ns target)、PE数量(32×32)和DRAM bandwidth (DDR4 64GB/s)做公平评估。On-chip SRAM由Memory Compiler生成。Cycle-accurate性能使用基于SCALEsim-v2的simulation framework评估。开源实现：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

---

