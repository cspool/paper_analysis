## Asymmetric GEMM (on GPU Tensor Cores)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric GEMM（非对称精度GEMM）指权重和激活使用不同精度的矩阵乘法，特指W4A8 GEMM（4-bit weight × 8-bit activation）。与Symmetric GEMM（W8A8, FP8）中操作数精度相同可直接输入Tensor Cores不同，Asymmetric GEMM需在main-loop中通过CUDA Cores将低精度操作数dequantize到高精度后才能执行MMA。这引入额外pipeline stage——dequantization——使用与MMA不同的硬件单元（CUDA Cores vs Tensor Cores），成为性能瓶颈或优化机会。

从kernel调度角度拆解术语：
```
Symmetric GEMM (W8A8):
  Main-loop: Load W8→SMEM → ldmatrix→RF → WGMMA(W8, A8)
  全在Tensor Core数据路径

Asymmetric GEMM (W4A8):
  Main-loop: Load W4→SMEM → LDS→RF → Dequant(W4→W8, CUDA Cores) → WGMMA(W8, A8)
  引入CUDA Core dequantization stage
  挑战: CUDA Cores 60 TFLOPS << Tensor Cores 990 TFLOPS INT8 (H100)

性能模型 (LiquidGEMM Eq.6):
  T = ⌈M/Mt⌉ · max(T_LD, α·N·K/Φ_CUDA + min(Mt,M)·2·N·K/Φ_TC)
  T_DQ = α · N · K / Φ_CUDA  ← dequant bottleneck
  消除bottleneck需: α ≤ 5.07 (memory-bound overlap) 或 α ≤ 5.05 (compute-bound)
```

术语一般如何实现？如何使用？
QServe：QoQ dequant (α≈10+) + 串行pipeline → bottleneck严重。LiquidGEMM：LiquidQuant (α≈0.875) + ImFP pipeline → dequant被有效隐藏，在所有batch size超越W8A8。计算重构Y=(WX^T)^T更好利用WGMMA m=64维度。W4A8的实用化取决于dequantization能否被pipeline有效隐藏——这是Asymmetric GEMM的核心系统挑战。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---
