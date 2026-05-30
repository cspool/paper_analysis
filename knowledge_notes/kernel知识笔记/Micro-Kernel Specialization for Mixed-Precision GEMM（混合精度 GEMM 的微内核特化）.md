## Micro-Kernel Specialization for Mixed-Precision GEMM（混合精度 GEMM 的微内核特化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Micro-Kernel Specialization 是 MxMoE 提出的 CTA 级 CUDA kernel 设计策略：为每种量化精度（如 W2A16, W4A16, W4A4-g128, W8A8）实现专用的 micro-kernel，而非开发一个 universal kernel 处理所有精度。每个 micro-kernel 是 CTA index-independent 的 CUDA device function，资源通过 C++ template 参数指定，memory access 针对该精度的计算-访存模式手工调优。例如：W2A16 micro-kernel 集成 fused dequantization + bit manipulation 优化 int-to-float 转换；W4A4-g128 micro-kernel 使用 multistage software pipelining 严格遵循 128 量化 group 约束。

该策略的关键优势：对比 universal kernel（所有精度共享同一代码路径），specialized micro-kernel 消除了运行时条件检查（避免阻碍 MAC-loop 展开），允许针对精度特性选择最优 tile size。对比手工为每种精度组合写 kernel（|S|! 个），specialized micro-kernels 只需实现 |S| 个可配置 micro-kernel，由 kernel generator 自动组合。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Micro-kernel 对比示例（W4A4）：

```
Universal Kernel (单 kernel 处理所有精度):
  // 运行时条件判断
  if (group_size == 128):
      // 受限 tile k 选择 (per-channel kernel tile k=256 不可用)
      load_and_dequant_group128(...)
  else:
      load_and_dequant_per_channel(...)
  # 性能: W4A4 per-channel 929 TOPS, W4A4 group128 412 TOPS

Specialized Micro-Kernel (MxMoE):
  // W4A4 per-channel micro-kernel
  template<int TileM, int TileN, int TileK>
  __device__ void w4a4_per_channel_micro_kernel(...) {
      // 专为 per-channel 优化的 dequant + MMA pipeline
      // 无运行时分支, 全循环展开
  }
  # 性能: W4A4 per-channel 1070 TOPS, W4A4 group128 667 TOPS
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MxMoE 的 kernel generator 根据 ILP 分配的方案，从 micro-kernel 库中选择对应的 micro-kernels，通过 resource configuration 统一 warp count 和 shared memory 后，编译为单个 fused mixed-precision Group-GEMM kernel。适用于任何需要混合精度并行执行的场景。推广到更多精度仅需增加 micro-kernel 实现（O(|S|)），而非手工枚举所有组合（O(|S|!)）。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
