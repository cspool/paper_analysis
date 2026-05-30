## Arithmetic Intensity and Roofline Model for LLM Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Arithmetic Intensity 定义为计算操作数（FLOP）与内存操作数（bytes transferred）的比值（FLOP/byte），用于判断 workload 是 compute-bound（高 arithmetic intensity）还是 memory-bound（低 arithmetic intensity）。Roofline Model 是性能分析工具：以 arithmetic intensity 为 x 轴，attainable performance (FLOP/s) 为 y 轴，绘制由 peak compute throughput 和 peak memory bandwidth 构成的"屋顶"曲线。SqueezeLLM 用 roofline model 分析 LLM 推理：单 batch 生成时每个权重 2 FLOP（乘+加）对应 2 bytes 加载（FP16）→ arithmetic intensity = 1 FLOP/byte，远低于 GPU 的"ridge point"（如 A5000: 222 TFLOPS / 768 GB/s ≈ 289 FLOP/byte），确认推理在 memory-bound 区域。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
Roofline Model for LLaMA-7B on A5000 GPU:

Peak Compute:  222 TFLOPS
Peak Memory BW: 768 GB/s
Ridge Point:    222/0.768 ≈ 289 FLOP/byte

LLM GEMV (batch=1):
  Work: 2 × d_model × d_ff FLOP (per FFN layer)
  Bytes: 2 × d_model × d_ff bytes (FP16 weights) + d_model bytes (activation)
  Arithmetic Intensity ≈ 1 FLOP/byte << 289 ridge point
  → Deep in memory-bound region
  → Performance bounded by: peak_memory_BW × arithmetic_intensity

量化改进:
  3-bit weights: Bytes reduced ~5.3x → effective arithmetic intensity ↑
  → 向上沿 memory-bound slope 移动 → 性能提升 ∝ 压缩比
  (但仍在 ridge point 以下 → 仍是 memory-bound)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Roofline analysis 的实现方法：(1) 测量或查询 GPU 的 peak compute (TFLOPS) 和 peak memory bandwidth (GB/s)；(2) 计算 target kernel/workload 的 arithmetic intensity；(3) 判断在 roofline 的哪个区域。SqueezeLLM 使用简化版 roofline model（Fig. 2）：固定 batch=1 推理的 arithmetic intensity，仅变化 weight bitwidth，观测延迟→确认线性关系→验证 memory-bound。这对量化设计的指导意义重大：在 memory-bound 区域，压缩比是唯一可改善性能的杠杆（不依赖更快的 compute），因此应优先考虑能最大化压缩比的方法（非均匀量化可优于均匀量化），而非追求整数算术的效率（均匀量化的主要优势在此场景下无意义）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
