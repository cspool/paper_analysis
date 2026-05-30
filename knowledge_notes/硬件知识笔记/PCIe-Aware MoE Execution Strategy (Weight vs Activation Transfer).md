## PCIe-Aware MoE Execution Strategy (Weight vs Activation Transfer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PCIe-Aware MoE Execution Strategy 是一种在 GPU 显存不足时，根据 PCIe 带宽和数据传输量做出 CPU/GPU 执行决策的硬件感知推理优化。Fiddler 论文识别出 MoE 推理中两种可选的数据传输策略：(1) **Weight Transfer**：通过 PCIe 将 expert 权重（~300MB/expert for Mixtral-8x7B FP16，3 个 4096×14336 矩阵）从 CPU memory 拷贝到 GPU memory，然后在 GPU 执行计算；(2) **Activation Transfer**：通过 PCIe 将 activation（s × 4096 × 2 bytes for BF16，s 为输入 token 数）从 GPU memory 拷贝到 CPU memory，在 CPU 执行计算后传回结果。

策略选择的核心依据是 PCIe 带宽约束：对于 small batch（s 小），activation 传输量极小（s=1 时仅 8KB），远小于 weight 传输量（300MB），因此 CPU 执行更优；对于 large batch（s 大），CPU 计算延迟线性增长超过 weight 传输 + GPU 计算的总延迟，因此 GPU+transfer 更优。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Fiddler 两种策略的硬件数据流对比（以 Mixtral-8x7B expert 为例）：

```
Strategy (b): Weight Transfer → GPU Execution
┌──────────┐     PCIe (~32-64GB/s)     ┌──────────┐
│ CPU RAM  │ ──── W (300MB) ────────→  │ GPU VRAM │
│ (pinned) │                           │          │
└──────────┘                           └──────────┘
                                            │
                                     GPU SM compute
                                     (Tensor Cores)
                                            │
                                       output ready
                                   
数据传输量: 300MB (恒定, 与 s 无关)
计算位置: GPU (高吞吐, 延迟恒定)
瓶颈: PCIe BW (2-5× GPU computation time)

Strategy (c): Activation Transfer → CPU Execution
┌──────────┐     PCIe (~32-64GB/s)     ┌──────────┐
│ GPU VRAM │ ── A (s×4096×2B) ────→  │ CPU RAM  │
└──────────┘                           └──────────┘
                                            │
                                     CPU AVX512 compute
                                     (SIMD cores)
                                            │
┌──────────┐     PCIe (~32-64GB/s)     ┌──────────┐
│ GPU VRAM │ ←─ output (s×4096×2B) ── │ CPU RAM  │
└──────────┘                           └──────────┘

数据传输量: s × 8KB (线性增长)
计算位置: CPU (低吞吐, 延迟线性增长)
瓶颈: CPU compute (for large s); PCIe BW (negligible)

决策阈值:
s_threshold = (gpu_const + trans_const) / cpu_slope
s < s_threshold → Strategy (c)
s ≥ s_threshold → Strategy (b)
```

Fiddler 评估的两种 PCIe 配置：
| Environment | PCIe | Bandwidth | Weight transfer (300MB) |
|-------------|------|-----------|------------------------|
| Env1 | Gen3 x16 | 32 GB/s | ~9.4 ms |
| Env2 | Gen4 x16 | 64 GB/s | ~4.7 ms |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Pinned memory**：CPU 侧 expert 权重使用 pinned memory 分配（`tensor.pin_memory()`），GPU DMA engine 可直接通过 PCIe 读取，无需 CPU staging buffer
- **异步传输**：使用 `cudaMemcpyAsync` 实现 CPU↔GPU 数据传输与计算 kernel 的潜在重叠（在独立 CUDA stream 上）
- **Latency measurement**：Fiddler 初始化阶段测量 trans_lat()（weight copy latency）和 activation copy latency（后者被证明 <1% of CPU single-input latency，在 latency model 中忽略）
- **适用条件**：仅当 GPU VRAM < total model size（即必须有部分参数在 CPU memory）时相关；若模型完全 fit in GPU memory，则始终选 Strategy (a)
- **硬件演进**：PCIe 带宽随世代翻倍（Gen3: ~32GB/s, Gen4: ~64GB/s, Gen5: ~128GB/s），更高带宽使 Strategy (b) 更优的 s_threshold 下移，减少 CPU 执行的需求

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
