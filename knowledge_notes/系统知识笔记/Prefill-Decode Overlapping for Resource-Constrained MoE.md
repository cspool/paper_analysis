## Prefill-Decode Overlapping for Resource-Constrained MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill-Decode Overlapping 是 MoE-Lens 提出的调度策略：在 CPU-GPU 混合 MoE 推理中，将 prefill 和 decode 阶段的序列交错调度，使 GPU GEMM、CPU attention 和 weight transfer 三者并行。与 MoE-Lightning 的分离调度（先完成所有 prefill 再 decode）不同，MoE-Lens 允许 decode 在部分 prefill 完成后即开始，prefill 和 decode 在同一 pipeline iteration 中混合执行。Equation 7 量化了 overlapping 对 KV cache 容量的放大：$C_{KV,eff} = \frac{p+g}{p+g/2}C_{KV}$。例如 p=98, g=256 → 有效容量放大 1.57×（等效于不增加物理内存的前提下支持更多并行序列）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MoE-Lens 的 Resource-Aware Scheduler 通过 Normal Inference Mode 实现 overlapping：
1. Decode Scheduler 先调度所有 active decode sequences
2. Prefill Scheduler 根据 $n_{real}$（GPU 饱和 token 阈值，Pipeline Profiler 实测）计算可额外调度的 prefill tokens
3. VSLPipe 将 prefill 和 decode tokens 分成 α/β 两组，每组 CPU attention 与另一组 GPU GEMM 重叠（每个 stage 分 CPU-only phase + GPU-only phase）
4. 当 KV cache 不足时切换到 Preemption Mode——抢占部分 decode sequences，利用 overlapping 隐藏 re-prefill 的重计算开销

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：Resource-Aware Scheduler（GPU-resident）、VSLPipe execution engine（§6.4）、Contiguous Data Mover（独立线程 weight transfer）。
- **与 continuous batching 的区别**：Continuous batching 面向 online serving 的 TTFT 优化（GPU-only），MoE-Lens overlapping 面向资源受限环境的 throughput 优化，需协调 CPU attention + GPU GEMM + weight transfer 三者并行。
- **效果**：70GB KV cache, g=32 时 GPU utilization ~90%；MoE-Lightning 的分离调度仅 16.5%。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
