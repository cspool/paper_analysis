## VSLPipe (Versatile Pipeline / 通用流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VSLPipe 是 MoE-Lens 的执行引擎核心，一种为 CPU-GPU 混合 MoE 推理设计的 software pipeline。两层创新：(1) Compute Graph Division——将 MoE transformer layer 重组为 GA (QKV proj + GPU Flash Attn) → C (CPU Decode Attn + KV Cache Store) → GB (O proj + MoE layer)，跨 layer 合并为 execution stage（每 stage = CPU-only phase + GPU-only phase）；(2) Dual-phase Pipeline——α/β 两组 token 交替执行，CPU attention 与 GPU GEMM 完全重叠。Pipeline 结构：prologue + N-1 main stages + epilogue，总 iterations 由 Equation 12 计算。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
VSLPipe 单 stage 执行流程：
- Phase 1: CPU 执行 α 组 decode attention ‖ GPU 执行 β 组 prefill GA (QKV + Flash Attn)
- D2H/H2D sync: offload KV values to CPU, load attention results to GPU
- Phase 2: GPU 执行 α 组 GA + GB (O proj + MoE) ‖ CPU 执行 β 组 decode attention
- 每个 stage 开始时 Contiguous Data Mover 预取下一 stage weights（独立线程，100MB packet 粒度）
- D2H/H2D per-stage data bounded by $2n(d + \frac{2d}{s})$，n=18500 时约 200MB

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **PyTorch + CUDA 实现**，手写 AVX512 CPU attention kernel（§6.6），CPU/GPU 之间通过 D2H/H2D transfer 交换 attention 结果和 KV cache 数据。
- **与其他 pipeline 的区别**：MoE-Lightning 分离 prefill/decode 为两阶段；MoE-GEN 的 ping-pong pipeline 在 module level 做 overlap；VSLPipe 跨 layer 重组 compute graph 实现 CPU-GPU 完全重叠。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
