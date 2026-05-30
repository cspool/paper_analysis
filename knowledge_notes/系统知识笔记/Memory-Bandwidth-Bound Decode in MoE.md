## Memory-Bandwidth-Bound Decode in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Memory-Bandwidth-Bound Decode 是 LYNX 识别并解决的 MoE 推理核心瓶颈。在 auto-regressive decode 阶段，每个 iteration 每个请求仅生成 1 个 token，导致 expert computation 的 arithmetic intensity（计算量/内存访问量之比）极低。对于 Qwen2-57B (k=8, N=64)，batch B=16 时 arithmetic intensity = B×k/N = 16×8/64 = 2 FLOPs/byte，远低于 H200 GPU 的计算能力（~67 TFLOPS BF16 vs ~3.35 TB/s HBM bandwidth，ratio ≈ 20 FLOPs/byte）。因此 decode 阶段 latency 由 HBM 带宽而非 GPU 计算能力决定，latency 与 active expert 数量成正比。

LYNX 通过实验量化了该瓶颈的严重程度：在 25ms TPOT SLO 下，median decode iteration 的 42% 时间花在从 HBM 获取 expert 权重上。随着 GPU compute 增速持续超过 memory bandwidth 增速（H200 的 compute/bandwidth ratio 远高于 A100），该瓶颈将日益严重。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
┌── MoE Decode Memory Bandwidth Analysis ────────────────────┐
│                                                              │
│  Decode iteration 时间分解（Qwen2-57B, B=16, H200）:        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Component             │ Time (ms) │ % of Total      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Attention (FlashAttn) │   3.2     │     19%         │    │
│  │  Expert Weight Load    │   7.1     │     42%  ← 瓶颈  │    │
│  │  Expert Computation    │   4.5     │     27%         │    │
│  │  Other (norm, router)  │   2.0     │     12%         │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Total                 │  16.8     │    100%         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Arithmetic Intensity 分析：                                  │
│  - Expert computation FLOPs ≈ B×k×2×d_model×d_ff            │
│  - Expert weight bytes = active_experts×2×d_model×d_ff×2B   │
│  - AI = FLOPs/bytes ≈ B×k/(active_experts×2)  (BF16)        │
│  - 当 AI < GPU compute/bandwidth ratio → memory-bound       │
│                                                              │
│  LYNX 的解决方案: 减少 active_experts:                       │
│  - Baseline: 55-60 active experts → AI ≈ 0.27                │
│  - LYNX: ~15 active experts → AI ≈ 1.07                     │
│  - 虽然仍 memory-bound，但 HBM 数据搬运量减少 ~73%           │
│  - Net latency 降低 1.09-1.30x                               │
└──────────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

解决 memory-bandwidth-bound decode 的现有方法分为三类：(1) 减少 expert 大小——pruning, quantization（LYNX 与此互补，在 INT4 量化上额外提供 7-10% speedup）；(2) 减少加载的 expert 数——per-token k reduction, expert clustering（但不在 batch 级别操作）；(3) 使用更快的内存——HBM3e, 更大的 on-chip cache。LYNX 提出的 batch-level dynamic expert reduction 是第四种方法，与 (1)(3) 正交。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
