## Per-Expert Offloading (vs Per-Layer Offloading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Per-Expert Offloading 是一种 MoE 专用的参数卸载策略，以单个 expert 为粒度进行 GPU/host 内存管理，而非以整个 Transformer 层为粒度。传统 per-layer offloading（如 HuggingFace accelerate 的 `device_map="auto"`）将每层所有参数（含全部 expert）作为一个整体加载/卸载——对于 Mixtral-8x7B，这意味着每次加载 8 个 expert（每个约 5.6B 参数在 FP16）但只使用 top-2，浪费 75% 的 PCIe 带宽。Per-expert offloading 仅加载 gate 选中的 top-k expert，大幅减少 host-to-device 传输量。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Mixtral-8x7B 16GB GPU + per-expert offloading 的内存布局：

```
┌────────────────────────────────────────────────────┐
│ GPU VRAM (16GB, fixed resident):                    │
│   - Embedding & LM head (FP16)                      │
│   - 32 × Attention blocks (4-bit quantized)         │
│   - 32 × MoE gates (FP16)                          │
│   - 32 × k cached experts (2/3-bit quantized)       │
│   - b=4 shared device buffers (expert-sized)        │
│   - KV Cache (inference activations)                │
├────────────────────────────────────────────────────┤
│ Host RAM (pinned):                                  │
│   - 32 × 8 experts (2/3-bit quantized)              │
│   - Contiguous per-expert buffers                   │
├────────────────────────────────────────────────────┤
│ Per-token execution flow:                           │
│   1. Gate selects top-2 experts                     │
│   2. Check per-layer LRU cache for each expert      │
│   3a. Hit → use directly from GPU buffer            │
│   3b. Miss → load from pinned RAM via PCIe DMA      │
│   4. Async prefetch next layer's predicted experts  │
│   5. Expert FFN computation                        │
└────────────────────────────────────────────────────┘
```

对比 per-layer offloading（如 accelerate）:
- Per-layer: 加载 8 experts × d_model² 字节 = 浪费 6/8 带宽
- Per-expert: 加载 0-2 experts × d_model² 字节 = 带宽利用率高
- Per-expert 需维护 per-layer cache 状态（LRU 队列），但 overhead 可忽略

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Expert 参数在 pinned memory 连续分配（`tensor.pin_memory()`），单次 `cudaMemcpyAsync` 完成传输
- GPU 侧分配 b=4 个临时 expert 大小 buffer，所有层共享（非 k×32 个），显著减少 GPU memory footprint
- 当 host RAM 不足时（如 Colab），expert 在 host RAM 和磁盘间换入换出，GPU cache 淘汰的 expert 写回 host RAM
- 代码开源：https://github.com/dvmazur/mixtral-offloading

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

HOBBIT/MoE-APEX 对 Per-Expert Offloading 的扩展（Mixed Precision Expert Cache）：
- **多精度 Expert Cache**：GPU memory 中维护两个分离 cache——High-Precision Cache（较大，存放 FP16/INT8 expert）和 Low-Precision Cache（较小，存放 INT4/INT2 expert）。分离管理避免低精度 expert 挤占高精度空间。
- **动态精度选择**：cache miss 时，基于 ||G(x)|| 的 unimportance degree score 决定加载精度：s≤T1→高精度, T1<s≤T2→低精度, s>T2→跳过。Expert Loader 通过 read() 系统调用从 CPU memory 加载对应精度版本。
- **多维缓存策略**：LHU + LRU + LFU + FLD 加权组合替代单一 LRU/LFU，高精度 miss 代价为 1，低精度为 B_l/B_h（如 1/4），策略目标是最小化加权 miss penalty 而非单纯 hit ratio。
- **序列级统计**：缓存统计（LRU/LFU/LHU 记录）在新 sequence 开始时重置，因为不同 sequence 的 expert 偏好分布不同（序列级 LFU 比模型级 LFU hit ratio 提升 4.5%）。
- 硬件配置示例（RTX 4090 24GB）：non-expert ~4GB，expert cache ~15GB（高精度 ~12GB + 低精度 ~3GB），KV cache ~3GB，余量 ~2GB。
