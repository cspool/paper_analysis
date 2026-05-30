## Dynamic Parallelism Transition Strategy (with INT4 Quantized Weight Backup)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Parallelism Transition Strategy 是 HAP 提出的在 MoE 推理中从 prefill 阶段到 decode 阶段切换 Expert 模块并行策略时，最小化切换开销的方法。当 HAP 的 ILP 求解器为 prefill（通信瓶颈，选 EP/DP）和 decode（计算瓶颈，选 TP）分配不同策略时，Expert 模块权重（约占 90% 总参数）需要重新分布。直接通过集合通信（AllGather/AllToAll）重分布权重的开销可能抵消策略切换带来的收益。HAP 提出两种过渡方式并动态选择更优方案：(1) 通过集合通信重分布权重（T_reshard）；(2) 从 CPU memory 中维护的 INT4 per-group 量化权重备份异步上传到 GPU 并反量化（T_upload + T_dequant）。决策公式：C_ij = min(T_reshard, max(0, T_upload + T_dequant - (T_attn + T_experts + T_comm)))，即上传+反量化可与当前层计算重叠时，过渡开销可被完全隐藏。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
┌──────────────────────────────────────────────────────────────┐
│ HAP 动态策略切换系统架构 (Mixtral-8x7B, 4×A6000)              │
│                                                              │
│ Phase 1: Prefill (Expert=EP, Attn=DP)                        │
│   GPU0: experts 0,1    GPU1: experts 2,3                      │
│   GPU2: experts 4,5    GPU3: experts 6,7                      │
│   ┌─ Attention (DP, no comm) ─┐                               │
│   │  各GPU独立计算 batch/4     │                               │
│   └───────────────────────────┘                               │
│   ┌─ Expert (EP, A2A dispatch/combine) ────────────────────┐ │
│   │  All-to-All dispatch: tokens路由到expert所在GPU          │ │
│   │  各GPU计算本地expert FFN                                 │ │
│   │  All-to-All combine: 结果返回                            │ │
│   └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│ Phase 2: 策略切换决策                                          │
│   T_reshard = estimate_reshard_time(all_expert_weights)      │
│   T_upload = estimate_upload_time(int4_weights)               │
│   T_dequant = lookup_dequant_dict(V_dequant)                  │
│   overlap = T_attn + T_experts + T_comm                      │
│   if T_reshard < max(0, T_upload + T_dequant - overlap):     │
│       method = "reshard"  # 直接AllGather重分布               │
│   else:                                                       │
│       method = "quantized_upload"  # CPU→GPU异步上传+反量化    │
│                          ↓                                    │
│ Phase 3: CPU→GPU 量化权重上传 (method = "quantized_upload")    │
│   ┌─ CPU Memory (pinned) ──────────────────────────────────┐ │
│   │  INT4 weight backup: per-group quantized, ~1/4 FP16     │ │
│   │  Multi-stream async upload:                              │ │
│   │    stream0: experts 0,1 → GPU0                           │ │
│   │    stream1: experts 2,3 → GPU1                           │ │
│   │    stream2: experts 4,5 → GPU2                           │ │
│   │    stream3: experts 6,7 → GPU3                           │ │
│   └─────────────────────────────────────────────────────────┘ │
│   ┌─ GPU Dequant Kernel ───────────────────────────────────┐ │
│   │  各GPU收到INT4权重后:                                     │ │
│   │    per-group dequant: BF16 = INT4 × group_scale         │ │
│   │    (group_size=128, scale per-group in BF16)            │ │
│   │  Dequant与第32层prefill计算重叠执行                       │ │
│   └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│ Phase 4: Decode (Expert=TP, Attn=DP)                          │
│   GPU0-3: 各持有完整expert的1/4中间维度                        │
│   ┌─ Expert (TP, AllReduce) ───────────────────────────────┐ │
│   │  各GPU计算部分FFN → AllReduce聚合                        │ │
│   │  单token通信量小，TP负载均衡优势明显                       │ │
│   └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

INT4 per-group 量化方案选择（Table I）：per-tensor 量化 GSM8K 从 58.3→55.6%（降 4.6%），per-group 量化 GSM8K 仅降至 58.0%（降 0.5%），MMLU 保持 67.7%。Per-group 量化以 128 元素为组，每组独立 scale factor，在精度保持和压缩率之间取得最优平衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HAP 使用 bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes) 实现 INT4 per-group 量化。CPU 侧维护量化备份于 pinned memory（支持快速 PCIe DMA 传输）。T_dequant 字典在初始化阶段通过 microbenchmark 构建：key = V_dequant（需反量化的参数量），value = 实测反量化时间。运行时根据实际 Expert 模块策略切换涉及的参数量查表。多 stream 异步上传使用 CUDA cudaMemcpyAsync，GPU 反量化 kernel 在独立 stream 上执行，与 prefill 最后几层的 attention/expert 计算重叠。该方案的额外内存开销为 ~1/4 FP16 expert 权重 + scale factors（约 5% 额外开销），但避免了 T_reshard（全部 expert 权重经 PCIe/NVLink 重分布）的高通信开销。

涉及论文标题：
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
