## MoE Disaggregated Inference (Attention-MoE Disaggregation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Disaggregated Inference 将 MoE 模型的 Attention 层和 MoE (FFN Expert) 层部署到独立的 GPU 子集群上，使每种层类型可以独立配置并行度（tensor parallelism degree、data parallelism degree）和资源规模（GPU 数量）。这与 monolithic deployment（所有层共享同一并行配置和 GPU pool）形成对比。

JANUS 论文的核心论点：Attention 层和 MoE 层在不同 batch size 和并行度下表现出截然不同的 scaling 行为——Attention 层在小到中等 batch size（B=16, B=64）下增加并行度几乎不降低延迟；而 MoE 层在所有 batch size 下都从更大并行度中持续受益（虽然 speedup 仍 sublinear）。强制两者使用相同并行度要么导致 Attention 资源浪费，要么 MoE 资源不足。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

解耦架构的全栈执行流程（以 JANUS 1A6E 配置为例）：

```
Cluster Layout:
  Attention Sub-Cluster: n_a=1 GPU (持有完整 Attention 权重 + Shared Expert)
  MoE Sub-Cluster:       n_e=6 GPU (持有 Expert 权重子集, 每GPU ≤ C experts)

Execution Flow (每 decode step, 单 MoE layer):
  1. Attention GPU 执行 Attention (MLA) + Shared Expert FFN
     (Shared Expert 计算与跨子集群通信 overlap)
  2. Attention GPU → NVSHMEM one-sided put → MoE GPU E0..E5
     发送完整 activation (非 per-expert packed)
  3. MoE GPU 本地执行 Gating (softmax top-K routing)
  4. AEBS GPU kernel: 将逻辑 expert IDs 映射为物理 replica IDs
  5. MoE GPU 本地执行 Expert FFN (仅计算本地持有的 experts)
  6. MoE GPU → NVSHMEM put → Attention GPU 返回结果
  7. Attention GPU: residual add → next layer

Key Configuration Space:
  n_a ∈ [1, n_max]: attention 实例数（每 GPU 一个实例）
  n_e ∈ [⌈E/C⌉, n_max]: MoE 实例数（E=总 expert 数, C=每 GPU capacity）
  Expert Placement: 哪些 expert replicas 放在哪些 MoE GPU 上
```

对比 Monolithic Deployment (SGLang, vLLM):
```
  所有 GPU 持有: Attention 权重 + 部分 Expert 权重 + KV Cache
  所有层使用相同的 TP/EP degree
  Scaling unit = 完整模型副本（如 DeepSeek-V3: 最小 16 H100）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点（基于 JANUS ~4K 行 Python + ~300 行 CUDA/C++ on SGLang）：
1. **Attention Side**: 复用 SGLang 的 continuous batching、request dispatching、RadixAttention KV-cache 管理
2. **MoE Side**: 每个 MoE instance 运行 AEBS GPU kernel + 本地 expert FFN
3. **Communication**: NVSHMEM one-sided putmem_signal/signal_wait + NCCL intra-node collectives
4. **Gating Location**: 放在 MoE 侧以减少 attention 侧 per-expert tensor packing 开销
5. **Controller**: Attention Controller (请求分发) + MoE Controller (expert placement + scaling 决策)

其他解耦式 MoE 推理系统对比：
- MegaScale-Infer: 解耦但 attention 侧 gating + 粗粒度 scaling
- xDeepServe: NPU superpod 解耦 + EPLB 调度 + 无 scaling policy
- EaaS: 弹性通信通道重构 + 无 scaling 优化

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
