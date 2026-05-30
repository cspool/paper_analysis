## Module-Based Batching（模块级批处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Module-Based Batching 是 MoE-GEN 提出的 MoE 推理批处理策略，核心思想是将 MoE 模型的 attention 和 expert 两类计算密集型模块**解耦**，分别为其分配不同的 micro-batch size（$b_a$ 和 $b_e$），通过多轮 attention 小批次累计 token，最终在 expert 模块形成大 batch 一次性执行。这与传统的 model-based batching（整个 forward pass 使用统一 batch size）形成对比：model-based batching 下 batch size 受 attention 模块的 peak memory 限制（如 DeepSeek-V2 仅为 8），导致每个 expert 在解码阶段仅处理 <10 个 token，GPU FLOPs 利用率低至 0.1%。Module-based batching 将 attention 和 expert 的 memory 需求解耦：attention 以 $b_a$（如 75）为微批次循环多次（约 $B/b_a$ 轮），累计所有 micro-batch 的输出 token 后，在 expert 阶段以 $B$（如 3640）的大 batch 一次性执行所有 experts。图 2 直观展示了两种策略的区别：model-based batching 的 batch 在穿过整个模型时大小不变；module-based batching 则在 attention 和 expert 模块之间增量构建 batch。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MoE-GEN 的 system architecture 围绕 module-based batching 组织为一个两阶段引擎（以解码阶段的单层为例）：

```
阶段1: Attention 积累 (Micro-batch = b_a)
  for i in 1..(B/b_a):   // 如 49 轮
    1a. Pre-Attention: QKV projection on GPU（batch = b_a）
    1b. Self-Attention: 按 split ratio ω 分流
        - CPU path (ω·b_a tokens): AVX kernel 直接读 host KV-cache
        - GPU path ((1-ω)·b_a tokens): HtoD copy KV-cache → GPU 计算
    1c. Post-Attention: output projection（concat CPU+GPU results）
    1d. HtoD engine: 预取下一轮 attention weights
    每个 attention micro-batch 的结果累积到 host memory

阶段2: Expert 批处理 (Accumulated Batch = B)
  2a. Router: 所有 B 个 token 统一通过 gating
  2b. for each expert j in 1..E:
        HtoD engine: prefetch expert_j weights → S_Expert GPU buffer
        GPU: compute expert_j on its assigned tokens
        （大 batch 下 token 均匀分配，每个 expert 收到 ~B·k/E tokens）
```

关键系统组件：Batching Scheduler（搜索最优 B, b_a, b_e, ω 配置）→ 分配 GPU memory buffers（KV-cache buffer, expert buffer, dense buffer）→ MoE-GEN Engine 执行 module-based batching loop → HtoD/DtoH engines 异步执行 memory copy。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Module-based batching 在 MoE-GEN 中通过以下方式实现：
- **实现语言**：约 3000 行 C++ 和 2000 行 Python，集成 HuggingFace generation pipeline。
- **Batch size 搜索**：offline profiling 各模块在不同 batch size 和 sequence length 下的 latency 和 peak memory，然后在搜索空间（B, b_a, b_e, ω, S_Expert, S_Params）中枚举，通过 DAG+DP 估算 critical path 选择吞吐最优配置。
- **适用场景**：offline high-throughput inference（benchmarking、data wrangling、feature extraction），不适合 interactive/latency-sensitive 场景（首 token 延迟高）。需要足够的 host memory 存储完整模型 + KV-cache（如 DeepSeek-V2 236B 约需 >400GB host memory）。
- **局限性**：batch size 受 host memory 容量约束（长 context 下 B 减小）。小 batch (≤32) 下优势减弱，因为 MoE-GEN 为预留大 batch 空间而 aggressively offload，在小模型/小 batch 场景下额外 offloading overhead 可能抵消收益。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching
