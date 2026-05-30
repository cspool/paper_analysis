## Topic-Aware MoE Serving (基于查询主题感知的MoE推理服务)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Topic-Aware MoE Serving 是 Stratum 提出的将用户查询的语义主题（topic）与 MoE expert 激活模式关联，利用 expert 在不同 topic 下的非均匀激活分布（expert affinity）来优化 expert 数据放置和请求调度的 serving 策略。核心洞察：预训练 MoE 模型在推理时表现出 domain-specific expert specialization——不同 topic 的查询（如 math vs. code）会高度偏向激活不同的 experts。Stratum 利用这一现象：(1) offline profiling 收集 per-topic expert hit rate（usage probability）统计；(2) online serving 时用轻量 topic classifier 对每个查询分类（6 个 coarse-grained topics 覆盖 93% 真实查询）；(3) scheduler 将同 topic 查询 batch 在一起 dispatch；(4) memory mapper 根据 batch 的 aggregated topic distribution 决定 expert 在 Mono3D DRAM tiers 中的放置（hot experts → fast tier, cold experts → slow tier）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Stratum Topic-Aware Serving 的端到端流程：
```
User Queries Arrive (Poisson Process, various topics)
         ↓
[Topic Classifier] — DistillBERT-based, 67M params, <10ms on CPU
  Output: topic tag ∈ {legal, humanity, CS, science, math, logic}
         ↓
[SLO-Aware Scheduler]:
  - Enqueue requests with topic tag
  - Periodically check queues:
    * Within SLO (TTFT) slack → wait for more same-topic requests
    * Near SLO deadline → dispatch current mixed-topic batch
  - Priority: batch same-topic requests together
         ↓
[Memory Mapper]:
  - Read batch's topic distribution
  - Aggregate per-topic expert hit rates from offline-profiled table
  - Algorithm 1: assign hot experts to fast Mono3D DRAM tier
  - Trigger expert swaps between batches (row-swap buffer)
         ↓
[Computation Mapper]:
  - Prefill phase → xPU (H100/RTX A6000)
  - Decode phase → Stratum NMP (Mono3D DRAM logic die)
         ↓
[Stratum Processing System]:
  - NMP executes expert FFN + attention with tiered access
  - Hot experts benefit from low tRCD (2.29ns vs 22.88ns)
         ↓
Output tokens returned to clients
```

SLO 约束下的调度策略：SLO = TTFT (Time to First Token)，定义请求从到达开始等待的最长时间。Scheduler 在每个 dispatch 决策点计算：(a) 当前队首请求的等待时间；(b) 若等待时间 < SLO slack，延长等待以收集更多同 topic 请求（提高 hot expert hit rate）；(c) 若等待时间接近 SLO，立即 dispatch（sacrifice hit rate 但满足 latency SLO）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Topic-Aware Serving 的实现需要：(1) Offline profiling——在目标 MoE 模型（如 Mixtral, OLMoE, Llama-4）上用各 topic 的数据集运行推理，统计每层每个 expert 的激活频率，构建 per-topic expert usage table；(2) Topic classifier——DistillBERT fine-tuned on 6-topic classification, with GPT-4o data augmentation for domain shift robustness；(3) Request generator——Poisson 过程模拟不同 topic 的请求到达；(4) System-level simulator——integrated request generator + scheduler + memory mapper + computation mapper + NMP cycle-level simulator。关键参数：topic classification accuracy（85.0% on Chatbot Arena），hot expert hit rate（31.6%-68.9% across models），expert swap overhead（<0.37% time）。Stratum 的方法本质是用 topic 作为 proxy 预测 expert 使用模式，将不可预测的 per-token expert routing 转化为可预测的 per-topic expert usage statistics。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
