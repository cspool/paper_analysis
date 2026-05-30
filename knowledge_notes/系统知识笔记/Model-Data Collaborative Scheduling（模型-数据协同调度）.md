## Model-Data Collaborative Scheduling（模型-数据协同调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model-Data Collaborative Scheduling 是 Sem-MoE 提出的核心调度范式，将 MoE 推理中的 **模型维度（expert placement）** 和 **数据维度（token/request routing）** 联合优化，而非传统的分离处理。传统方法将两者视为独立问题：SGLang/vLLM 的 expert placement 采用简单的轮询策略，请求调度采用 continuous batching/FCFS，两者互不感知。Sem-MoE 利用 token-expert activation affinity 作为桥梁，通过 offline co-clustering 同时求解 expert-to-device 和 token-to-device 的最优映射，再在 online 阶段协同执行。目标函数为加权和：min θ·load_imbalance + (1-θ)·remote_activation，受限于每个 token 和 expert 的唯一分配约束。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

协同调度的三层对应关系：

```
┌─────────────────────────────────────────────┐
│            Offline Co-Clustering            │
│  Input: C_p (token-expert confidence table) │
│  Output: E (expert→device), T (token→device)│
├─────────────────────────────────────────────┤
│   Model Scheduling          │  Data Scheduling          │
│  (expert placement)        │  (token/request routing)  │
├────────────────────────────┼───────────────────────────┤
│ Layer 1: Experts clustered │ Attention-DP: requests    │
│ by co-activation pattern   │ scheduled to affine device │
│ Layer 2: Gate matrix cols  │ Attention-TP: tokens      │
│ shuffled for transparency  │ shuffled via SRS kernel    │
│ Layer 3: ...               │ Layer 3: ...              │
└────────────────────────────┴───────────────────────────┘
```

Attention-DP 场景：每个请求 r 根据其 token 组成的 aggregate affinity score 被分配到最匹配的 DP rank。请求级聚合公式：S_r = argmax_j Σ_{token_i∈r} R_{ij}。配合 workload-aware round-robin 保证每 E 个请求一轮，各 rank 分配一个，防止解码阶段负载偏斜。

Attention-TP 场景：利用 inter-layer expert-expert affinity（2-gram Markov chain）增强 token 级预测。当某 token 的 token-level confidence 低时（如 OOV token），切换到 device-sequence-based prediction：Pr(D_k^(L)|D^(L-1), D^(L-2))。两表竞争机制选置信度高者。

Memory overhead：token-to-device table T 约 11.72 MB（DeepSeek-V2, int16），完全驻留 GPU。Online inference 中查询为 O(1) 查表操作，额外计算仅 argsort + tensor shuffle。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现为 SGLang 插件模块：offline solver 用 Python 实现（交替优化算法，Algorithm 1），online scheduler 扩展 SGLang request scheduler + 自定义 Triton SRS/SAG kernel。Offline solver 在 CPU 上运行（一次性，部署前），online 查表在 GPU 上 O(1) 完成。20% 数据用于训练预测器生成调度表，80% 用于在线推理评估。Cross-dataset transfer 验证了调度表的泛化性。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
