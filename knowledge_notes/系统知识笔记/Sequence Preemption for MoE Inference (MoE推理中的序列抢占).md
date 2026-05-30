## Sequence Preemption for MoE Inference (MoE推理中的序列抢占)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sequence Preemption 是 MoE-Lens Resource-Aware Scheduler 的 Preemption Mode 机制：当 KV cache capacity 不足以容纳所有 active decode sequences 时，选择部分 sequences 暂停执行，回收其 KV cache blocks，分配给剩余 sequences；被抢占的 sequences 重新注入 Prefill Scheduler 队列，需 re-prefill 其 KV cache。利用 prefill/decode overlapping 隐藏 re-prefill 的重计算开销（re-prefill 与正常 decode 并行）。与 vLLM 的 swap-based preemption 不同：MoE-Lens 丢弃 KV cache 并 recompute（因 CPU memory 已满，无法存储额外 preempted KV cache）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
触发流程：Decode Scheduler 每次 inference pass 后估算下一 pass 所需 KV cache blocks → 若需求 > 可用 → Preemption Mode → select subset for preemption → reclaim KV cache → 抢占 sequences 作为新 prefill 重新注入 pipeline。Preemption 频率与 KV cache 容量成反比：70GB KV cache + g=64 时约一半时间在 preemption，throughput 波动显著；210GB + g=64 时极少 preemption，throughput 平滑（图 13）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 GPU memory 中维护 scheduling state，实时 track 每 sequence 的 KV cache block 使用。
- 适用场景：resource-constrained batch processing，KV cache 容量不足以支持所有 sequences 同时 decode。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
