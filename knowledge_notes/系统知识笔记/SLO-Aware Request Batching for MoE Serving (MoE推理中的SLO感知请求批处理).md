## SLO-Aware Request Batching for MoE Serving (MoE推理中的SLO感知请求批处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLO-Aware Request Batching 是 Stratum scheduler 的核心策略，在满足 Service-Level Objective（TTFT 延迟约束）的前提下，最大化同 topic 请求的 batch 规模。其核心权衡：更大的 batch = 更高的 hot expert hit rate（因 grouped same-topic requests 的 expert activation 分布更集中）= 更高的 NMP throughput；但更大的 batch = 更长的等待时间 = 可能违反 SLO。Scheduler 使用 adaptive batching window：当 enqueued same-topic requests 足够多时立即 dispatch；当等待时间接近 SLO threshold 时无论 batch 多小都 dispatch。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# SLO-Aware Batching Algorithm (per dispatch cycle)

Input: Request queues Q[6] (per-topic), SLO threshold T_slo
Output: Batch B to dispatch

max_wait = max(request.wait_time for request in all queues)
if max_wait >= T_slo * 0.9:  # Near SLO violation
    # Emergency dispatch: take all waiting requests regardless of topic
    B = {all requests in all queues with wait_time >= T_slo * 0.7}
    return B

# Normal mode: try to build same-topic batch
dominant_topic = argmax_t(len(Q[t]))
B_candidates = Q[dominant_topic]

# Check if batch is "worth it" (sufficient size for hot expert locality)
if len(B_candidates) >= min_batch_size:
    B = B_candidates[:max_batch_size]
    return B

# If dominant topic too small, wait or include mixed topics
if max_wait < T_slo * 0.5:  # Still have slack
    wait_next_cycle()        # Hope more same-topic requests arrive
else:
    # Time running out: include other topics to fill batch
    B = B_candidates
    for t in other_topics_sorted_by_queue_size:
        B.extend(Q[t][:remaining_capacity])
    return B
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SLO-aware batching 的实现权衡：(1) Batch size vs. hot expert hit rate——相同 topic batch 越大，aggregated expert usage distribution 越集中（hot experts 越突出），Algorithm 1 能将更多高频 expert 放入快 tier；(2) Wait time vs. SLO slack——topic classifier 的 85% accuracy 意味着 15% 的请求被错误分类，但这些请求仍被 batch 在同一 topic group 中（因为它们被分类为该 topic），对 expert hit rate 的影响取决于分类错误是否导致错误的 expert affinity 假设；(3) Mixed-topic batch penalty——当 SLO 压力迫使 dispatch mixed-topic batch 时，不同 topic 的 hot expert 集合不同，导致 tier placement 的 benefit 被稀释（需要将多个 topic 的 experts 都放入快 tier，快 tier 容量有限），这是 Stratum 在 strict SLO 下的主要性能退化来源。论文中 batch size=1（最坏情况）下的 hot expert hit rate 仍能维持一定水平（因单个 topic 内的 expert usage 仍高度 biased）。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
