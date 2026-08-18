## 抢占式调度（Preemption / 调度器降级：KV 容量触发的 Running→Waiting 重排）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
抢占式调度是 LLM serving 调度器在内存耗尽（"Reasoning Cliff"）时的防御机制：当活跃请求的聚合 KV cache 逼近 HBM 上限时，调度器把部分 Running 请求降级到 Waiting 队列（或 swap 到 CPU 主机内存）以释放 KV 块、避免 OOM。它本质是"内存流量整形"——把 KV 容量作为调度约束，牺牲活跃请求的执行连续性换取系统不崩溃。本论文把抢占与重算惩罚定性为 reasoning 负载尾部延迟尖峰的直接来源。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# vLLM 调度器在 KV 压力下的抢占流程（reasoning 负载，本论文观测）
while True:
    kv_usage = aggregated_kv_occupancy()          # 聚合 KV 占用（PagedAttention 块级统计）
    if kv_usage >= HBM_threshold:                 # 逼近 100%（10K 并发下数分钟即到）
        victim = select_preemptible(Running)      # 按调度策略选抢占对象
        demote(victim, Running → Waiting)         # 释放其 KV 块；或 swap 到 CPU 主机
    if prefix_cache_hit(victim):                  # 恢复时先试 prefix caching
        resume_with_partial_recompute(victim)
    else:
        full_prefill_recompute(victim)            # 命中失败 → 全量 prefill 重算（尾部延迟灾难）
```
Annotations：抢占触发条件是聚合 KV 占用而非计算负载；恢复路径依赖 prefix cache 命中，内存耗尽时通常失败→重算；重算惩罚使 E2E 尾部延迟失控（论文 Fig.3d 的 Waiting/Running 振荡与 TPOT 恶化）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
vLLM 等引擎内置抢占（swap/preempt 策略，可配 swap space）；论文的工程建议是避免触发而非优化恢复——(1) KV-aware 并发帽：按 HBM headroom、活跃序列长度、预期 decode KV 增长设定 max_num_seqs 上限（论文 8B 模型最优 ≈2K 而非 10K）；(2) admission control：在请求准入时估计其未来 KV 增长并预留 decode 容量（"reasoning cliff" 在长输出下会前移到 prefill 期，batch 4K/5K 时准入即失败）；(3) DP 场景配 memory-aware routing 防各 replica 独立进入抢占重负载。论文结论：并发提高占用率只在 KV 未饱和前有效，超过即触发抢占与重算、吞吐增益崩塌——调度器应把抢占率作为一阶服务质量指标。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
