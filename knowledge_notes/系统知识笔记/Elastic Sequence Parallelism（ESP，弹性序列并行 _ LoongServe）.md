## Elastic Sequence Parallelism（ESP，弹性序列并行 / LoongServe）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ESP（Elastic Sequence Parallelism）是 LoongServe（SOSP 2024，Peking University）提出的请求级动态序列并行：把集群中所有实例组成统一 SP 池（共享相同 TP 大小），为每个请求批动态分配不同 SP 大小，在请求到达时按资源需求"弹性"调整 SP 度，无需重分片 LLM 参数（SP 只重分 token）。相对静态配置 SP 的非 SP 系统，ESP 能响应高度动态的请求长度需求；相对固定 SP 系统，能避免长请求被小 SP 拖慢、短请求被大 SP 浪费。在 Tetris（ISCA'26）论文中，ESP 是 SOTA baseline：Tetris 指出其三点不足——(1) 统一 TP 无法满足 prefill（偏好小 TP 灵活分配）与 decoding（偏好大 TP 压低延迟）的异构需求；(2) 贪心静态 batching 为整批请求赋最大 SP、缺乏全局负载感知，过度 SP 扩张恶化全局 TTFT 分布；(3) 请求级 SP 粒度 + ring 同步造成实例空转（资源碎片）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# LoongServe ESP 运转流程
请求批到达 → 调度器 DP 选实例集、为批分配 SP 大小（动态规划最小化 per-batch prefill 延迟）
         → 批内请求 prefill 计算（SP 实例组 ring attention）
         → 整批 prefill 完成 → 集体进入 decoding（静态 batching，批固定）
# Tetris 对其的对比例子（16 实例、TP=1、各 1s 排队延迟）：
#   贪心给 32k 请求 SP=16 → 后到 16k 请求 TTFT=(1.53s,1.84s)
#   若给 32k 请求 SP=8、留 8 实例给 16k → TTFT=(1.58s,1.31s)（全局平均/max 更优）
```
Annotations: ESP 的弹性=请求粒度 SP 变化（同一请求内不变）；调整 SP 只改 token 分配、不改权重；prefill 与 decoding 统一 TP 是其架构约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LoongServe 开源（https://github.com/LoongServe/LoongServe，含 artifact 复现），基于 vLLM 扩展；四步调度（dispatch、elastic instance allocation、batching、elastic scaling plan，多项式复杂度迭代粒度决策）；proactive scale-down（prefill 阶段复用通信减少迁移）+ multi-master 分布式 decoding（decode 扩容不迁 KV）。效果：最高 3.85×（vs chunked prefill）、5.81×（vs prefill-decoding 分离）吞吐。Tetris 以其为 baseline（TP=1/TP=4 配置）并在其基础上提出 CDSP（chunk 级粒度 + 解耦集群异构 TP + 负载感知 SP 扩张控制）。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
