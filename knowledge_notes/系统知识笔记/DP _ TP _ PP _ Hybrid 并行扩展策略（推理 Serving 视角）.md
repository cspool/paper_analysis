## DP / TP / PP / Hybrid 并行扩展策略（推理 Serving 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 推理把模型与请求分布到多 GPU 的四种并行策略：Data Parallelism（DP，每卡一份完整权重、按请求分流，无跨卡通信）、Tensor Parallelism（TP，单层权重按 head/维度切分到多卡、每层 all-reduce 同步，聚合集体 HBM）、Pipeline Parallelism（PP，层按顺序切分到多卡、stage 间传激活、引入 pipeline bubble）、Hybrid（分层组合，如节点内 TP + 节点间 PP/DP）。本论文（系统表征）给出推理侧（非训练侧）的核心洞见：并行策略的本质是"内存容量 × 通信开销 × 并发"的权衡——TP 释放 KV 容量但引入同步，PP 减内存但放大 bubble，DP 简单但每卡独立撞容量墙。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# 8×H200 上同一推理负载（batch 2K，DeepSeek 系模型）的策略对比（本论文实测）
# 14B（密集，权重小）：DP=8 最优（332s）；PP=2+TP=4 慢 3.5×（1172s）——权重单卡放得下，TP 通信不划算
# 32B（密集，权重 64GB/卡）：DP=4+TP=2 最优（484s）；纯 TP=8（686s）> 纯 DP=8（857s）
#    → TP=2 只分片一半权重释放 HBM 给 KV（防抢占），DP=4 维持并发——"Right-Sized TP"
# Llama-405B（密集 frontier）：仅 TP=8 或 PP=8 可行；TP=8=986s，PP=8=7537s（7.6× 慢）
#    → 405B 全参数激活 + 1.05MB/token KV 使 PP 每 stage 装不下足够 micro-batch → bubble 无法隐藏
# DeepSeek-R1-671B（MoE frontier）：PP=4+TP=2=1663s < 纯 TP=8=2047s
#    → 激活仅 37B，计算-通信比低，TP=8 的 all-reduce 同步成瓶颈；MLA 压缩 KV 让 PP 深 micro-batch 填气泡
```
Annotations：DP/TP/PP 的选择由"容量诱导的抢占成本 vs TP 通信成本"决定（Observation 5）；frontier 密集 vs 稀疏模型的偏好相反（Observation 6）；决策应在 admission/profile 时按模型具体 profiling，而非固定策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Serving 引擎（vLLM/SGLang）暴露 --tensor-parallel-size / --pipeline-parallel-size 等配置，DP 由多实例/多副本实现；TP 在 NVLink 域内（本论文 8×H200 NVLink 900GB/s）做逐层 all-reduce，PP 跨 stage 传激活（PCIe/节点间链路带宽更低，405B 场景 PP 激活流量饱和互联）。使用要点（本论文指南）：小模型（≤14B）纯 DP；中模型（32B 级）最小 TP 释放容量 + DP 提并发；密集 frontier 高维度 TP（聚合容量+带宽、激活 node-local）；稀疏 MoE frontier 高 PP + 低 TP（降同步、填气泡）。注：训练侧 3D 并行（Megatron）与推理侧策略语义相同但优化目标不同（训练重吞吐、推理重容量/SLO），本条目以推理 Serving 视角为准；kernel/运行时映射视角见本库 kernel 调度层 TP/PP 条目。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
