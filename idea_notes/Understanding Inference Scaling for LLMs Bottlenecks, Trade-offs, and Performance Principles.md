## Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles

- baseline方法是什么？
  - baseline 是当前主流 LLM Serving 的"标准缩放启发式"：①盲目最大化并发/批大小（max_num_seqs、max_num_batched_tokens 拉满）以摊销 kernel 启动开销并提高 GPU 占用率；②把 Data Parallelism 当作默认扩展策略（避免跨卡通信，直接复制模型权重、按请求分流）；③对任意模型套用固定/单一并行策略（如统一 DP 或统一 TP），不按模型架构与序列长度区分；④把推理当作单一 workload，只关注峰值 FLOPs 与整体吞吐，不区分 prefill 与 decode 的硬件需求。在标准 chat 负载（OSL≈500、decode 短）下这些启发式工作良好，但在 reasoning（CoT）负载下失效。
  - baseline 全栈执行例子（DeepSeek-8B 在 8×H200、DP=8、max_num_seqs=10K 下跑 Natural Reasoning 推理负载）：
    ```
    算法pipeline层：8B 密集模型 + GQA（8 KV head，KV 足迹仍随层数线性增长），模型权重复制 8 份到 8 卡
    系统框架层：vLLM FCFS 调度器把 10K 请求全量准入 → PagedAttention 逐请求分配 KV 块 → 无容量感知的并发控制
    编译框架层：论文未明确说明（baseline 用 vLLM 内置 kernel，未涉及编译框架修改）
    kernel调度层：论文明确抽象掉 kernel 级优化（未明确说明具体 kernel 实现）
    硬件架构层：8×H200（141GB HBM3e/卡）各自独立跑一个模型副本；每个请求 10k reasoning token 的 KV 持续累积，
               聚合 KV 占用数分钟内冲到 100% → vLLM 调度器 preempt（Running→Waiting 队列）→ prefix cache 命中失败
               → 全量 prefill 重算；HBM 带宽利用 40%–85% 剧烈振荡（sawtooth）；TPOT 从 0.08s 涨到 0.48s，
               E2E 尾部延迟失控，吞吐增益崩塌（Capacity Trap）
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法是系统表征驱动的"并行策略决策框架"：把推理负载分类为 Capacity-Bound（KV 容量）而非单纯 compute/bandwidth-bound，用多指标遥测（TTFT、TPOT、生成吞吐、E2E、KV 占用、HBM 带宽、请求状态机）定位瓶颈，然后按模型规模与架构给出最优并行策略：①KV-aware 并发上限——不用 max_num_seqs 盲目拉满，而是按 HBM headroom、活跃序列长度、预期 decode KV 增长动态设置并发帽（Observation 1）；②小模型（≤14B）纯 DP，中模型（32B）DP=4+TP=2 的"Right-Sized TP"（只用最小 TP 释放容量、其余靠 DP 提并发，比纯 TP 快 ≈30%），避免 TP 通信开销（Observation 5）；③frontier 密集模型（Llama-405B）高维度 TP=8（TP 聚合 HBM 容量+带宽、保持激活 node-local），不用 PP（其 KV 限制 micro-batch 深度放大 pipeline bubble，PP=8 慢 7.6×）（Observation 6）；④frontier 稀疏 MoE 模型（DeepSeek-R1-671B）用 PP=4+TP=2 混合（MLA 压缩 KV 使 PP 可加深 micro-batch 填气泡，低 TP 度降低 all-reduce 同步开销，1663s vs 纯 TP=8 的 2047s）；⑤prefill/decode 资源发散 → 主张架构级 prefill/decode 解耦（prefill 用高 TFLOP 加速器、decode 用内存中心化层次 + CXL/HBF/3D 堆叠），以及 KV-aware admission control 与 decode throttling 把调度变成"内存流量整形"问题（Observation 7/9）。对应解决 baseline 缺陷：并发/DP 盲目缩放引发的抢占-重算惩罚 → 显式容量规划；单一并行策略 → 模型特定 profiling 选择 DP/TP/PP；把内存容量当一等设计参数而非只看 FLOPs/带宽。
  - 论文方法全栈执行例子（同一 DeepSeek-8B + 32B 与 R1-671B 推理负载的推荐配置）：
    ```
    算法pipeline层：8B/14B/32B 蒸馏密集模型用 GQA；R1-671B 用 MoE（激活≈37B）+ MLA 低秩 latent 压缩 KV
               （MLA 与 PP 协同：KV 足迹小→每 stage 可容纳更多 micro-batch→pipeline bubble 被填满）
    系统框架层：vLLM 调度器改为 KV-aware 并发帽（如 2K 而非 10K，E2E 最优甜点）；32B 用 DP=4+TP=2：
               TP=2 把权重分片释放 HBM 给 KV（64GB→32GB/卡），DP=4 维持集群并发；
               R1-671B 用 PP=4+TP=2：PP 分阶段减内存、TP=2 最小化 all-reduce 同步；
               对 5K batch 触发 chunked prefill（每迭代只处理部分 prompt token）防 OOM、限制 convoy 模式
    编译框架层：论文未明确说明（本文不涉及编译框架实现）
    kernel调度层：论文明确抽象掉 kernel 级优化（未明确说明具体 kernel 实现；讨论中主张 decode 侧
               KV locality/带宽调度优先于峰值计算利用率）
    硬件架构层：8×H200 上 TP=8 跑 Llama-405B（权重 800GB 分片到 8×141GB、激活 node-local、NVLink 900GB/s
               all-reduce）；R1-671B 用 PP 跨 8 卡分 4 stage + 每 stage TP=2；decode 阶段 HBM 带宽饱和 ≈65–85%；
               前瞻：prefill 卸载到高 TFLOP 加速器 + decode 用 HBM→DDR/LPDDR→CXL→NVMe 分层 + 3D 堆叠带宽层，
               调度器按"内存流量整形"预留 decode KV 容量
    ```
