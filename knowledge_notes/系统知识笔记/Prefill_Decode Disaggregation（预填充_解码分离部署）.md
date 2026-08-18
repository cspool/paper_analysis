## Prefill/Decode Disaggregation（预填充/解码分离部署）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill/Decode Disaggregation（PD 分离）是把 LLM 推理的两个阶段部署到不同硬件资源上的架构：prefill（处理 prompt、compute-bound、几何级并行）放新/强 GPU，decode（逐 token 自回归生成、memory-bound、算术强度低）放旧/弱 GPU，两实例之间通过高速互连（NVLink/RDMA）传输 KV Cache。本论文（Rearchitecting the Datacenter Lifecycle for AI）不实现新分离机制，而是把 PD 分离作为 operation 阶段的软件优化之一（表 VIII：Disaggregation [88],[94],[110],[125],[127]，TCO 影响显著）纳入生命周期 TCO 评估：prefill/decode 相位分离后可把不同相位路由到最匹配的硬件代际——compute-bound prefill 用新 GPU、memory-bound decode 用旧 GPU，从而延长旧硬件的有用寿命、推迟刷新采购，实现跨 stage 协同（operation → IT provisioning：异构代际被更高效复用）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在本论文 TCO 框架中的运转流程：
```
请求到达 → operation 层路由：新请求 prefill 分到新代 H100/H200 池（compute 密集）
         → decode 阶段经高速互连把 KV Cache 传给旧代 A100/V100 池（memory 密集）
         → IT provisioning 层：因旧 GPU 仍承担 decode 高价值工作，
           刷新决策改为"延长旧代寿命、跳过 B100/B200"
         → build 层：为支撑该分离拓扑，网络选 hierarchical
           （NVLink intra-server + InfiniBand intra-rack + Ethernet inter-rack）
         → 蒙特卡洛 TCO 验证该组合优于全量换新
```
该用法与专门 serving 论文（Splitwise/DistServe 等）不同：后者关注单集群内分离的调度/传输机制，本文把 PD 分离抽象为"改变硬件代际负载画像"的生命周期杠杆——分离让不同代 GPU 各得其所，使"刷新"从全量换代变成按相位按需补充，这是 TCO 层面 40% 降幅的重要来源之一。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：Splitwise（静态分离，独立 prefill/decode 实例）、DistServe（P/D 分离 + 各自最优并行度）、LoongServe（动态）、vLLM/SGLang 的 PD 分离部署（KV cache transfer 经 NCCL/RDMA）；本论文在模拟层面通过 roofline+SLO goodput 对分离前后各代 GPU 的负载与 TCO 建模，不提供部署代码。论文开源框架 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）支持"disaggregated serving"策略（README 列出的 5 种 lifecycle policy 之一），可直接跑 `dc-tco run --policy disaggregated-serving` 对比分离 vs 非分离的 TCO。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
EPD-Serve 补充视角（encoder-prefill-decode 三分池，MLLM 场景）：EPD Disaggregation（EPD-Serve，ICML'25）把 MLLM 推理拆为 Encode/Prefill/Decode 三个独立可调度实例、专用 GPU 池 + 池内静态并行度，跨阶段经异步 tensor 传输（E-P 特征传输、P-D KV cache 传输，用 MM Store/Mooncake 等）。RESONATOR 以 EPD-Serve 为 MLLM 专用 baseline 对比：EPD-Serve 用 6×A100（2 encoder+4 LLM）静态分池、池边界固定导致 GPU 利用不足（小/纯文本请求浪费 encoder 池），RESONATOR 用 4×A100（省 33% GPU）动态调整 encoder TP/DP + intra-GPU 隔离保护 decode，在 Kimi-VL-16B/Qwen2-VL-7B 上全面领先（TTFT 最高 2.31×、E2E 1.81×、TPOT 1.99×，负载越高差距越大）——结论：消除静态池边界的 co-scheduling 比阶段分池更优。

Tetris 补充视角（ISCA'26，PD 分离 + 动态 SP 的异构并行组合）：Tetris 的集群即 prefill-decoding 解耦架构，但相对纯 TP/PP 的 PD 分离（Splitwise/DistServe 等）新增了"动态 SP"维度：prefill 侧把所有实例连成统一 SP 池（TP=1 for 8B）以支持 CDSP 的 chunk 级 SP 变化（调整 SP 只需重分 token、不需重分片权重）；decoding 侧用大 TP（TP=8）的 DP 实例独立部署。PD 比例为 1:1（权衡 TTFT/TBT）。跨阶段数据路径：prefill 各实例把 chunk KV cache 经 handshake 式 backend 分配流式传送到目标 decoding 实例，receive 侧收齐后以 continuous batching 插入 decode。与 LoongServe（统一 TP、不分离）对比，PD 分离让 decode 用上大 TP（TTFT 评估时 LoongServe Disaggregated 即此架构的 baseline）；论文以 LoongServe Disaggregated 与 Fixed-SP Scheduling 作为解耦架构对照，Tetris 在相同 P/D 比例与 TP 配置下靠 CDSP 调度获胜（容量 +20%-45%、TTFT 最高 4.35× 更低）。
涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
