## Continuous Lookahead Pipelining（连续前瞻流水线）

术语是什么？
Continuous Lookahead Pipelining 是 PROBE 提出的核心调度范式：将 MoE 推理中的负载均衡从"被动调整"（reactive adjustment）转变为"主动准备"（proactive preparation）。其核心思想是——虽然 token 到达是随机的，但深度模型的语义 routing 具有可预测性——PROBE 利用当前 MoE 层的执行时间，为下一层异步执行 Predict（预测 expert 分布）、Plan（优化 expert 复制决策）、Prefetch（传输 expert 权重），三个控制阶段均完全隐藏在主流关键路径之后。

从系统架构角度拆解术语：
双轨（dual-track）执行模型是 Continuous Lookahead Pipelining 的架构基础：
```
Main Stream (Deterministic Track)          Auxiliary Track (Control Plane)
├─ Layer L-1 Attention (Compute)           │
├─ Layer L-1 All-to-All Dispatch (Net)     ├─ Predict for Layer L (MLP+AllGather)
│                                          │   ↓
├─ Layer L-1 MoE Compute (Compute)         ├─ Plan for Layer L (Single-SM CUDA)
│                                          │   ↓
├─ Layer L-1 All-to-All Combine (Net)      │   (Prefetch paused to yield bandwidth)
│                                          │
├─ Layer L Attention (Compute)             ├─ Prefetch resume (P2P expert xfer)
│  ...                                     │  ...
```
关键约束：Predict 利用 All-to-All Dispatch 的网络带宽闲置期；Plan 利用 MoE Compute 的 SM 闲置（仅占 1 个 SM）；Prefetch 通过 split-phase transmission 避免与 All-to-All Combine 竞争带宽。

术语一般如何实现？如何使用？
在 PROBE 中通过 SGLang + DeepEP + NVSHMEM 实现。Predictor 在 dispatch 启动时并行执行 MLP inference + 全局 All-Gather；Planner 为单 SM CUDA kernel（kmax=16 iterations）；Prefetch 为自定义 Triton kernel 的 P2P put 操作，在 MoE compute 和下一层 attention 期间分阶段传输。适用于任何需要零开销在线重新配置的 MoE 推理系统。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
