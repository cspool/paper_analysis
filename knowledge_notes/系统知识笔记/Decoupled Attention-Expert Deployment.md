## Decoupled Attention-Expert Deployment

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Decoupled Attention-Expert Deployment（解耦 Attention-Expert 部署）是一种 MoE 推理的部署范式，将 transformer 的 attention 模块和 expert/FFN 模块部署在**不同的 GPU worker 集合**上。Attention Workers (AWs) 处理 self-attention 计算并维护 per-request KV cache；Expert Workers (EWs) 处理 expert FFN 前向计算（stateless）。AW 和 EW 之间通过高性能网络（RDMA/NCCL all-to-all）交换 token embeddings。这一范式由 MegaScale-Infer（字节跳动/PKU, SIGCOMM 2025）率先在大规模生产环境中验证，核心动机是：attention 是 memory-bound 而 expert/FFN 是 compute-bound，分离部署可实现独立扩缩容——AWs 按 data parallelism 扩展以增加请求容量，EWs 按 expert parallelism 扩展以增加 expert 吞吐量。Tarragon 在此范式基础上增加了故障恢复能力。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 Mixtral-8×7B 在 Tarragon 中的执行为例：
1. **AW 侧**：用户请求到达 cluster gateway，round-robin 分发到某 AW。AW（基于 vLLM compute engine）对 layer ℓ 执行 attention 计算，更新 KV cache，产生 token embeddings。
2. **Gating + 分发**：Router G(t) 选 top-k expert。AW 通过 REFE（见该术语）将 token embeddings + metadata 通过 RDMA data-plane QP 发送到对应 EW。
3. **EW 侧**：EW 从多个 AW 收集同 layer ℓ、同 expert 的 tokens，聚合成 batch，执行 expert FFN（libtorch），将输出通过 RDMA 返回 AW。
4. **AW 聚合**：AW 用 gating weights 加权求和 expert 输出，进入 layer ℓ+1。
关键架构特征：(a) AW-EW 通信是 many-to-many 非对称模式，不同于标准 NCCL collective（如 all-reduce）；(b) 每层存在同步屏障——AW 必须等所有选中 expert 返回后才进入下一层；(c) AW scalable by data parallelism（每个 AW 服务不相交的请求子集），EW scalable by expert parallelism。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MegaScale-Infer 使用 Ping-Pong Pipeline：将 batch 分为 micro-batch 在 AW 和 EW 间交替传输以重叠计算和通信。实现了最高 1.90× per-GPU throughput vs vLLM/TensorRT-LLM。
- Tarragon 在 MegaScale-Infer 基础上增加：REFE（C++ 扩展 + Python shim）、双 QP RDMA 设计（control-plane + data-plane）、ERT 动态路由、自愈机制。
- 已在 vLLM 社区以 "Attention-FFN Disaggregation (AFD)" 名称讨论集成（GitHub Issue #22799）。
- 核心 trade-off：解耦带来灵活扩缩容和更高 GPU 利用率，但引入跨节点通信延迟（vs 单体部署中 NVLink 通信）。

MegaScale-Infer uses this paradigm with key innovations:
- **Ping-Pong Pipeline Parallelism**: Splits a request batch into m micro-batches that shuttle between attention nodes and expert nodes, keeping both sides busy and hiding communication overhead. Optimal number of micro-batches: m ≥ 2(1 + T_c/T_f), where T_f = max(T_a, T_e). With m=2, idle time is eliminated; with m≥3, communication is fully overlapped with computation.
- **Deployment Plan Search**: An optimization algorithm (Algorithm 1) that enumerates tp_a, tp_e, n_a, m combinations and uses a performance model (profiling-based linear models for GEMM time + network bandwidth utilization profiles) to binary search the maximum batch size satisfying SLO constraints. Objective: maximize throughput per unit cost.
- **Heterogeneous Deployment**: Attention nodes use GPUs with high per-cost memory bandwidth/capacity (e.g., H20: 51.9 GB/$, 2214.1 GB/s/$); expert nodes use GPUs with high per-cost compute (e.g., L40S: 335.2 TFLOPS/$). Achieves 1.86× per-cost throughput vs TensorRT-LLM.
- **M2N Communication Library**: A custom high-performance communication library (~4900 lines C/C++ + ~5000 lines Python) using GPUDirect + RDMA write with immediate + CUDA stream blocking (cuStreamWaitValue32) + GDRCopy flush, eliminating NCCL's GPU-to-CPU copies, group initialization overhead, and GPU synchronization instability. Achieves 4.2× higher throughput and 68.2% lower latency vs NCCL.
- **Production impact**: Deployed at ByteDance on ~10,000 GPUs, reducing serving costs by 1.5–2.0×.

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---
