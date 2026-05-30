## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - JANUS 在 SGLang 上实现了面向 MoE 的解耦式推理系统，核心 Serving 调度实现包括三个层次：
    1. **Attention-MoE 解耦架构（Section 3.2-3.3）**：将 Attention 层和 MoE 层部署到独立的 GPU 子集群（attention nodes / MoE nodes），支持各层类型独立配置并行度。通过 Adaptive Two-Phase Communication 实现低延迟跨子集群数据交换——Phase 1: 同节点多个实例通过 NVLink 聚合中间激活；Phase 2: 聚合后的大块数据通过 GPUDirect RDMA 跨节点传输。根据资源配置和流量负载自适应选择两种传输模式：Case-1（直接点对点传输）或 Case-2（一对一中继 + 节点内 NVLink 多播）。Gating 放置在 MoE 侧以简化通信、避免 per-expert tensor packing。
    2. **Activated-Expert-Balanced Scheduling (AEBS, Section 3.4)**：轻量级 GPU kernel 在每 MoE 层执行激活专家调度——收集当前 batch 的 top-k 路由结果 → 将单副本 expert 固定分配到唯一持有实例 → 多副本 expert 贪心分配到当前负载最低的实例 → 重写路由结果为物理副本 ID → dispatch token。**无 CPU-GPU 同步、无跨 GPU 协调**，调度开销在微秒级。
    3. **Fine-Grained SLO-Aware Resource Scaling（Section 3.5）**：基于 Roofline 和 Little's Law 构建 TPOT 性能模型（Eq. 1-3），使用 Monte Carlo 估计 a_max（最大激活 expert 数），通过 bounded binary search 求解稳态 batch size，枚举 (n_a, n_e) 搜索空间选择满足 SLO 的最小 GPU 配置。同时优化 expert placement 以避免高频共激活 expert 被放置在同一实例（min-max co-activation load，Appendix B Algorithm 3）。
  - 实现量：~4K 行 Python + ~300 行 CUDA/C++，基于 SGLang。
  - 实验比较：
    - Baselines：SGLang（monolithic TP/EP）、MegaScale-Infer（解耦式，随机 expert 调度 + attention 侧 gating + 粗粒度 scaling）、xDeepServe（解耦式，EPLB 调度 + attention 侧 gating + 4 GPU 粒度 scaling）
    - 指标：TPOT（per-token SLO）、per-GPU Throughput（TPG）
    - 数据集：ShareGPT（avg 16 in + 256 out）、BurstGPT（合成动态到达）
    - 结果：JANUS 相比 SGLang/MegaScale-Infer/xDeepServe 分别提升 per-GPU throughput 最高 4.7×/2.2×/3.3×，满足 TPOT SLO

- 硬件平台是什么，配置是什么。
  - 最多 4 节点 GPU 集群：
    - 每节点：128 CPU cores，2TB host memory，8× NVIDIA H100 (80GB)
    - GPU 间互联：900 GB/s NVLink (intra-node)
    - 跨节点互联：400 Gbps InfiniBand NIC per GPU
  - 软件环境：SGLang + NVSHMEM + NCCL + GPUDirect RDMA

- 开源Serving框架是什么。修改了什么。
  - **Serving 框架**：SGLang (https://github.com/sgl-project/sglang)。
  - 核心修改：
    1. **解耦架构改造**：将 SGLang 的 monolithic 部署拆分为 attention instances 和 MoE instances 两组，每个 instance 运行于一块 GPU。Attention 侧复用 SGLang 的 request batching/dispatching/KV-cache 管理。MoE 侧每个 instance 持有 expert 子集。
    2. **Adaptive Two-Phase Communication**：使用 NVSHMEM 的 one-sided putmem_signal/signal_wait 原语实现跨子集群数据交换。Intra-node collectives 使用 NCCL。将 layer index + token count 等元数据打包进 signal value 避免单独传输。共享 expert 放置在 attention 侧，与跨子集群通信重叠执行。
    3. **AEBS Scheduler**：GPU kernel 实现，每 MoE 层在每个 MoE instance 上独立执行（synchronization-free），通过冗余计算避免跨 GPU 协调。
    4. **Scaling Controller**：MoE controller + attention controller 周期性（15 min 间隔）收集 activation statistics，运行 Algorithm 2 搜索最优 (n_a, n_e)，增量调整实例数量。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未明确给出 JANUS 开源链接，基于 SGLang 实现。
  - **JANUS Serving 框架执行全过程（DeepSeek-V2 decode 阶段，1A6E 配置为例）**：

    ```
    ① 输入：用户请求到达 → Attention Controller 将请求分发到 n_a=1 个 attention instance (GPU A)。
    
    ② Attention Instance (GPU A)：
       - SGLang continuous batching 管理 in-flight decode batch B。
       - 执行 Attention 层 (MLA)：KV cache 常驻 GPU A HBM → Q/K/V 计算。
       - 执行 Shared Expert（overlap 阶段）：在等待 MoE 结果时计算共享 expert FFN。
    
    ③ Cross-Sub-Cluster Communication (每 MoE 层)：
       注意: Attention instance 在 GPU A, MoE instances 在 GPU E0–E5 (共 n_e=6)
       - Phase 1 (Intra-node 聚合): 若同一 attention node 有多个 attention instances，
         通过 NVLink NCCL 聚合中间激活 → 减少跨节点传输次数。
       - Phase 2 (Inter-node 传输): GPU A 通过 NVSHMEM putmem_signal (GPUDirect RDMA)
         将激活直接写入 MoE instance GPU 的接收 buffer → signal 通知完成。
         元数据 (layer index, token count) 打包在 signal value 中。
       - 根据配置选择 Case-1 (直接点到点) 或 Case-2 (一比一中继+多播)。
    
    ④ MoE Gating (MoE 侧 GPU E0–E5)：
       - 接收完整 activation → Router softmax(W_gate·h) → Top-K routing。
       - 每 GPU 独立运行 AEBS GPU kernel:
         → 收集当前 batch 所有 token 的激活逻辑 expert IDs
         → 单副本 expert 固定分配；多副本 expert 贪心选负载最低实例
         → 重写每个 token 路由为物理 replica IDs
         → Dispatch token 激活到持有对应 replica 的 GPU
       - AEBS 开销: <90μs (batch=4096, 16 MoE instances)
    
    ⑤ Expert FFN (MoE 侧 GPU E0–E5)：
       - 各 GPU 对收到的 tokens 执行本地 expert FFN (GEMM via cuBLAS)。
       - MoE 层延迟由 max(a_max) 决定——即激活 expert 数最多的 GPU 决定。
    
    ⑥ Combine (反向两阶段通信)：
       - Phase 1: MoE 侧 intra-node all-reduce 聚合中间结果。
       - Phase 2: NVSHMEM putmem_signal 将结果传回 attention GPU。
    
    ⑦ 输出：Attention GPU 完成 Shared Expert + MoE output 的 residual add → next token。
    
    ⑧ Scaling Loop (15 min 间隔)：
       - MoE Controller 收集各层 activation statistics。
       - 用 recent trace 构建 Monte Carlo â_max 查找表。
       - Algorithm 2: 枚举 (n_a, n_e) 搜索空间 → 求解 Eq. (2) 稳态 B* → 检查 TPOT SLO + memory feasibility → 选择 min(n_a+n_e)。
       - 增量调整实例数: 添加/移除 attention 和 MoE instances → 重新运行 expert placement (Algorithm 3, 最小化 co-activation 共现)。
    ```

    关键性能数据（DeepSeek-V2, H100）：
    | SLO | JANUS TPG (tok/s/GPU) | vs SGLang | vs MegaScale | vs xDeepServe |
    |-----|----------------------|-----------|-------------|---------------|
    | 200ms | 最高 | up to 4.7× | up to 2.2× | up to 3.3× |
    | 150ms | 最高 | up to 4.7× | up to 2.2× | up to 3.3× |
    | Dynamic trace (24h) | — | 39% GPU-h节省 | 16% GPU-h节省 | — |
