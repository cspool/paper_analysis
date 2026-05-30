## MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

- baseline方法是什么？
  Baseline 为 vLLM（tensor parallelism only）和 TensorRT-LLM（tensor parallelism + expert parallelism for expert layers），二者均为 attention 和 FFN 模块**共置部署**（colocated）。以 vLLM 为例说明全栈执行路径：
  - **算法层**：MoE 模型推理时，每层依次执行 attention → gating → expert FFN。Decoding 阶段 attention 为 memory-intensive（需访问每个请求的 KV cache），FFN 为 compute-intensive（weights 可跨 batch 共享）。但在 MoE 中，因 sparse activation（每个 token 仅激活 top-K expert），给定 batch size B 时每个 expert 仅处理 B × top-K / #experts 个 token。以 Mixtral 8×22B 为例，B=156 时每个 expert 仅得 ~39 tokens，理论 MFU 仅 25%（= top-K / #experts = 2/8）。
  - **系统框架层**：vLLM 使用 PagedAttention + continuous batching 管理 KV cache，通过 tensor parallelism 跨多个 GPU 切分权重矩阵。整个模型（含 attention 和 FFN）部署在同一组 GPU 上。Mixtral 8×22B / DBRX 需要最小 8 GPU 单节点，Scaled-MoE 需多节点（pipeline parallelism）。TensorRT-LLM 额外支持 expert parallelism（expert 分布到不同 GPU），但 attention 和 FFN 仍共置。
  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。
  - **kernel 调度层**：vLLM 使用 FlashAttention + custom CUDA kernels，TensorRT-LLM 使用 custom kernel optimizations。NCCL All-to-All 用于 MoE 层的 token dispatch/collect（expert parallelism 模式）。无 communication-computation overlap 设计。
  - **硬件架构层**：NVIDIA 80GB Ampere GPU（A800），200 Gbps InfiniBand inter-node，400 GB/s NVLink intra-node。
  Baseline 核心缺陷：
  1. **FFN GPU 利用率低**：MoE 的 sparsity（top-K / #experts）直接降低每个 expert 的有效 batch size，使 FFN 从 compute-intensive 退化为 memory-intensive，MFU 仅 25%（Mixtral 8×22B）或更低。
  2. **无法独立扩展**：Attention 和 FFN 共置部署，无法分别根据其 memory-intensive vs compute-intensive 特性独立优化并行策略和硬件选择。
  3. **通信开销无法隐藏**：All-to-All 通信与计算串行执行，GPU 在通信期间空闲。
  4. **同构部署浪费成本**：Attention（memory-intensive）和 FFN（compute-intensive）被迫使用相同 GPU 类型，无法利用 heterogeneous hardware 的性价比优势。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MegaScale-Infer 通过 **Disaggregated Expert Parallelism + Ping-Pong Pipeline Parallelism + High-Performance M2N Communication** 三层次设计解决上述缺陷。全栈执行路径（以 Mixtral 8×22B, tp_a=2, tp_e=1, n_a=4, E=8, m=3 为例）：
  - **算法层 — Disaggregated Expert Parallelism + Ping-Pong Pipeline**：
    1. **Module Disaggregation**：将每个 Transformer layer 的 attention 模块和 expert FFN 模块物理分离到不同 GPU 节点。Attention node 复制全部 attention 参数（data parallelism），expert node 按 expert parallelism 分布 experts（每 expert node 1-8 GPU，tensor parallelism within node）。
    2. **Independent Scaling**：n_a 个 attention node 聚合请求到 E 个 expert node，每个 expert 的有效 batch size = b_a × n_a × K/E。通过调节 n_a 可使 expert 从 memory-intensive 变为 compute-intensive（满足 b_e ≥ F/B，即 batch ≥ compute/memory bandwidth ratio）。
    3. **Ping-Pong Pipeline**：将 global batch B 拆分为 m 个 micro-batch。在 Layer ℓ：micro-batch 0 在 attention → expert → attention（Layer ℓ+1），micro-batch 1 在 attention → expert，micro-batch 2 在 attention。3 个 micro-batch 交替流动，使 attention 和 expert 在对方计算时持续 busy，且通信被计算覆盖（T_c < T_f 时）。
    4. **Deployment Plan Search（Algorithm 1）**：枚举 tp_a, tp_e, n_a, m，SIMULATE 函数通过性能模型（ROI-based GEMM timing + profiling 获取的 linear model coefficients + network bandwidth utilization profiling）binary search 最大 B 满足 SLO（T_iter ≤ 150ms TBT），选最大化 throughput per unit cost 的 plan。
    5. **Heterogeneous Hardware Selection**：attention node 选高 per-cost 内存带宽/容量 GPU（如 H20: 51.9 GB/$, 2214.1 GB/s/$），expert node 选高 per-cost 计算 GPU（如 L40S: 335.2 TFLOPS/$）。
  - **系统框架层 — M2N Communication + Fused Kernels**：
    1. M2N 通信库（PyTorch extension, ~4900 行 C/C++ + ~5000 行 Python）：替代 NCCL 的 peer-to-peer primitives，使用 GPUDirect + RDMA write with immediate 消除 GPU-to-CPU 拷贝，使用 CUDA event + cuStreamWaitValue32 消除 GPU synchronization，使用 GDRCopy flush 确保数据一致性。
    2. Flux-based kernel fusion：将 TP 的 all-gather 与 GEMM 融合为单 kernel。
    3. Sequential operator fusion：gating + top-k + token scatter 融合，减少 kernel launch 和 memory access。
    4. Expert load balancing：on-device redundancy based on expert popularity，greedy approximation 分配 experts 到 nodes。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层 — M2N Communication Optimization**：
    1. M2N Sender chain：cudaEventSynchronize → cuStreamWaitValue32(block stream) → RDMA write with immediate → poll CQ → shared memory flag wake stream。
    2. M2N Receiver chain：cudaEventSynchronize → cuStreamWaitValue32(block stream) → poll CQ → GDRCopy flush → shared memory flag wake stream。
    3. Traffic optimizations：高优先级 ACK queue 隔离 ACK 与 data 包；congestion control 微调减少 rate-limiting。
    4. 对比 NCCL：消除 batch-of-8 group op 限制、GPU sync instability、group init overhead；对比 DeepEP CPU vs GPU 通信 trade-off（本场景 ~256KB/pair 下 CPU single-thread 足以饱和带宽）。
  - **硬件架构层**：
    同构：8×NVIDIA 80GB Ampere GPU per node, 200 Gbps InfiniBand, 400 GB/s NVLink。
    异构：H20（96 GB, 4096 GB/s, 148 TFLOPS）+ L40S（48 GB, 864 GB/s, 362 TFLOPS）。无硬件修改。
  对比 baseline 的改进映射：
  - **FFN GPU 利用率低 → Disaggregated Expert Parallelism 增大有效 batch size**：Colocated baseline 下每个 expert 仅得 B×K/E tokens → disaggregated 下每个 expert 得 B×n_a×K/E tokens（n_a 个 attention node 的请求聚合）。以 Mixtral 8×22B 为例，n_a=4 时 expert batch size 增至 4×，FFN 从 25% MFU 提升至 compute-bound。端到端 decoding throughput per GPU 提升 2.56× vs vLLM，1.28× vs TensorRT-LLM。
  - **无法独立扩展 → Attention Replication + Expert Parallelism 独立配置**：Attention 使用 data parallelism（按需 n_a replica）、expert 使用 expert parallelism（按需 E expert nodes），各自独立选择 tp size 和 GPU 类型。Deployment plan search 自动找到平衡 T_a ≈ T_e 的配置，最大化 GPU 利用率。
  - **通信开销无法隐藏 → Ping-Pong Pipeline Parallelism**：m=1 时 attention/expert 互相等待（idle time）→ m=2 时双方同时 busy → m=3 时通信被计算完全覆盖（T_c < T_f 条件满足时），throughput 额外提升 1.10×–1.38×（模型越大、通信开销越大，m 增加收益越显著）。
  - **同构部署浪费成本 → Heterogeneous Deployment**：H20（高 per-$ bandwidth）为 attention + L40S（高 per-$ compute）为 expert。Decoding throughput per unit cost 提升 3.24× vs vLLM on H20，1.86× vs TensorRT-LLM on H20。同时吞吐 per unit power 提升 1.80×（H20 500W vs L40S 350W 的功耗-性能效率差异化利用）。
