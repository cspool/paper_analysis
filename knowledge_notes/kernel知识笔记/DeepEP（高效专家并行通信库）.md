## DeepEP（高效专家并行通信库）

术语是什么？
DeepEP 是 DeepSeek 开源的专用 MoE 通信库（https://github.com/deepseek-ai/DeepEP），为 Expert Parallelism 的 All-to-All 通信提供优化实现。支持 normal mode（低延迟 kernel，适用于推理和微调）与 high-throughput mode（高带宽 kernel，适用于训练）。核心优化包括：token deduplication（同一 rank→同一 expert 的 token 合并发送）、topology-aware routing（感知 NVLink/NVSwitch 拓扑的通信路径选择）、SM-efficient dispatch kernel（减少 SM 占用，留出计算资源）。

从kernel调度角度拆解术语：
DeepEP 在 PROBE 中的使用：
```
// PROBE 以 DeepEP normal mode 作为 All-to-All 后端
// 在每层 MoE 执行中：

// Dispatch:
DeepEP.dispatch(
    tokens,           // per-rank local tokens
    expert_to_rank,   // static mapping (baseline placement)
    topk_indices,     // Router output
) → routed_tokens     // tokens grouped by target rank

// Combine:
DeepEP.combine(
    expert_outputs,   // per-expert FFN outputs
    token_indices,    // original token ordering
) → ordered_outputs   // outputs in original batch order
```
PROBE 选择 DeepEP normal mode 而非 high-throughput mode，因为：(1) normal mode 延迟更低，对 TPOT 更友好；(2) 低 SM 占用为 planner kernel 和 prefetch kernel 留出 SM 资源；(3) normal mode 更容易与 CUDA Graph 兼容。

术语一般如何实现？如何使用？
开源实现 (GitHub: deepseek-ai/DeepEP)，支持 NVLink/NVSwitch 和 RDMA (InfiniBand)。提供 Python API 和 CUDA kernel 级接口。在 PROBE 中作为 SGLang 的通信后端替代默认 NCCL All-to-All。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- UCCL-EP Portable Expert-Parallel Communication

UCCL-EP 将 DeepEP 扩展到异构硬件平台：通过 CPU-proxy-based 架构替代 IBGDA，使 DeepEP 的功能（token deduplication、hierarchical reduce、LL/HT mode）能在非 NVIDIA NIC（AWS EFA、Broadcom Thor-2）和非 NVIDIA GPU（AMD MI300X）上运行。UCCL-EP 保持 DeepEP API 兼容，作为 drop-in replacement 使用。在 NVIDIA-only 平台上 UCCL-EP 性能与 DeepEP 原版可比（HT mode dispatch latency < 5% 差异）。
