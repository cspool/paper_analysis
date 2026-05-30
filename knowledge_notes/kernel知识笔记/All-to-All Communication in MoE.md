## All-to-All Communication in MoE

术语解释
All-to-All通信是MoE分布式推理中的核心集合通信操作，用于在Expert Parallelism中将token根据Router输出分发到持有对应expert的设备，以及将expert输出传回原始设备。MoEShard 采用不同策略：通过 token 全复制（all GPU send all tokens）替代 all-to-all scatter/gather，在 NVLink 高带宽下 overhead 可忽略。

术语是什么？
在MoE Expert Parallelism中，一个MoE层需要两次All-to-All操作：
1. **All-to-All Dispatch**：将每个设备上的token根据Router的expert选择分发到持有对应expert的设备
2. **All-to-All Combine**：将expert计算后的输出传回原始设备

执行时间由计算和通信两个阶段主导，通信通常是瓶颈。

MoEShard 的替代方案：不做 all-to-all scatter/gather token 路由，而是每 GPU 将所有 token 发送给所有其他 GPU（全复制），然后每 GPU 本地计算所有 expert 的 partial output，最后 gather 回源 GPU pointwise sum。此方案虽然通信量更大（每 GPU 发送全部 token），但利用 NVLink 高带宽（600 GiB/s）吸收，且换来 perfect load balancing。

从kernel调度角度拆解术语。
标准All-to-All vs 分层All-to-All的执行流程：

```
# 标准All-to-All（所有GPU直连）
for src_gpu in range(num_gpus):
    for dst_gpu in range(num_gpus):
        if src_gpu != dst_gpu:
            tokens_to_send = tokens_assigned[src_gpu][dst_gpu]
            NCCL_Send(tokens_to_send, dst_gpu)
            # 复杂度：O(N^2)，N为GPU数

# 分层All-to-All（Tutel/DeepSpeed-MoE方式）
# 阶段1：Intra-node（同节点内GPU）
for src_gpu in node_gpus:
    gather_tokens_from_all_gpus_in_node()  # 高带宽NVLink
    
# 阶段2：Inter-node（跨节点）
node_gateway_gpu.send_to_other_nodes()     # 低带宽网络
# 复杂度降低：先局部聚合再跨节点传输

# 脉动式All-to-All（Aurora方式）
# 有序传输token避免带宽竞争
sorted_gpus = sort_by_priority(token_urgency)
for gpu in sorted_gpus:
    schedule_transfer(gpu, time_slot=optimal_slot)
# 理论最小化通信时间
```

通信压缩策略：
- TA-MoE：根据网络拓扑自适应调整传输数据量
- DeepSpeed-TED：消除不必要信息传输

数据范式创新：
- Janus：移动expert而非token——"以数据为中心"的方法，当expert变化频率低于token时更高效
- ExFlow：通过确保token上下文一致性（所有GPU拥有所有请求的上下文而非仅自己的），将两次All-to-All减少为一次

术语一般如何实现？如何使用？
- 基于NCCL的alltoall集合操作
- 分层实现利用NVLink（intra-node高带宽）和IB/RoCE（inter-node低带宽）的带宽差异
- 与NCCL版本相关——不同版本的all-to-all实现性能差异显著
- 现代优化：NVLink one-sided alltoall（TensorRT-LLM）替代AllGather+ReduceScatter

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Accelerating Distributed MoE Training and Inference with Lina
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

**EPS-MoE 的 All-to-All 优化**：
EPS-MoE 将传统 TP+TP 模式中的 ncclAllReduce 分解为两部分以支持 Expert Pipeline：
- **Dispatch 阶段**：`ncclReduceScatter + all2all` 替代 `ncclAllReduce`，通信量从 V_{TP+TP}=2P/D(D-1) 降至 V_{TP+EP}=P/D(D-1)+V_{DP+EP}(P/D,D)
- **Combine 阶段**：`all2all + ncclAllGather` 替代 `ncclAllReduce`，同样减少通信量

这种分解使得 all2all 和 GEMM 可以在 Expert Pipeline Scheduler 中并行调度。通过限制 GEMM 占用的 SM 数（如 116 SM），为通信 kernel 留出 SM（16 SM），实现计算与通信在不同 SM 上的同时执行。

**Task-MoE 对 All-to-All 的消除（Kudugunta et al., EMNLP 2021）**：
Task-MoE 通过 task-level routing（decoder 侧）从根本上消除了 decoder 的 all-to-all 通信。因为同一 task 的所有 token 路由到相同的 experts（同 device），无需 token dispatch/combine 跨设备通信。Token-MoE 解码时 26.9% (WMT) ~ 36% (200 langs) step time 用于 all-to-all，Task-MoE 仅 0.0%-0.2%。这种消除仅在 decoder 侧生效（encoder 仍用 token-level routing），但因 decoder 每步时间是 encoder 的 200x，实际效果等同于全栈消除。

**Lina 的 All-to-All 优化视角**：
Lina 识别出 Training 和 Inference 中 All-to-All 瓶颈的不同根因：
- Training: All-to-All 与 Allreduce 在 backward pass 中重叠，公平共享网络带宽，All-to-All 被延长 median 1.83x（worst 4.14x）
- Inference: 真实请求下 expert popularity 高度倾斜（max/min ratio 4.02x~5.56x），导致各 link 的 All-to-All 传输量不均

Lina 的解决方案：
1. Training: Tensor partitioning → micro-ops → priority queue (All-to-All priority > Allreduce) → 确保 All-to-All 获满带宽
2. Training: All-to-All pipelining → 每个 All-to-All micro-op 完成后立即启动对应 FFN → 消除计算等待
3. Inference: Unequal split All-to-All → 按 device 实际 token 量动态拆分（非均匀）→ 匹配 popular expert link 高带宽需求

**LUFFY 的 All-to-All 优化视角**：
LUFFY 采用与上述方案正交的方法——不优化 all-to-all 通信本身的调度，而是减少需要传输的 token 数量：
1. Token Condensation: 识别并凝聚相似 token，减少 dispatch all-to-all 的 token 数量（如约 62% 的相似 token 可被凝聚）
2. Sequence Migration: 改变 combine all-to-all 的目标 GPU 路由，将跨 GPU token 拉取路径隐藏为 intra-GPU 路径
LUFFY 的实测通信加速：1.76×-3.72× vs Vanilla Expert Parallelism（取决于模型和 expert 数）

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

**ETR 论文中的 All-to-All 优化视角 (CoC + Locality Loss)**：
ETR 从两个维度优化 All-to-All 通信：(1) CoC (Communication over Computation)：利用 Ascend MTE (Memory Transfer Engine) 的远程内存访问能力，将 MatMul 和 All-to-All 通信融合为统一细粒度 kernel，实现计算与通信的流水线重叠——计算当前 batch 时预取下一 batch 的通信数据；(2) Locality Loss：通过 KL(D_c||D_l) 惩罚跨节点路由，将 token 优先分配至同节点 expert，直接减少跨节点 All-to-All 通信量。此外，自适应容量降低 C 减少了 token padding，间接缩小 All-to-All 传输量。实测在 32N/64N/256N Ascend NPU 集群上，idle time (含通信等待) 占比显著下降（见图5），训练效率提升 5.4%-46.6%。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

---
