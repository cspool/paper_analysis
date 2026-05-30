# 知识库_kernel调度

## Triton-based MoE Operator Implementation (基于 Triton 的 MoE 算子实现)

术语解释
BrownoutServe 使用 Triton 语言（而非 C++/CUDA）重写 MoE 模块的所有算子，包括 expert FFN 计算、token dispatch/combine、united expert 调用。Triton 在 Python 层面编写 tile-level 计算逻辑，编译为高效 GPU kernel，与 PyTorch 无缝兼容。

术语是什么？
Triton（Tillet et al. 2019）是一个 Python-based 的 GPU 编程语言和编译器。开发者用 Python 语法描述 tile-level 并行计算，Triton 编译器自动生成优化的 GPU kernel（PTX/CUDA），处理 shared memory 管理、thread block 调度、memory coalescing 等底层优化。相比手写 CUDA C++，Triton 开发效率更高且性能接近手写优化水平。

BrownoutServe 选择 Triton 的原因：(1) 与 PyTorch 无缝兼容，简化 BrownoutMoE 与 PyTorch 模型代码的集成；(2) 自动处理 GPU-specific 优化，降低开发维护成本；(3) MoE 的 token dispatch/combine 操作涉及大量 gather/scatter，Triton 的编程模型更友好。

从kernel调度角度拆解术语：
Triton-based BrownoutMoE kernel 执行流程：

```
// 1. Gate Kernel (Triton)
// grid: (num_tokens,), block: expert computation per token
@triton.jit
def gate_kernel(x_ptr, centroid_ptr, scores_ptr, topk_idx_ptr, topk_val_ptr,
                N_TOKENS, N_EXPERTS, HIDDEN_DIM, TOPK):
    pid = tl.program_id(0)  # token index
    if pid >= N_TOKENS: return
    
    # Load token hidden state
    x = tl.load(x_ptr + pid * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM))
    
    # Compute affinity with all experts (x^T · e_i)
    scores = tl.zeros([N_EXPERTS], dtype=tl.float32)
    for e in range(N_EXPERTS):
        centroid = tl.load(centroid_ptr + e * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM))
        scores[e] = tl.sum(x * centroid)  # dot product
    
    # Softmax + Top-K
    topk_vals, topk_idxs = tl.topk(tl.softmax(scores), TOPK)
    tl.store(topk_idx_ptr + pid * TOPK + tl.arange(0, TOPK), topk_idxs)
    tl.store(topk_val_ptr + pid * TOPK + tl.arange(0, TOPK), topk_vals)

// 2. Fused MoE FFN Kernel (Triton) - S1 tokens
// grid: (num_experts_in_S1,), block: per-expert batch computation
@triton.jit
def fused_moe_kernel(tokens_ptr, weights_ptr, outputs_ptr,
                     expert_offsets, N_EXPERTS_S1, HIDDEN, INTERMEDIATE):
    eid = tl.program_id(0)  # expert index
    if eid >= N_EXPERTS_S1: return
    
    # Load expert's token range
    start = tl.load(expert_offsets + eid)
    end = tl.load(expert_offsets + eid + 1)
    n_tokens = end - start
    
    # Gather tokens → 单个大 batch
    tokens = tl.load(tokens_ptr + start * HIDDEN + ...)  # [n_tokens, HIDDEN]
    
    # FFN: gate_proj → silu → up_proj → down_proj
    w_gate = tl.load(weights_ptr + eid * W_SIZE + ...)   # [HIDDEN, INTERMEDIATE]
    w_up = tl.load(weights_ptr + eid * W_SIZE + ...)     # [HIDDEN, INTERMEDIATE]
    w_down = tl.load(weights_ptr + eid * W_SIZE + ...)   # [INTERMEDIATE, HIDDEN]
    
    # GEMM tiles
    gate_out = tl.dot(tokens, w_gate)  # [n_tokens, INTERMEDIATE]
    up_out = tl.dot(tokens, w_up)      # [n_tokens, INTERMEDIATE]
    act_out = tl.silu(gate_out) * up_out
    out = tl.dot(act_out, w_down)       # [n_tokens, HIDDEN]
    
    # Scatter results back
    tl.store(outputs_ptr + start * HIDDEN + ..., out)

// 3. United Expert Kernel (Triton) - S2 tokens
// grid: (num_groups,), block: per-group concat + FFN
@triton.jit
def united_expert_kernel(tokens_ptr, ue_weights_ptr, outputs_ptr,
                         group_offsets, N_GROUPS, HIDDEN, INTERMEDIATE):
    gid = tl.program_id(0)  # group index
    if gid >= N_GROUPS: return
    
    # Concat all tokens from this group's experts
    start = tl.load(group_offsets + gid)
    end = tl.load(group_offsets + gid + 1)
    batch_size = end - start
    
    if batch_size <= 0: return  # empty group
    
    concat_tokens = tl.load(tokens_ptr + start * HIDDEN + ...)  # [batch_size, HIDDEN]
    
    # United Expert FFN (same structure as original expert)
    ue_w_gate = tl.load(ue_weights_ptr + gid * UE_W_SIZE + ...)
    ue_w_up = tl.load(ue_weights_ptr + gid * UE_W_SIZE + ...)
    ue_w_down = tl.load(ue_weights_ptr + gid * UE_W_SIZE + ...)
    
    gate_out = tl.dot(concat_tokens, ue_w_gate)
    up_out = tl.dot(concat_tokens, ue_w_up)
    act_out = tl.silu(gate_out) * up_out
    out = tl.dot(act_out, ue_w_down)
    
    tl.store(outputs_ptr + start * HIDDEN + ..., out)
```

术语一般如何实现？如何使用？
- 开发流程：用 `@triton.jit` 装饰器标注 kernel 函数 → 定义 grid（thread block 分布）→ Python 调用 → Triton JIT 编译为 GPU code
- 与 CUDA 的关系：Triton 生成 PTX 中间表示，经 PTX→SASS 编译在 GPU 上执行；性能通常达到手写 CUDA 的 80-95%
- BrownoutServe 使用 Triton 而非 CUDA 的原因：简化 brownout routing（大量条件分支）的实现，与 PyTorch 无缝集成

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

## GPU-side Block Table for PagedAttention (GPU 端 PagedAttention Block Table)

术语解释
BrownoutServe 对 vLLM 的 PagedAttention 实现进行了优化：将 KV cache 的 block table 从 CPU 移至 GPU 显存，block table 的查询/映射/更新操作实现为 GPU kernel，消除 CPU→GPU 的数据传输延迟。

术语是什么？
PagedAttention（Kwon et al. 2023, vLLM）将 KV cache 管理类比操作系统分页：逻辑 KV cache 被切分为固定大小的 "block"（如 16 tokens/block），通过 block table 将逻辑 block index 映射到物理 GPU memory 中的实际存储位置。原有实现中，block table 存储在 CPU 端，每次 attention kernel 执行前需将 block table 从 CPU 传输到 GPU。

BrownoutServe 的优化：将 block table 直接分配在 GPU 显存中（torch tensor on device），block table 操作（logical→physical 映射查询、新 block 分配、block 驱逐）作为 GPU kernel 执行，消除 CPU-GPU 同步点。

从kernel调度角度拆解术语：
```
// vLLM original: CPU-side block table
// 每 iteration:
//   CPU: block_table[req_id] = [physical_block_0, physical_block_1, ...]
//   CPU → GPU: memcpy(block_table)                    ← 额外数据传输
//   GPU: launch attention_kernel(block_table, Q, K, V)
//   GPU: attention = flash_attn(Q, K_from_blocks, V_from_blocks)

// BrownoutServe optimization: GPU-side block table
// 初始化: block_table = torch.zeros(..., device='cuda')  ← 分配在 GPU
// 每 iteration:
//   GPU: launch block_table_update_kernel(block_table)    ← 无需 CPU 参与
//   GPU: launch attention_kernel(block_table, Q, K, V)   ← 同一 GPU memory
//   GPU: flash_attn 直接读取 GPU block_table

// GPU-side block table lookup kernel (简化)
@triton.jit
def block_table_lookup_kernel(q_ptr, k_cache_ptr, v_cache_ptr,
                               block_table_ptr, output_ptr,
                               SEQ_LEN, BLOCK_SIZE, HEAD_DIM):
    pid = tl.program_id(0)  # query position index
    if pid >= SEQ_LEN: return
    
    q = tl.load(q_ptr + pid * HEAD_DIM + ...)
    
    # GPU-side block table lookup
    block_id = pid // BLOCK_SIZE
    offset = pid % BLOCK_SIZE
    
    # 直接从 GPU memory 读 block table
    physical_block = tl.load(block_table_ptr + block_id)
    
    # 读取 KV cache
    k = tl.load(k_cache_ptr + physical_block * BLOCK_SIZE * HEAD_DIM + offset * HEAD_DIM + ...)
    v = tl.load(v_cache_ptr + physical_block * BLOCK_SIZE * HEAD_DIM + offset * HEAD_DIM + ...)
    
    # Attention compute
    attn_score = tl.sum(q * k) / tl.sqrt(HEAD_DIM)
    out = attn_score * v
    tl.store(output_ptr + pid * HEAD_DIM + ..., out)
```

术语一般如何实现？如何使用？
- 实现依赖：PyTorch tensor on CUDA、Triton kernel、或手写 CUDA kernel
- 关键收益：(1) 消除 CPU→GPU block_table 传输延迟；(2) 减少 kernel launch 次数（block table 更新和 attention 可在同一 kernel 或同一 stream 上执行）；(3) 与 GPU continuous batching 的 scheduling 更紧密集成
- 注意事项：GPU 显存中 block table 占用空间极小（每个 block 仅需一个 int32 index），不影响模型可用的显存空间

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs


## Expert Sharding (Tensor Sharding of Experts / 专家张量切分)

术语解释
Expert Sharding 是 MoEShard (EuroMLSys '25) 提出的 MoE 推理优化策略，将每个 expert 的权重矩阵（W_i 和 W_o）按张量维度切分到所有 GPU（而非将完整 expert 分配到不同 GPU），使每个 GPU 持有所有 expert 的部分 shard，所有 GPU 处理全部 token 的 partial computation，通过 pointwise sum 恢复完整输出，实现与路由分布无关的 perfect load balancing。

术语是什么？
在传统 Expert Parallelism 中，不同 GPU 持有不同完整 expert，token 按 routing 结果发送到对应 GPU。当 routing 倾斜时，热门 expert 所在 GPU 过载、冷门 expert 所在 GPU 空闲。Expert Sharding 的核心洞察：expert 计算本质上是两次矩阵乘法（x · W_i · W_o），其中 W_i（shape [h_i, h_o]）可列切分、W_o（shape [h_o, h_i]）可行切分到 |G| 个 GPU，每个 GPU g 计算 x · W_i^g · W_o^g → 部分输出 y_g，最终 Σ y_g 等价于完整 expert 输出。因为所有 GPU 的计算量完全相等（均处理全部 token × 所有 expert 的 1/|G| shard），天然实现 perfect load balancing，无需 profiling、专家复制或 token dropping。

从kernel调度角度拆解术语：
```
# Expert Sharding 的 kernel 执行流程（MoEShard, Algorithm 1 + Section 3.2）
# 假设 4 GPU, 128 experts, batch_size=B, seq_len=S, hidden_dim=h

# --- Shard 准备（一次性，推理前完成）---
# 每个 expert e 的 W_i [h_i, h_o] 列切分为 |G| 份
# 每个 expert e 的 W_o [h_o, h_i] 行切分为 |G| 份
# GPU g 持有: {W_i^g[e] : shape [h_i, h_o/|G|]  for all e in E}
#           {W_o^g[e] : shape [h_o/|G|, h_i]  for all e in E}

# --- Forward Pass（per MoE block）---
# Step 1: Token Routing（每 GPU 独立）
m_expert = ROUTER(x)  # x: [B*S, h], m_expert: [B*S], token→expert 映射

# Step 2: Metadata Exchange（轻量 all-to-all broadcast）
m_sizes = count_per_expert(m_expert)  # size=[|E|], 每个 expert 收多少 token
broadcast m_sizes to all GPUs

# Step 3: Token Scatter（全复制 - 与 EP 的本质区别）
# 每 GPU 发送全部 token 给所有其他 GPU
# 每 GPU 发送 ≈ B*S*h*4 bytes → 88 MiB (batch=250, seq=120, h=768)
# NVLink 3.0 600 GiB/s → ~0.15ms, negligible
W[g][e] = tokens from GPU g assigned to expert e  # 2D 组织

# Step 4: Sharded Expert Computation（核心 kernel）
for e in E:                                    # 128 experts
    # Fusion opt 1: concatenate tokens for expert e from all GPUs
    tokens_e = cat([W[0][e], W[1][e], ..., W[|G|-1][e]])  # 合并同 expert token
    shard = LOAD_SHARD(rank, e)                 # W_i^rank[e], W_o^rank[e]
    partial = tokens_e @ W_i_shard @ W_o_shard  # partial output per GPU
    # Fusion opt 2: MegaBlocks sparse MM 将所有 expert 计算融合为 1 kernel
    split partial back to per-GPU results

# Step 5: Gather & Aggregate
send partial results back to source GPUs
y_final = sum(all partial outputs)  # pointwise addition across GPUs
```

与其他 sharding 策略的对比分析（Section 3.2）：
- W_i row-wise + W_o column-wise: W_i row 切分需在 x·W_i 后 cross-GPU sum（中间同步）
- W_i column-wise + W_o column-wise: x·W_i 后需 cross-GPU concat（中间同步）
- **W_i column-wise + W_o row-wise (MoEShard)**: 两次矩阵乘法之间无需中间同步，最优

术语一般如何实现？如何使用？
- 基于 PyTorch 实现，源码: https://github.com/sacs-epfl/moe-inference
- 要求所有 GPU 等容量等算力（同构集群），expert shard 数可被 GPU 数整除
- 代价：token 全复制（NVLink 高带宽吸收）和 partial output 求和（pointwise add, 可忽略）
- 适用场景：encoder-based MoE, batch size 较大（≥100），routing 高度倾斜
- 局限：小 batch（10）时因 token 全复制 overhead 可能慢于 DeepSpeed EP；decoder autoregressive 生成未验证

涉及论文标题：
- Accelerating MoE Model Inference with Expert Sharding

---

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

## Hierarchical All-to-All

术语解释
分层All-to-All是针对MoE Expert Parallelism的通信优化策略，将全局All-to-All拆分为intra-node（节点内高带宽）和inter-node（节点间低带宽）两个层次，充分利用两级带宽。

术语是什么？
传统All-to-All允许所有GPU直接相互通信，带宽使用效率低。分层All-to-All的关键洞察：
- 同节点GPU通过NVLink/NVSwitch互联，带宽极高（~900GB/s）
- 跨节点GPU通过InfiniBand/RoCE互联，带宽相对较低（~200GB/s）
- 应先在同节点内聚合数据，再跨节点传输，减少跨节点通信量

从kernel调度角度拆解术语。
```
# 分层All-to-All伪代码
def hierarchical_alltoall(tokens, expert_assignment, node_size):
    # tokens: [num_gpus, tokens_per_gpu]
    # node_size: GPUs per node
    
    # 阶段1：Intra-node gather（高带宽NVLink）
    for node in nodes:
        node_buffer = []
        for gpu in node.gpus:
            # 收集该节点内所有GPU需要发往其他节点的token
            for dst_gpu in range(num_gpus):
                if dst_gpu not in node.gpus:
                    node_buffer.append(tokens[gpu][dst_gpu])
    
    # 阶段2：Inter-node exchange（低带宽网络）
    # 每节点仅一个gateway GPU参与跨节点通信
    for src_node in nodes:
        gateway_gpu = src_node.gateway
        for dst_node in nodes:
            if src_node != dst_node:
                NCCL_Send(gateway_gpu, node_buffer, dst_node.gateway)
    
    # 阶段3：Intra-node scatter（高带宽NVLink）
    for node in nodes:
        distribute_received_tokens_to_gpus()
    
    # 加速比：1.4x-2x（ScheMoE vs 标准All-to-All）
```

实现框架：
- Tutel：统一的权重/数据布局支持自适应切换，无需重新格式化
- HetuMoE：针对大规模MoE的分层All-to-All实现
- DeepSpeed-MoE：分层通信 + 同数据路径token合并

术语一般如何实现？如何使用？
- 基于NCCL的send/recv primitives
- 需要配置node topology信息（哪些GPU在同一节点）
- gateway GPU选择策略影响性能
- 与expert放置策略配合使用效果最佳

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---

## MoE-Specific CUDA Kernels (Quantized Expert Computation)

术语解释
针对MoE量化expert的专用CUDA kernel，处理低精度权重（INT4/INT2/INT1）与浮点激活值的矩阵乘法，在kernel内完成反量化+浮点计算，实现真正的推理加速。

术语是什么？
量化的MoE模型如果不使用专用kernel，只能在计算前统一反量化为FP16，然后再用标准GEMM——这种情况下只能节省内存，无法加速计算。专用kernel在kernel内部融合反量化和矩阵乘法：
- **MoE-CSP**：处理4-bit/8-bit量化权重的CUDA kernel，kernel内反量化 + FP32计算
- **QMoE**：1-bit压缩格式 + 专用GPU kernel，on-the-fly反量化
- 通用方案：W4A16 kernel（4-bit权重 + 16-bit激活）

从kernel调度角度拆解术语。
```
# MoE-CSP量化kernel伪代码
__global__ void moe_quantized_expert_kernel(
    int8_t* W_q,       // INT4量化权重 [d_ffn, d_model/2]
    half* x,           // FP16输入激活值 [d_model]
    half* y,           // FP16输出 [d_ffn]
    float* scales,     // 量化scale [d_ffn]
    int d_model, int d_ffn
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_ffn) return;
    
    float acc = 0.0f;
    for (int col = 0; col < d_model/2; col++) {
        // 读取打包的INT4值（2个INT4打包为1个INT8）
        int8_t packed = W_q[row * d_model/2 + col];
        // 解包为两个INT4
        int4_t w0 = (packed & 0x0F) - 8;  // 反量化的一部分
        int4_t w1 = (packed >> 4) - 8;
        // 反量化 + MAC（乘累加）
        acc += scales[row] * (float(w0) * __half2float(x[2*col]) + 
                              float(w1) * __half2float(x[2*col+1]));
    }
    y[row] = __float2half(acc);
}

# QMoE 1-bit kernel的关键差异
# 权重为1-bit，反量化仅涉及±1乘法
# acc += (bit == 1 ? scale : -scale) * x[col]
```

术语一般如何实现？如何使用？
- CUDA C++编写，使用__half（FP16）、int4_t等数据类型
- 需要处理内存对齐（INT4打包格式）和bank conflict
- 性能关键：shared memory使用优化、warp-level同步
- 开源实现：bitsandbytes、GPTQ、AutoGPTQ等库中的量化kernel
- 典型加速比：W4A16 kernel 1.5x-3x vs FP16 GEMM

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models

---

## Block-Sparse GEMM for MoE (MegaBlocks)

术语解释
由 Gale et al. (Stanford, 2023) 提出，将 MoE 计算重新表述为 block-sparse 操作，开发专用 block-sparse GPU kernel，在不丢弃 token 的前提下高效处理 MoE 的动态负载。

术语是什么？
传统 MoE 实现先 scatter token 到各 expert 组的连续 buffer，然后对每组执行 dense GEMM。这需要数据拷贝且可能因 expert capacity 限制丢弃 token。MegaBlocks 将 MoE 问题直接映射为 block-sparse matrix multiplication：按固定 block size（如 128×128）划分 token-expert 映射矩阵，只对非空 block 执行 batched dense GEMM。

从kernel调度角度拆解术语。
```
# MegaBlocks: token_expert_map S -> block-sparse -> batched dense GEMM
S = token_expert_map                          # [T, N]
blocks = split_into_blocks(S, block_size)     # List[Block]
nonzero_blocks = [b for b in blocks if b.nnz > 0]
for blk in nonzero_blocks:
    x_blk = x[blk.token_indices]                # [B, d_model]
    w_blk = experts_weights[blk.expert_id]      # [d_ffn, d_model]
    y_blk = x_blk @ w_blk.T                     # dense GEMM on block
```

术语一般如何实现？如何使用？
- 开源：https://github.com/stanford-futuredata/megablocks
- 基于 CUDA C++ + CUTLASS block-sparse GEMM
- GitHub Stars (2024.6): 1.1K
- 局限：scatter-to-group 数据拷贝增加内存，不易扩展到非 FFN 专家

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Accelerating MoE Model Inference with Expert Sharding
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts（使用 gpt-neox + MegaBlocks + liger kernel (Triton) 进行 dropless MoE 训练，sequence length 2048, global batch size 1024）
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models
- Continual Pre-training of MoEs How robust is your router（CPT 实验使用 GPT-NeoX + Megablocks grouped GEMM kernel 在 64×A100 上进行 dropless MoE 训练。granular MoE (E=31, K=3) 每步约 1680ms，switch MoE (E=8, K=1) 约 1517ms，dense baseline 约 880ms。Megablocks grouped GEMM 将同一 batch 中路由到不同 expert 的 token 的 FFN 矩阵乘法打包为单次 batched GEMM，避免逐 expert 的小矩阵乘法 kernel launch overhead）

**MoEShard 中的 MegaBlocks 使用**：MoEShard 将 MegaBlocks 的 variable-sized block-sparse MM 用作 expert kernel fusion 的第二层优化。第一层（per-expert token concatenation）将 kernel launch 从 |E|×|G| 降至 |E|；第二层（MegaBlocks）进一步降至 1 次 kernel launch，使 kernel launch 数独立于 expert 数量。消融实验（Section 4.4）：expert < 64 时 MegaBlocks kernel 创建 overhead 使无 MegaBlocks 版略优；expert ≥ 64 时 MegaBlocks 优势递增；128 expert + 变 batch size 时 MegaBlocks 版全区间最优。

**Duo-LLM 中的 MegaBlocks 引用**：Duo-LLM 引用 MegaBlocks 证明 block-sparse matmul 可在单 GPU 上高效执行 MoE。论文假设 duo FFN 路由策略若减少 FLOPs 也将减少延迟，因为 auxiliary small FFN 足够小可与 big FFN 共存于单节点。但 Duo-LLM 未实现具体 efficient kernel 或 serving system，developing efficient implementation 被声明为"beyond the scope of this work"。

---

## ScatterMoE / ParallelLinear

术语解释
由 Tan et al. (Mila, 2023) 提出，通过 ParallelLinear 模块执行分散组的并行线性运算，避免 MegaBlocks 的 scatter-to-group 数据拷贝，中间表示保持为 PyTorch-native tensor。

术语是什么？
ParallelLinear 在不先将 token 拷贝到连续 buffer 的情况下直接执行分组矩阵运算。中间表示（如 hidden states）保持为标准 PyTorch tensor，便于扩展到非 FFN 专家模块。

从kernel调度角度拆解术语。
```
def parallellinear_forward(x, weights, group_indices):
    # x: tokens sorted by expert but not contiguous
    # group_indices: list of (start, end) per expert
    y = zeros(total_tokens, d_out)
    for expert_id, (start, end) in enumerate(group_indices):
        if start == end: continue
        y[start:end] = x[start:end] @ weights[expert_id].T
    return y
```

术语一般如何实现？如何使用？
- 开源：https://github.com/shawntan/scattermoe
- 基于 PyTorch + CUDA batched/strided GEMM
- GitHub Stars (2024.6): 140

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

**DS-MoE 中的使用**：DS-MoE (Pan et al., 2024) 在推理阶段使用 SimpleMoE 的 ParallelLinear 操作进行 MLP 层的稀疏推理。训练阶段使用 dense computation（所有 expert 全激活），无需 ParallelLinear。推理时采用混合策略：MLP 层（sparsity 高，active ratio <30-40%）使用 ParallelLinear 进行 sparse expert computation；Attention 层（sparsity 低，active ratio >60%，sparse overhead > dense benefit）使用 torch.nn dense 计算。DS-MoE 的 expert sampling 支持三种策略：Threshold（per-token 自适应选择超阈值 experts）、TopK（固定 K）、Threshold-TopK（先统计 batch 内平均 expert 数再统一 K，兼顾自适应和 batch 效率）。

---

## PIT (Permutation Invariant Transformation) for MoE

术语解释
由 Zheng et al. (2023) 提出，利用 Permutation Invariant Transformation（f(P·x) = P·f(x)，P 为置换矩阵）将 MoE 的稀疏微 tile 重组为 GPU 高效 dense tile，在不改变计算结果的前提下提升 GPU 利用率。

从kernel调度角度拆解术语。
```
# PIT Tiling: sparse micro-tiles -> dense tile -> GEMM -> inverse
micro_tiles = extract_per_expert_tiles(input_tokens, tile_size)
dense_tile = pit_rearrange(micro_tiles)      # 置换不变性保证等价
result = dense_gemm(dense_tile, merged_W)     # high GPU utilization
output = pit_inverse_rearrange(result)         # 恢复原始顺序
```

术语一般如何实现？如何使用？
- 作为深度学习编译器图优化 pass
- 需要形式化证明 operator 满足 PIT 性质
- 开源情况：论文未明确给出独立开源链接

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models

## Demand-Priority Scheduling Strategy (需求优先级调度策略)

术语解释
APTMoE 提出的 CUDA kernel 调度策略，用于协调三层加载阶段（inter-stage/inter-layer/inter-expert）产生的数据移动 kernel 对同一 PCIe 带宽的竞争。由于同方向的数据移动 kernel 不能在 GPU 上并发执行，三层加载可能互相阻塞。Demand-Priority Scheduling 通过 PriorityQueue + CUDA Event 前探机制，在 kernel 启动前动态决定加载顺序。

术语是什么？
核心机制：
- **优先级分配**：inter-expert（最高，当前层实时必需）> inter-layer（中等，下层预加载）> inter-stage（最低，下个 stage 预加载）
- **Kernel 启动前调度**：由于 CUDA 不支持 kernel 中断/恢复，在 kernel launch 前决定下一个加载目标，而非在 kernel 执行中
- **CUDA Event 前探**：在 load_stream 的倒数第二个 data movement kernel 前插入 event，通过 event.query() 检测加载进度。当 event 触发时，前一个 kernel 仍在执行，隐藏 launch overhead
- **Inter-Stream 同步**：每个 model block 关联一个 event，load_stream 完成数据移动后 record，comp_stream 的对应计算 kernel 等待该 event

从kernel调度角度拆解术语。
```
# comm_scheduler.py 核心逻辑（基于论文描述）
class DemandPriorityScheduler:
    def __init__(self):
        self.queues = {
            'inter_expert': PriorityQueue(priority=HIGH),   # 当前层实时 expert
            'inter_layer':  PriorityQueue(priority=MEDIUM), # 下层预测 expert
            'inter_stage':  PriorityQueue(priority=LOW),    # 下个 stage 预取
        }
        self.comp_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
        self.block_events = {}  # model_block_name -> torch.cuda.Event
    
    def schedule_and_launch(self):
        # 1. 选择最高优先级非空队列
        queue = self.select_highest_priority_nonempty_queue()
        
        # 2. 从队列中取预定义数量的数据移动 action
        actions = queue.pop_batch(batch_size=PREFETCH_BATCH)
        
        # 3. 在倒数第二个 action 前插入 cuda_event
        if len(actions) > 1:
            probe_event = torch.cuda.Event()
            # 先启动前 n-1 个 actions
            for action in actions[:-1]:
                self.launch_load_kernel(action)
            # 插入 probe event
            self.load_stream.record_event(probe_event)
            # 启动最后一个 action
            self.launch_load_kernel(actions[-1])
        else:
            self.launch_load_kernel(actions[0])
        
        # 4. 周期性查询 (CPU-GPU 同步)
        if probe_event.query():  # event 已触发 → 加载正在进行
            self.schedule_and_launch()  # 发起下一批加载
    
    def launch_load_kernel(self, block_name):
        # 发起 host→device cudaMemcpyAsync
        with torch.cuda.stream(self.load_stream):
            load_data(block_name)  # cudaMemcpy Host→Device
            # 记录完成 event → comp_stream 依赖
            event = torch.cuda.Event()
            event.record()
            self.block_events[block_name] = event
    
    def wait_for_data(self, block_name):
        # comp_stream 等待对应 block 的数据加载完成
        self.comp_stream.wait_event(self.block_events[block_name])
        # 执行计算
        execute_computation(block_name)
```

术语一般如何实现？如何使用？
- 基于 PyTorch 的 `torch.cuda.Stream`（双流）和 `torch.cuda.Event`（同步原语）
- PriorityQueue 用 Python `queue.PriorityQueue` 或自定义实现
- 使用 `torch.cuda.Event.query()` 的非阻塞查询 + CPU 端轮询实现 proactive scheduling
- 关键约束：同方向 cudaMemcpy 不能并发 → 必须串行化但可以提前调度
- 在 APTMoE 中位于 `comm_scheduler.py`，与 `offload.py`（加载决策）和 `R_solver.py`（分配求解）协同工作

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

## CUDA Stream Overlap for Pipeline Offloading (流水线卸载的 CUDA 流重叠)

术语解释
在使用 offloading 技术的流水线并行训练中，通过维护独立的 CUDA stream 分别执行计算（comp_stream）和数据移动（load_stream），利用 GPU 的并发执行能力使计算与 PCIe 数据搬移并行进行，以隐藏数据加载延迟。

术语是什么？
APTMoE 采用双流架构：
- **comp_stream**（计算流）：执行 MHA、gate operation、expert FFN 的 forward/backward 计算
- **load_stream**（加载流）：执行 host→device 的 cudaMemcpyAsync（加载下一层/下一个 stage 的参数）

两个 stream 通过 `torch.cuda.Event` 建立依赖关系：load_stream 加载完某个 block 后 record event，comp_stream 执行该 block 的计算前 wait 该 event。这确保了在优化 overlap 的同时不违反数据依赖。

从kernel调度角度拆解术语。
```
# GPU 时间线（以 2 个 micro-batch 的 forward 为例）
# load_stream:  [L_S1_MHA][L_S1_Gate][L_S1_E1][L_S2_MHA]...
# comp_stream:            [C_S1_MHA][C_S1_Gate][C_S1_E1]...
#                <------- overlap region ------->

# 伪代码
for micro_batch in micro_batches:
    # load_stream: 异步预取
    with torch.cuda.stream(load_stream):
        for block in preload_blocks:
            data = load_from_host(block)         # cudaMemcpyAsync H→D
            event = torch.cuda.Event()
            event.record(load_stream)
            ready_events[block] = event
    
    # comp_stream: 等待数据就绪后计算
    with torch.cuda.stream(comp_stream):
        for block in stage_blocks:
            if block in ready_events:
                comp_stream.wait_event(ready_events[block])
            execute_forward(block)               # MHA / gate / expert FFN

# 关键性能指标：
# - 加载时间完全隐藏率 = (overlap_time / total_load_time) × 100%
# - APTMoE 的三阶段加载使该比例显著高于 Mobius（Mobius 在 MoE 场景下加载阻塞计算）
```

重叠效率取决于 data-to-computation ratio：MoE 的数据量远大于 dense model（多个 expert 参数），因此重叠更困难。APTMoE 通过选择性加载（仅高热度 expert → GPU）降低 data 量，使重叠更可行。

术语一般如何实现？如何使用？
- PyTorch `torch.cuda.Stream()` 创建独立流
- `torch.cuda.current_stream()` 获取/设置当前流
- CPU 端使用 `psutil.Process().cpu_affinity()` 将进程绑定到指定 CPU 核心，避免 compute 和 I/O 线程竞争
- 注意：同一方向的 cudaMemcpy 操作即使在不同 stream 中也会在 GPU 端序列化（硬件限制）→ 因此 APTMoE 需要 demand-priority scheduling 来协调

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

---

## Tensor Partitioning for Communication Scheduling

术语解释
Tensor Partitioning 是将大型通信张量（如 All-to-All 和 Allreduce 的 gradient/activation tensors）分割为统一大小的小块（chunk/micro-op），以便调度器精确控制每个通信原语的发射时机和带宽分配。Lina 首次将其应用于 MoE 训练的 All-to-All 优先级调度。

术语是什么？
在分布式 MoE 训练中，Lina 将每个 gradient tensor 沿 token 维度分割为固定大小（如 30MB）的 micro-ops。每个 micro-op 作为一个独立的通信单元进入 priority queue，由调度器按优先级发射。关键设计：
- 不跨 gradient 混合 chunk（保持 concat 简洁）
- 使用 LibTorch 内置 `chunk` 和 `cat` API
- Partition overhead: preprocessing+concatenation 平均 1.02% step time

从kernel调度角度拆解术语。
```
# Lina Tensor Partitioning 伪代码
def partition_gradient_for_scheduling(grad, partition_size):
    """将 gradient tensor 分为 micro-ops"""
    # grad: 梯度张量，沿 token 维度
    # partition_size: 固定 micro-op 大小（如 30MB）
    num_chunks = ceil(grad.numel() * grad.element_size() / partition_size)
    micro_ops = torch.chunk(grad, num_chunks, dim=0)  # 沿 token 维度分割
    return micro_ops  # 每个 micro-op 大小均匀

# 在 backward pass 中使用
micro_ops = partition_gradient_for_scheduling(grad, 30*1024*1024)
for op in micro_ops:
    if op.type == ALLTOALL:
        priority_queue.push(op, priority=HIGH)
    else:
        priority_queue.push(op, priority=LOW)

# 调度循环
while not priority_queue.empty():
    if priority_queue.has_priority(HIGH):
        op = priority_queue.pop(HIGH)
        launch_nccl_alltoall(op)
    else:
        op = priority_queue.pop(LOW)
        launch_nccl_allreduce(op)
```

术语一般如何实现？如何使用？
- LibTorch `tensor.chunk(chunks, dim)` 沿指定维度分割
- 分割后逐个包装为 micro-op 入队
- Priority queue 按类型 (All-to-All > Allreduce) 优先级出队
- 分区大小需要平衡：小于 10MB 导致每 micro-op 传输 overhead 过大，大于 50MB 粒度不够精细
- 最优分区大小取决于模型和集群配置（Lina 默认 30MB）

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---

## Micro-Op Communication Scheduling

术语解释
Micro-Op Communication Scheduling 是 Lina 提出的训练端通信调度优化。将 All-to-All 和 Allreduce 通信分解为统一的 micro-ops，通过 priority queue 保证 All-to-All 优先获得满带宽，而 Allreduce micro-ops 仅在无 All-to-All 待处理时发射，同时 All-to-All micro-ops 与 Expert FFN 计算组成 pipeline。

术语是什么？
Lina 的 Micro-Op Scheduler 运行在每个 device 上的单线程中，维护一个 priority queue：
- All-to-All micro-op: HIGH priority，始终优先发射
- Allreduce micro-op: LOW priority，仅当队列无 All-to-All 时发射
- Combine computation 阶段停止 Allreduce 发射（预示 All-to-All 即将到来）

配合 Expert Packing，使 FFN micro-op time 与 All-to-All micro-op time 对齐，最大化 pipeline efficiency。

从kernel调度角度拆解术语。
```
# Lina Micro-Op Scheduler
class MicroOpScheduler:
    def __init__(self, partition_size=30*1024*1024):
        self.pq = PriorityQueue()
        self.partition_size = partition_size
        self.in_combine_phase = False
    
    def enqueue_gradient(self, grad, op_type):
        micro_ops = torch.chunk(grad, 
            ceil(grad.numel()*grad.element_size() / self.partition_size), dim=0)
        priority = HIGH if op_type == ALLTOALL else LOW
        for op in micro_ops:
            self.pq.push(op, priority)
    
    def schedule_loop(self):
        while True:
            # 始终优先 all-to-all
            if self.pq.has(HIGH):
                op = self.pq.pop(HIGH)
                launch_alltoall(op)
                # 每 micro-op 完成后立即启动对应 FFN
                trigger_expert_ffn_for_tokens(op.tokens)
            elif self.pq.has(LOW) and not self.in_combine_phase:
                op = self.pq.pop(LOW)
                launch_allreduce(op)
            else:
                # idle: 等待下一 micro-op 或 combine 结束
                yield_cpu()
    
    def on_combine_phase_start(self):
        self.in_combine_phase = True
    
    def on_combine_phase_end(self):
        self.in_combine_phase = False
```

All-to-All Micro-Op Pipelining 时间线:
```
         All-to-All Micro-Op 1          All-to-All Micro-Op 2          All-to-All Micro-Op 3
Stream b |====A2A-chunk1====|           |====A2A-chunk2====|           |====A2A-chunk3====|
Stream a                    |FFN-tokens1|                   |FFN-tokens2|                   |FFN-tokens3|
                            <----------- Pipelining: computation hidden behind communication ----------->
```
FFN 在第一个 All-to-All micro-op 完成后立即启动，无需等待全部 All-to-All 完成。

术语一般如何实现？如何使用？
- 每 device 单线程 priority queue (C++ `std::priority_queue`)
- LibTorch `chunk`/`cat` API 做 tensor partition/concatenation
- NCCL 通信原语 (ncclAllToAll / ncclAllReduce)
- Expert Packing 配合：当 FFN micro-op << All-to-All micro-op 时，增加每 device expert 数使 FFN 时间对齐 All-to-All（pipeline efficiency 从 33% 提升至 86%）
- Overhead: micro-op 传输 overhead 平均 +1.7% vs 不分区

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---

## Intra+Inter Rank All-Reduce (秩内+秩间全规约)

术语解释
Intra+Inter Rank All-Reduce 是 SYMI 提出的新型 all-reduce 实现，支持同一 rank 内多个 GPU slots 持有同一 expert class 的 replica（intra-rank expert data parallelism），同时保持跨 rank 的梯度同步。传统 NCCL all-reduce 仅支持跨 rank 同步，不支持同 rank 内多 replica 场景，限制了 expert placement 灵活性。

术语是什么？
传统 expert gradient all-reduce：每个 expert class 的 r 个 replica 分布在 r 个不同 rank 上，执行 NCCL all-reduce 同步梯度。限制是 expert 最多只能被复制 N 次（每个 rank 最多 1 个 instance），导致 sub-optimal placement 和 up to 20% extra token drops。

SYMI 的三步 all-reduce：
- Step 1 (Intra-rank): 每个 rank 内选举一个 slot representative，其他 slot 将 gradient 累加到 representative
- Step 2 (Inter-rank): 仅在 representative 间执行 all-reduce（跨 rank）
- Step 3 (Intra-rank broadcast): representative 归一化后将结果广播回同 rank 其他 slot

从kernel调度角度拆解术语：
```
# SYMI Intra+Inter Rank All-Reduce（per expert class）
def syMI_allreduce_expert_grads(expert_id, grads_per_slot):
    # grads_per_slot: dict {slot_idx: gradient_tensor} for local slots of this expert
    
    # Step 1: Intra-rank sum (local GPU computation, no network)
    rep_slot = min(grads_per_slot.keys())  # elect representative
    for slot, grad in grads_per_slot.items():
        if slot != rep_slot:
            grads_per_slot[rep_slot] += grad  # accumulate to rep
    
    # Step 2: Inter-rank all-reduce (NCCL, only on representatives)
    rep_grads = [grads_per_slot.get(rep_slot, zeros_like(...)) for each rank]
    allreduced_rep_grad = allreduce(rep_grads[local_rank])  # NCCL
    
    # Step 3: Intra-rank broadcast (local GPU, copy)
    num_local_replicas = len(grads_per_slot)
    normalized_grad = allreduced_rep_grad / num_local_replicas
    for slot in grads_per_slot:
        grads_per_slot[slot] = normalized_grad
    
    return grads_per_slot
```

术语一般如何实现？如何使用？
- 需要配合 Expert Placement Scheduler 的 contiguous assignment——优先将同 expert class 的 replica 放在同 rank 内
- 优势：减少 inter-node 网络流量（同 rank 内通信为本地 GPU 操作，无网络开销）
- NCCL 限制：传统 NCCL all-reduce 要求每个 rank 恰好一个参与 tensor，SYMI 的 intra-rank sum 步骤绕过此限制
- SYMI 论文实测：此 all-reduce 实现比传统实现更高效（配合 locality-enhanced placement）

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---

## Communication Group Pre-Registration (通信组预注册)

术语解释
Communication Group Pre-Registration 是 SYMI 避免运行时 NCCL communication group 创建开销的优化技术。由于 SYMI 每 iteration 的 expert placement 变化，各 expert class 的 all-reduce 通信组也会变化。在训练初始化时预创建所有可能的 contiguous-rank 通信组，训练期间仅需查表选择，避免每 iteration 动态创建 NCCL group 的巨大开销。

术语是什么？
NCCL 的 communication group 创建（`ncclCommInitAll` 或 `torch.distributed.new_group`）是阻塞式、单线程同步操作，在大集群（N=2048）中单次创建耗时可能超过 1000 秒（MegaScale 论文数据）。若 SYMI 每 iteration 动态创建新的 expert 通信组，开销完全不可接受。

从kernel调度角度拆解术语：
```
# 初始化阶段（training startup, 一次性开销）
def pre_register_groups(N_ranks):
    groups = {}
    # 仅注册 contiguous-rank groups（Expert Placement Scheduler 
    # 保证 expert 按 contiguous 方式分配）
    for start in range(N_ranks):
        for end in range(start + 1, N_ranks + 1):
            rank_list = list(range(start, end))
            groups[(start, end)] = torch.distributed.new_group(
                ranks=rank_list, 
                backend='nccl'
            )
    return groups  # O(N²/2) groups, 跨 expert 和 layer 复用

# 训练阶段（per-iteration, O(1) 查表）
def get_comm_group(expert_id, placement):
    ranks_with_expert = sorted(find_ranks_with_expert(expert_id, placement))
    # 因为 contiguous assignment, 这些 rank 是连续的
    return pre_registered_groups[(ranks_with_expert[0], ranks_with_expert[-1] + 1)]
```

术语一般如何实现？如何使用？
- 关键前提：Expert Placement Scheduler 的 contiguous assignment 策略——使 expert 实例始终分配到连续的 rank 集合上
- 组数量：O(N²/2)，例如 N=2048 → 约 2M groups，每个 group 仅包含 ranks 列表的 metadata（内存可管理）
- 组复用：同一 group 可被不同 expert、不同 layer、不同 iteration 共享（只要 rank 集合相同）
- 替代方案：使用 NCCL 的 dynamic group 或 P2P 通信，但 batch point-to-point 在 SYMI 中用于梯度收集和权重分发

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---

## Gradient Collection Load-Balancing (梯度收集负载均衡)

术语解释
Gradient Collection Load-Balancing 是 SYMI Optimizer 中用于高效收集梯度 shards 到 optimizer partition 的算法（Algorithm 2）。由于 optimizer 均匀分片在 N 个节点上，每个 optimizer partition 需要从持有对应 expert instance 的 rank 收集梯度 shard。该算法优先使用本地传输（零网络开销），远程传输则 round-robin 分配以避免网络热点。

术语是什么？
在 SYMI 中，每个 expert e_i 的梯度需要从 r_i 个 replica instance 收集到 N 个 optimizer partition。`get_source(expert_id, dst_rank)` 决定哪个 source rank 为指定的 (expert, optimizer_dst) 对提供梯度：
- 如果 dst_rank 本地持有该 expert 的 instance → 直接本地 PCIe 传输（无网络开销）
- 否则 → 从 r_i 个 remote replicas 中 round-robin 选择一个，确保梯度负载均匀分布

从kernel调度角度拆解术语：
```
# SYMI Algorithm 2: Gradient Collection
def get_source(exp_id, dst_rank):
    if dst_rank in exp_to_rank_map[exp_id]:
        return dst_rank  # local transfer preferred
    candidates = sorted(exp_to_rank_map[exp_id])
    idx = dst_rank % len(candidates)  # round-robin
    return candidates[idx]

def collect_grads():
    recv_tuples = {}  # (src_rank, dst_rank)
    send_tuples = {}  # (dst_rank, partition_idx)
    
    for exp_id in all_experts:
        # Each optimizer partition determines its source
        for dst_rank in range(N):
            src = get_source(exp_id, dst_rank)
            recv_tuples[(exp_id, dst_rank)] = (src, dst_rank)
    
    for slot, exp_id in local_expert_map.items():
        for dst_rank in range(N):
            if get_source(exp_id, dst_rank) == local_rank:
                send_tuples[(exp_id, dst_rank)] = dst_rank
    
    # Batch point-to-point: single batch_isend_irecv for all pairs
    batch_isend_irecv(send_tuples, recv_tuples)
```

术语一般如何实现？如何使用？
- 使用 PyTorch distributed 的 batch_isend_irecv（point-to-point communication）
- Round-robin 策略确保 hotspot free——任何单个 expert instance 不会成为多个 optimizer partition 的梯度源
- 本地优先策略减少约 E/(sN) 比例的网络通信（当 expert 本地有 replica 时）
- 与 weight distribution phase 对称：updated weights 从 optimizer partition 反向发送到 expert slots

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)


## Atomic Compare-and-Swap CUDA Kernel for Expert Substitution (Atomic CAS 专家替换 CUDA Kernel)

术语解释
Atomic Compare-and-Swap (CAS) CUDA Kernel for Expert Substitution 是 BuddyMoE 实现的 GPU 并行 buddy substitution kernel，使用 CUDA thread block 和 atomic CAS 操作实现无锁的专家并行替换，在 ~0ms 内完成 per-token multi-expert 的 buddy substitution，不引入 noticeable latency overhead。

术语是什么？
Kernel 配置：grid(T, 1, 1) × block(K, 1, 1)——每个 CUDA thread block 处理一个 token 的 K 个 expert 的替换，block 内每个 thread 负责一个 expert。Shared memory U_t[E] 维护当前 token 的已分配 expert set（初始化为 S 中的 expert indices）。每个 thread 检查自己的 expert e_id 是否 GPU-resident（M[e_id]），若非 resident 则遍历 buddy list B[e_id] 寻找 GPU-resident 且不在 U_t 中的 buddy。Atomic CAS（atomicCAS(&U_t[b_id], false, true)）保证无锁的 uniqueness constraint——第一个成功 claim 该 buddy 的 thread 获得它，其他 thread 必须寻找不同 buddy。

从kernel调度角度拆解术语：
```
__global__ void buddy_substitute_kernel(
    int* S, bool* M, int* B, int T, int K, int E, int H
):
    __shared__ bool Ut[E]  # block-level shared memory per token
    
    # Initialize Ut from current expert indices
    for i in range(K):  # parallel init
        e = S[blockIdx.x * K + i]
        Ut[e] = true
    __syncthreads()
    
    e_id = S[blockIdx.x * K + threadIdx.x]
    if M[e_id] == false:  # CPU-resident, need replacement
        for r in range(H):  # search buddy list up to H
            b_id = B[e_id * B_stride + r]
            if M[b_id] and atomicCAS(&Ut[b_id], false, true):
                S[blockIdx.x * K + threadIdx.x] = b_id
                break
    # No suitable buddy found: leave original e_id (fallback to on-demand)
```

Kernel 调度特征：
- 所有操作在 GPU memory 内完成（查 B table、M mask、atomic CAS），无 CPU↔GPU 传输
- grid(T,1,1) 使不同 token 的 thread blocks 在 SM 间并行
- Shared memory Ut[E] 需 E ≤ 64（DeepSeek-V2-Lite）→ shared memory per block = 64 bytes → 可忽略
- Atomic CAS 在 L2 cache 层面操作，延迟 < 100 GPU cycles，远小于 expert FFN GEMM

术语一般如何实现？如何使用？
- 实现为 CUDA __global__ kernel，在 llama.cpp 的 CUDA backend 中集成
- 在 router 输出后、expert FFN 计算前调用
- 若 no suitable GPU-resident buddy → fallback 到 prefetch original expert（pay transfer）或 skip expert per baseline MoE drop policy
- CAS 操作的竞争仅在同一 token 的多个 thread 同时选中同一 buddy 时发生（低概率：buddy lists ≤ 16 且 expert 总数 ≥ 64）

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

## CPU Chunked Attention Verification Kernel（CPU 分块注意力验证 Kernel）

术语是什么？通过联网搜索让回答具体和精准。
CPU Chunked Attention Verification Kernel 是 SpecMoEOff 为 speculative decoding 在 MoE offloading 场景下设计的 CPU 端 attention 算子。它处理 Q∈R^{n×d}, K∈R^{(l+n)×d}, V∈R^{(l+n)×d}（其中 n 为 draft tokens 数，l 为 prefix tokens 数）的 chunked attention 计算，是 speculative decoding verification 阶段的核心算子。

该 kernel 解决的关键问题：在 MoE offloading 场景下，target model 的 KV cache 全部存储在 CPU DRAM 中。若将 KV cache 传回 GPU 做 attention，会产生大量 CPU→GPU 传输开销；若在 CPU 上对每个 draft token 独立做 attention（GEMV），则需重复读取 KV cache n 次。CPU Chunked Attention Kernel 通过一次性读取 KV cache 并为所有 n 个 draft tokens 做 batch attention（GEMM），同时解决了两个问题。

从kernel调度角度拆解术语：
```
# CPU Chunked Attention Kernel 计算流程
输入: Q ∈ R^{n×d}  (n draft tokens queries)
      K ∈ R^{(l+n)×d}  (prefix + draft keys, from CPU DRAM KV cache)
      V ∈ R^{(l+n)×d}  (prefix + draft values, from CPU DRAM KV cache)
      M_draft ∈ {0,1}^{n×n}  (仅 draft-to-draft causal mask)
      # draft-to-prefix 全为 1, 不存储

# Step 1: Q@K^T via Intel MKL SGEMM
# [n, d] @ [d, l+n] → [n, l+n]
scores_full = mkl_sgemm(Q, K.T) / sqrt(d)

# Step 2: Apply mask (仅 draft 部分)
# scores_full[:, :l] 无需 mask（全 1）
scores_full[:, l:] += causal_mask(M_draft)  # M_draft: 下三角=0, 上三角=-inf

# Step 3: Softmax + Weighted Sum via Intel MKL SGEMM
attn_weights = softmax(scores_full, dim=-1)  # [n, l+n]
output = mkl_sgemm(attn_weights, V)          # [n, d]

# Batch 扩展: b 个 requests 各自独立并行
```

与 Baseline 方案的对比：

| 方案 | Q@K^T 次数 | KV Cache 读取 | CPU-GPU 传输 | Mask 内存 |
|------|-----------|-------------|------------|----------|
| GPU chunked attention | 1× (b×n GEMM) | 0 | KV cache 全量传输 | n×(l+n) |
| Naive CPU decode (per-token GEMV) | n× (b×1 GEMV) | n× 重复 | 0 | 无 |
| PyTorch CPU prefill | 1× | 重复计算 prefix | 0 | n×(l+n) |
| **SpecMoEOff CPU Chunked** | **1× (b×n GEMM)** | **1×** | **0** | **n×n** |

术语一般如何实现？如何使用？
基于 Intel oneAPI Math Kernel Library (MKL) 的 SGEMM 实现矩阵乘法，利用 CPU SIMD (AVX-512) 和 MIMD (multi-core) 能力。Mask 压缩：仅存储 n×n draft-to-draft 部分（draft-to-prefix 固定为 1，无需存储），内存从 O(n·(l+n)) 降至 O(n²)。在 SpecMoEOff 系统中，CPU Chunked Attention 随 draft length 增加逐渐成为 target model 的性能瓶颈（Table 3: 4.29s CPU Attention vs 3.53s GPU MoE），说明 CPU attention kernel 是系统性能的关键路径。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

**BigMac 的 All-to-All 维度缩减（Jin et al., 2025）**：
BigMac 从算法/模型结构层面直接减少 All-to-All 通信量——通过 DCCA（descend-communicate-communicate-ascend）策略将 All-to-All 通信从 full hidden dimension h 移至压缩后的低维 r·h。通信量公式：$C_{BigMac} = 2 \times top\_k \times \frac{ep-1}{ep} \times b \times s \times (r \cdot h) = r \times C_{baseline}$。当 r=0.25 时，通信量减少 75%（如 GPT3-XL + 64 experts + ep=32: 1,488 GB → 372 GB）。该方法仅改变模型结构（projection 顺序 + expert 内部结构），不修改 All-to-All 通信原语，因此与 Tutel 的 2DH All-to-All、Lina 的 micro-op scheduling 等系统优化正交叠加。BigMac 在 Megatron 上训练加速 1.53-3.09×，在 Tutel 上加速 1.71-3.09×，在 DeepSpeed-Inference 上推理吞吐提升 1.62-3.11×。


## Shared Tensor Based Dependency Resolving

术语解释
Shared Tensor Based Dependency Resolving 是 Comet 提出的细粒度通信-计算重叠方法，通过识别 MoE layer 中通信和计算操作共享的缓冲区（shared tensor）并对其沿特定维度分解和重调度，将粗粒度 chunk 级 pipeline overlap 升级为 token/tile 级重叠，消除通信(token级)与计算(tile级)之间的粒度不匹配。

术语是什么？
在 MoE layer 中，两个 producer-consumer pipeline 各有一个 shared tensor——layer0 的 shared tensor 是 dispatch buffer [M×topk, N]（通信的输出、GEMM 的输入），layer1 的 shared tensor 是 GEMM 输出 buffer [M×topk, N]（GEMM 的输出、reduce+通信的输入）。Shared tensor 的分解和重调度基于两个原则：
1. **沿独立维度分解**：选择 consumer operator 数据独立的维度——layer0 沿 M（token）维度（每个 token 独立），layer1 沿 N（hidden）维度（各列独立）。不能沿 consumer 需要 reduce 的维度分解。
2. **重调度对齐 tile 粒度**：分解后的 sub-tensor 按原始 GEMM tile 粒度重组，调度策略优先处理 producer 侧立即可用的数据（最小化数据依赖等待）。

从kernel调度角度拆解术语：

```
# Comet Layer0 (Communication→Computation Pipeline) 的 Shared Tensor 流程
# shared_tensor = dispatch_buffer [M×topk, N]

# Step 1: 沿 M 维度分解 shared tensor
sub_tensors = decompose_along_dim(shared_tensor, dim=M)
# 每个 sub_tensor 对应一个或少量 token，token 粒度

# Step 2: 按 source_rank 排序 sub-tensor
sorted_tensors = sort_by_source_rank(sub_tensors)
# local tokens 聚集在前（无需通信），remote tokens 聚集在后（需 NVSHMEM get）

# Step 3: 重调度 GroupGEMM tile 计算顺序
for tile in GroupGEMM.tiles:
    if tile.only_contains_local_tokens():
        priority = HIGH     # 立即开始，无数据依赖
    elif tile.partial_remote_ready():
        priority = MEDIUM   # remote token 已通过 NVSHMEM 到达
    else:
        priority = LOW      # 等待 NVSHMEM 传输完成
    schedule(tile, priority)

# Layer1 (Computation→Communication Pipeline):
# 沿 N 维度分解 → column-wise GEMM
for col_block in range(0, N, T_N):        # T_N = GEMM tile N 维度大小
    for expert in local_experts:
        partial = GEMM_tile(expert, col_block)  # 只计算当前列块
    # T_N 列完成后立即 reduce + 通信
    topk_reduce(partial_results[:, :col_block * T_N])
    NVSHMEM_write_to_remote(ready_tokens)
    # 继续下一列块的计算（与 reduce/通信重叠）
```

在 Hopper GPU (132 SMs) 上，layer0 的 GEMM tile 重调度使 local token 的计算在 NVSHMEM 拉取 remote token 期间进行，实现了 token-tier 的重叠。layer1 的 column-wise 执行使 reduce+通信与后续列 GEMM 重叠，将传统 per-expert 串行改为列方向并行。

术语一般如何实现？如何使用？
- 依赖 NVSHMEM 的 Unified Virtual Address 实现 token 级 fine-grained remote I/O（替代 NCCL 粗粒度 all-to-all）
- Shared tensor buffer 由 NVSHMEM 分配（size = 2×M×N bytes），跨所有 MoE layers 和 experts 全局复用
- 需要修改 GroupGEMM 的 tile 调度顺序（layer0: remote-dependency 最小化优先；layer1: column-wise）
- 适配 expert parallelism + tensor parallelism 混合场景：TP 下 shared tensor 沿 N 维进一步分片，但分解策略不变
- Comet 代码 ~12k lines C++/CUDA + 2k lines Python，集成在 Megatron-LM 中

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts


## Thread Block Specialization (for Fused Communication-Computation Kernels)

术语解释
Thread Block Specialization 是 Comet 提出的 fused kernel 设计模式，将通信（NVSHMEM I/O）和计算（GEMM）分配到同一 GPU kernel 内但隔离到不同的 thread blocks 中，由 GPU hardware scheduler 并发调度。这替代了传统的"垂直融合"（在 GEMM thread block 的 prologue/epilogue 中插入通信 I/O），避免了 fine-grained I/O 干扰高性能计算流水线（尤其在 Hopper TMA 异步流水线中）。

术语是什么？
在 Comet 的 fused kernel 中，两类 thread block 共存于同一 kernel launch：
- **GEMM Thread Blocks**（n^p 个）：使用标准 CUTLASS Hopper 实现——producer warp 用 TMA async copy (cp.async.bulk) 从 global memory 加载到 shared memory，consumer warp 在 tensor core 上执行 MMA。通信 I/O 完全不侵入 GEMM 流水线。
- **通信 Thread Blocks**（n^c 个）：执行 NVSHMEM get/put 进行 token 级跨 GPU 数据传输，以及 top-K reduce 操作。从 global memory 读取 GEMM 输出，处理后写回 local 或 remote memory。

两类 TB 隔离的关键优势：(1) GEMM TB 使用与融合前完全相同的 CUTLASS 实现，零性能退化；(2) 通信 I/O 的延迟波动不传播到计算流水线；(3) 可独立调节 n^c/n^p 比例适配不同负载。

从kernel调度角度拆解术语：

```
# Comet Fused Kernel 的 Thread Block 组织（Hopper, 132 SMs）
# GEMM TB 和通信 TB 在同一 kernel 内并发执行

# GEMM Thread Block（标准 CUTLASS Hopper, 每 TB 占 1 SM）:
def GEMM_thread_block(tile_A, tile_B, output_tile):
    # Producer warp: TMA async load
    cp.async.bulk(shared_A, global_A[tile_A])  # 硬件执行，不占线程
    cp.async.bulk(shared_B, global_B[tile_B])
    mbarrier.arrive_expect_tx(expected_bytes)
    
    # Consumer warp: Tensor Core MMA
    while not mbarrier.ready():
        continue  # or compute on previous tile
    accumulator = mma(shared_A, shared_B)
    # 纯计算，无通信 I/O 侵入
    store(output_tile, accumulator)

# 通信 Thread Block:
def comm_thread_block(token_indices, expert_output):
    # Step 1: 从 global memory 读取 GEMM 输出
    output = load_global_memory(expert_output, token_indices)
    
    # Step 2: Top-K reduce
    reduced = topk_reduce(output, routing_weights)
    
    # Step 3: NVSHMEM 写入 remote 或 local
    for token in reduced:
        if is_remote(token):
            nvshmem_put(token.dst_rank, token.data, token.offset)
        else:
            store_local(token.data, token.offset)
```

与垂直融合的对比：垂直融合中同一个 TB 执行 `TMA load → GEMM → comm I/O`，remote I/O 的延迟（数百 cycles）阻塞 GEMM 流水线，且 Hopper TMA 异步流水线会被同步 I/O 打破。Thread block specialization 通过空间隔离（不同 SM 上的不同 TB）而非时间隔离（同一 TB 内的阶段切换）解决此问题。

术语一般如何实现？如何使用？
- 依赖 CUDA cooperative groups 或 grid-level synchronization 来协调 GEMM TB 和通信 TB 之间的 producer-consumer 依赖
- GEMM TB 使用标准 CUTLASS 模板生成，端口到 Ampere/Volta 仅需替换对应架构的 compute TB 实现
- 通信 TB 使用 NVSHMEM API（`nvshmem_put`, `nvshmem_get`, `nvshmem_wait`）
- SM 资源限制：每个 SM 只能容纳 1 个 thread block（Hopper 上每 SM 1 TB），GEMM TB 和通信 TB 竞争 SM
- 总 TB 数 = SM 数 (132 on H800)，n^c + n^p ≤ 132

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts


## Adaptive Thread Block Assignment

术语解释
Adaptive Thread Block Assignment 是 Comet 提出的运行时资源分配策略，针对 fused kernel 中通信 TB（n^c）和计算 TB（n^p）的比例进行自适应选择。由于通信和计算负载随输入 token 长度 M、并行策略（TP×EP）动态变化，最优的 n^c/n^p 分割点也随之改变。Comet 通过 offline profiling + runtime lookup 实现自适应。

术语是什么？
Comet 预编译多个不同 n^c/n^p 比例的 kernel 变体，在部署前对每种 (M, EP, TP) 配置进行 profiling，记录最优 n^c 值为 metadata。运行时根据实际配置查表选择最优 kernel。该方法基于观察：最优 n^c 随输入 token 长度 M 增大而增大（计算负载增长快于通信负载），随 TP 减小而增大（TP 越小每个 GPU 的 expert 越多，通信占比越高）。

从kernel调度角度拆解术语：

```
# Comet Adaptive Assignment 决策流程

# Offline Profiling Phase（部署前执行一次）:
metadata = {}
for M in candidate_token_lengths:         # e.g. [256, 512, ..., 16384]
    for (EP, TP) in candidate_parallelisms: # e.g. [(8,1), (4,2), (2,4)]
        best_nc = None
        best_latency = INF
        for nc in range(0, total_SMs):     # total_SMs = 132 (H800)
            np = total_SMs - nc
            kernel = precompiled_kernels[(nc, np)]
            latency = profile_kernel(kernel, M, EP, TP)
            if latency < best_latency:
                best_latency = latency
                best_nc = nc
        metadata[(M, EP, TP)] = best_nc

# Runtime Phase（每次 MoE layer forward）:
def comet_moe_forward(M, EP, TP):
    key = (M, EP, TP)
    nc = metadata[key]      # O(1) lookup
    kernel = precompiled_kernels[nc]
    kernel.launch(shared_tensor, routing_map, expert_weights)
```

观察到的规律（Figure 8）：
- TP=8, M=4096 → optimal n^c=18；M=16384 → optimal n^c=26
- M=16384, TP=8 → optimal n^c=26；TP=4 → optimal n^c=46
- 解释: 通信和计算的数据量均随 M 线性增长，但各自的 SM 资源需求 scalability 不同——计算需要更多 SM 处理更大的 M，而通信 I/O 饱和带宽后额外 SM 收益递减

术语一般如何实现？如何使用？
- 预编译内核库包含多个 n^c 变体（如 n^c ∈ {18, 26, 46, ...}），每种变体为独立 CUDA kernel
- Profiling 需在目标硬件上进行（不同 GPU 架构的 SM 数、NVLink 带宽、计算能力不同）
- 运行时查表 O(1)，无调度开销
- 局限：无法处理 M 的连续变化（只能匹配 profiled 离散点）；极端 imbalanced token distribution 时最优值可能偏移
- 未来可扩展为 runtime 动态调度的 feedback-based 自适应

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts


## NVSHMEM

术语解释
NVSHMEM 是 NVIDIA 的 OpenSHMEM 实现，为 GPU 集群提供 Partitioned Global Address Space (PGAS) 编程模型。它创建跨多 GPU 内存的全局地址空间，支持从 CUDA kernel 内部、CUDA stream 或 CPU 发起单边通信操作（put/get/atomic），无需目标 GPU 的显式参与。

术语是什么？
NVSHMEM 的核心特性：
- **GPU-Initiated (Intra-Kernel) Communication**：在 CUDA kernel 内直接调用 `nvshmem_put`/`nvshmem_get`/`nvshmem_atomic_*`，无需返回 CPU 端。消除 kernel launch/synchronization 边界。
- **One-Sided Communication**：仅发起方 PE 需要活跃，目标 GPU 无需显式同步。匹配 GPU 的大规模并行和延迟隐藏架构。
- **Fine-Grained Data Access**：支持小粒度、不规则、动态的通信模式，这恰是 NCCL 粗粒度 collective 做不到的。
- **硬件支持**：Intra-node 走 NVLink/PCIe (CUDA IPC)，Inter-node 走 InfiniBand/RoCE (RDMA)。

在 Comet 中，NVSHMEM 是 fine-grained overlapping 的通信基础：替代 NCCL all-to-all（必须等全部 token 就绪后整体传输），实现 token 级 remote I/O——每个 token 独立通过 NVSHMEM 读写 remote GPU memory，使计算 tile 可以立即消费已到达的 token 而无需等待整批。

从kernel调度角度拆解术语：

```
# NVSHMEM vs NCCL 在 MoE 中的对比

# NCCL 方式（coarse-grained）:
all_tokens_ready = wait(all_to_all_complete)  # 阻塞等待全部 token
for expert in experts:
    expert_gemm(all_tokens_ready[expert])      # 批量计算

# NVSHMEM 方式（fine-grained, Comet）:
# 与 GEMM 在同一 fused kernel 内
for tile in tiles:
    for token in tile:
        if is_remote(token):
            # 按需拉取单个 token（可能还在传输中）
            nvshmem_get(token.data, src_rank, offset)
        # token 就绪后立即可用于 tile 计算
    compute_tile(tile)
```

NVSHMEM 在 Comet 中的内存开销：buffer size = M×N elements（BF16/FP16 时为 2MN bytes），Mixtral M=4096 时仅 32MB，Qwen2 仅 16MB，跨所有 MoE layers 和 experts 全局复用。

术语一般如何实现？如何使用？
- 开源：https://github.com/NVIDIA/nvshmem
- API 风格：类似 MPI 单边通信（`shmem_put`, `shmem_get`, `shmem_barrier_all`），但在 GPU kernel 内调用
- 需要 NVLink 或 PCIe peer-to-peer 支持；多节点需 InfiniBand/RoCE RDMA
- 相比 NCCL 的优势：composable、low-level、fine-grained；劣势：无 collective 优化（需自行管理同步）
- 典型用途：不规则通信模式（BFS、sparse computation）、通信-计算融合 kernel（Comet、Flux）
- 文档：https://docs.nvidia.com/nvshmem/api/index.html

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts


## GroupGEMM (Grouped GEMM for MoE)

术语解释
GroupGEMM 是针对 MoE 场景的批量矩阵乘法 kernel，将多个 expert 的独立 GEMM 操作合并为单次 kernel launch，避免 per-expert kernel launch overhead，同时利用 GPU 并行性提升小 expert 的计算效率。

术语是什么？
在 MoE 的 expert FFN 中，每个 expert 需要处理不同数量、不同来源的 tokens。传统实现为每个 expert 独立 launch GEMM kernel，当 expert 数量多且单个 expert token 少时产生大量 kernel launch overhead 和 GPU under-utilization。GroupGEMM 将所有 expert 的 GEMM 打包为单次调用，内部处理不同 expert 的不同输入形状（variable-sized batched GEMM）。Comet 基于 CUTLASS 的 GroupGEMM 模板生成高效率 kernel，并对 tile 调度顺序进行重排以实现通信-计算重叠。

从kernel调度角度拆解术语：

```
# GroupGEMM 在 MoE layer0 中的执行（Comet 版本）
# 输入: tokens 已按 expert 分组，各 expert token 数不同

# CUTLASS GroupGEMM 的 tile 调度（标准）:
for problem_idx in range(num_experts):
    m = token_counts[problem_idx]  # 该 expert 的 token 数（变化！）
    for tile_m in range(0, m, TILE_M):
        for tile_k in range(0, K, TILE_K):
            for tile_n in range(0, N, TILE_N):
                # 标准 tile 计算
                GEMM_tile(tile_m, tile_n, tile_k)

# CUTLASS GroupGEMM 的 tile 调度（Comet - 重排序）:
# 按数据依赖重排序: local token tiles 优先
tiles_sorted = sort_tiles_by_remote_dependency(
    all_tiles, 
    key=lambda t: count_remote_tokens(t),  # remote token 少的优先
    ascending=True
)
for tile in tiles_sorted:
    GEMM_tile(tile)
```

在 Hopper 架构上，CUTLASS GroupGEMM 内部使用 TMA (cp.async.bulk) 指令实现异步 global→shared memory 数据传输，producer warp 发起 TMA 请求后 consumer warp 在 tensor core 上执行 MMA，形成软件流水线。Comet 保持此流水线不变，仅在 tile 调度层面注入重排序逻辑。

术语一般如何实现？如何使用？
- CUTLASS 3.x 提供 `cutlass::gemm::kernel::Gemm` 的 Grouped GEMM 变体（`cutlass::gemm::grouped` 命名空间）
- NVIDIA 也提供专门的 grouped_gemm 库：https://github.com/fanshiqing/grouped_gemm
- Megatron-LM 默认使用 CUTLASS GroupGEMM（Megatron-Cutlass baseline）
- Comet 扩展了 GroupGEMM 的 tile 调度和 shared tensor 管理，增加 NVSHMEM 通信 TB 形成 fused kernel
- Triton 也支持 grouped GEMM（通过 `tl.dot` 的 block-pointer API）

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

**EPS-MoE 的 GroupGemm vs DenseGemm 动态切换**：
EPS-MoE 通过 profiling 发现：GroupGemm 和 DenseGemm 在不同输入规模下各有优势：
- **m < 2048**（如 decode 阶段）：GroupGemm 效率更高
- **m ≥ 4096**（如 prefill 阶段）：DenseGemm 效率反超 GroupGemm

关键洞察：(1) 对于 GroupGemm，当输入 size 达到一定阈值后，增加 group 数和 SM 数都不会进一步提高吞吐量（图5b,c）；(2) 通过水平切分输入按行+权重按专家切分，当 pipeline 数 N=E（专家数）时，GroupGemm 退化为 DenseGemm，可利用 cublas 的更高效率。

EPS-MoE 的 load-aware 自适应策略：根据输入 token 数 m 动态选择 GEMM 实现，在 Expert Pipeline Scheduler 中各 pipeline stage 可独立选择 GroupGemm 或 DenseGemm。


## DualPipe Pipeline Parallelism (DualPipe 流水线并行)

术语解释
DualPipe 是 DeepSeek-V3 提出的双向流水线并行调度算法，用于 MoE 大模型分布式训练。核心创新：(1) 将每个 forward/backward chunk 拆分为 attention、all-to-all dispatch、MLP、all-to-all combine 四个组件（backward 进一步拆分为 backward for input 和 backward for weights）；(2) 通过手动调整 GPU SM 比例实现 all-to-all 和 PP 通信与计算的完全重叠；(3) 双向调度：micro-batches 从 pipeline 两端同时注入，减少 pipeline bubble。比 1F1B 和 ZB1P 的 bubble 更小，仅需 PP stages 和 micro-batches 可被 2 整除。

术语是什么？
DualPipe 的关键设计：(1) **Chunk 分解**：forward chunk = [Attention | Dispatch | MLP | Combine | PP_Comm]，backward chunk = [Attn_BW_Input | Attn_BW_Weight | Disp_BW | MLP_BW_Input | MLP_BW_Weight | Comb_BW | PP_Comm]；(2) **Overlap 策略**：一对 forward+backward chunk 中，通信（all-to-all + PP）完全与计算重叠；(3) **Bubble formula**：Bubble = (PP-1)/(PP) * (F&B-3W)/(F+B-W)，小于 ZB1P 和 1F1B；(4) **内存**：峰值激活内存 = PP/(PP+1) * (2× normal)，需保留两份模型参数，但大 EP size 下参数显存占比小，总体可接受。

从kernel调度角度拆解术语：
```
=== DualPipe 调度时间线 (8 PP stages, 双向) ===

Time ──────────────────────────────────────────────────────────────►

正向 micro-batch (forward direction):
  Stage0: [F0_Attn][F0_Disp][F0_MLP][F0_Comb]  [F1_Attn][F1_Disp]...
  Stage1:          [F0_Attn][F0_Disp][F0_MLP][F0_Comb]  [F1_Attn]...
  ...

反向 micro-batch (reverse direction, symmetric):
  Stage7: [Fr0_Attn][Fr0_Disp][Fr0_MLP][Fr0_Comb]  [Fr1_Attn]...
  ...

一对 Forward+Backward Chunk 的重叠细节:
  Forward:  ┌─Attn─┬─Disp─┬──MLP──┬─Comb─┐
  Backward: │Attn_BW_In│Attn_BW_W│Disp_BW│MLP_BW_In│MLP_BW_W│Comb_BW│
  Overlap:  │██████████│        │███████│         │        │███████│
            ↑ All-to-all communication hidden ↑   ↑ PP comm hidden ↑

SM 分区策略:
  - 计算 SMs: 112/132 (attention + MLP forward/backward)
  - 通信 SMs: 20/132 (dispatch + combine, warp specialization)
  - 动态调整: 根据实际 workload 在通信 channel 间分配 warp 数
```

术语一般如何实现？如何使用？
DeepSeek-V3 训练使用 16-way PP，配合 64-way EP（跨 8 nodes）和 ZeRO-1 DP。DualPipe 要求：(1) pipeline stages 和 micro-batches 可被 2 整除（无需 micro-batches 被 stages 整除，比 Chimera 更灵活）；(2) bubble 和激活内存不随 micro-batch 数增加而增长。与 Chimera 的对比：Chimera 要求 micro-batches 被 PP stages 整除，DualPipe 仅要求可被 2 整除。DualPipe 的关键优化：随着模型 scale up，只要维持恒定的计算-通信比，cross-node fine-grained experts 的 all-to-all 通信开销可近零。

涉及论文标题：
- DeepSeek-V3 Technical Report

## Cross-Node All-to-All Communication with Warp Specialization (基于Warp特化的跨节点全对全通信)

术语解释
Cross-Node All-to-All Communication with Warp Specialization 是 DeepSeek-V3 为 MoE 训练中 expert dispatch/combine 设计的高效通信 kernel。使用 warp specialization 技术（Bauer et al. 2014），将 20 SMs 划分为 10 个通信通道，每个通道内由不同 warp 分别处理 IB send、IB-to-NVLink forward、NVLink receive（dispatch）或 NVLink send、NVLink-to-IB forward+accumulate、IB receive+accumulate（combine）。定制 PTX 指令 + auto-tuned chunk size 最小化 L2 cache 污染。

术语是什么？
通信 kernel 的设计基于 H800 集群拓扑：节点内 NVLink 160 GB/s ≈ 3.2× IB 50 GB/s。策略：(1) token 先通过 IB 传输到目标节点上同 in-node index 的 GPU，再通过 NVLink 转发到持有目标 expert 的 GPU，IB 和 NVLink 传输完全流水线重叠；(2) 每 token 限制最多 4 个节点（M=4），平均每节点选 3.2 experts，实际 K_r=8，理论上可扩展到 13 experts 而不增加通信开销；(3) 仅 20/132 SMs 即可跑满 IB+NVLink 带宽。

从kernel调度角度拆解术语：
```
=== Cross-Node All-to-All Dispatch Kernel ===

// 20 SMs, 10 communication channels, warp specialization
// 每个 channel 处理一组 token 的 dispatch

Channel[k] (k=0..9):
  // Warp 0: IB Send
  for token in channel_tokens:
    if token_has_remote_experts:
      data = load_FP8_activation(token)            // HBM → registers
      post_IB_send(data, target_node, target_gpu)  // RDMA write to remote

  // Warp 1: IB-to-NVLink Forward (on target node)
  for incoming_ib_token:
    data = IB_recv_buffer → shared_memory
    forward_via_NVLink(data, target_expert_gpu)    // intra-node NVLink

  // Warp 2: NVLink Receive (on target GPU)
  for incoming_nvlink_token:
    data = NVLink_recv_buffer → HBM (expert_input_buffer)

// Dynamic warp allocation: 根据实际 workload 调整各 task 的 warp 数
// L2 cache 优化: PTX ld.global/st.global with cache eviction hints (cg/evict)
// Chunk size auto-tuning: 平衡 throughput vs L2 interference

=== Combine Kernel (reverse direction) ===
Channel[k]:
  // Warp 0: NVLink Send
  expert_output → NVLink send to aggregation GPU within node

  // Warp 1: NVLink-to-IB Forward + FP32 Accumulation
  NVLink_recv → accumulate in shared_memory (FP32) → IB send

  // Warp 2: IB Receive + FP32 Accumulation (on source node)
  IB_recv → FP32 accumulate → HBM (final output)
```

术语一般如何实现？如何使用？
Warp specialization 的关键：每个 warp 独立执行一个通信子任务，通过 shared memory 进行 warp 间数据交换。PTX 指令优化：使用带 cache bypass hint 的 load/store 指令（绕过 L2 cache 避免污染计算 SMs 的数据）。与 computation stream 重叠：dispatch/combine kernel 在独立 CUDA stream 上执行，与 attention/MLP 计算并行。此设计也应用于推理阶段的 all-to-all 通信（prefill 和 decode），但 decode 阶段使用 direct P2P IB（IBGDA）替代 warp specialization pipeline。

涉及论文标题：
- DeepSeek-V3 Technical Report

## Fine-Grained FP8 Quantization (细粒度FP8量化)

术语解释
Fine-Grained FP8 Quantization 是 DeepSeek-V3 提出的低精度训练量化策略，通过在小粒度元素组（1×128 tile for activation, 128×128 block for weight）级别进行缩放，解决 FP8 格式动态范围有限导致的量化误差问题。与传统的 per-tensor scaling 不同，fine-grained scaling 使每个小 group 有独立的 scaling factor，从而更好地容纳 outlier 值（如激活中的 massive activations）。

术语是什么？
量化粒度：(1) Activation: 1×128 tile（per token per 128 channels），即每个 token 沿 hidden dimension 每隔 128 channels 用一个 scale；(2) Weight: 128×128 block（per 128 input channels per 128 output channels）。Scale factor 沿 GEMM inner dimension K 方向，非标准 FP8 GEMM 直接支持，需结合 CUDA Core promotion 实现：每 N_c=128 个 WGMMA 结果取出，在 CUDA Core 上乘 scale factor 并做 FP32 累积。

从kernel调度角度拆解术语：
```
=== Fine-Grained FP8 Quantization + GEMM ===

// Activation quantization (1×128 tile-wise, online)
// X_BF16: [M, K], batch M tokens × inner dim K
for i in 0..M-1:                        // per token
    for j in 0..(K/128)-1:              // per 128-channel tile
        tile = X_BF16[i, j*128:(j+1)*128]
        scale_X[i,j] = max(abs(tile)) / 448.0   // E4M3 max = 448
        X_FP8[i, j*128:(j+1)*128] = tile / scale_X[i,j]

// Weight quantization (128×128 block-wise, online)
// W_BF16: [K, N]
for i in 0..(K/128)-1:
    for j in 0..(N/128)-1:
        block = W_BF16[i*128:(i+1)*128, j*128:(j+1)*128]
        scale_W[i,j] = max(abs(block)) / 448.0
        W_FP8[i*128:(i+1)*128, j*128:(j+1)*128] = block / scale_W[i,j]

// FP8 GEMM with Scaled Accumulation
// C = X_FP8 × W_FP8, with per-group scaling
C = zeros([M, N], FP32)
for k_step in 0..(K/128)-1:
    // WGMMA: [M, 128] × [128, N] → [M, N] partial (Tensor Cores, ~14-bit)
    partial = WGMMA(X_FP8[:, k_step*128:(k_step+1)*128],
                     W_FP8[k_step*128:(k_step+1)*128, :])
    if (k_step+1) % 4 == 0:              // N_c=128 elements interval
        // CUDA Core FP32 promotion + dequantization
        scale = scale_X[:, k_step//4] * scale_W[k_step//4, :]  // broadcast
        C += FP32_promote(accumulator) * scale
    else:
        accumulate(TensorCore, partial)  // limited precision
```

术语一般如何实现？如何使用？
与 microscaling (MX) 格式理念一致（NVIDIA Blackwell 已宣布支持）。H800 上实现限制：Tensor Core 不支持 per-group scaling，需额外 CUDA Core 步骤；频繁的 Tensor Core ↔ CUDA Core 数据移动限制效率。未来硬件建议：Tensor Core 直接接收 scaling factors，在 MMA 内部完成 group-scaled 累积+dequantization。在线量化效率问题：需从 HBM 读取 BF16 值进行量化，再写 FP8 回 HBM，建议融合 FP8 cast + TMA access 为单 fused 操作。与 TransformerEngine 的 delayed scaling 不同：DeepSeek-V3 使用 online max 计算替代历史值推断，更精确。

涉及论文标题：
- DeepSeek-V3 Technical Report

## Expert Parallelism with Node-Limited Routing (专家并行与节点限制路由)

术语解释
Expert Parallelism (EP) 是 MoE 模型的分布式并行策略：将不同 expert 的完整权重分配到不同 GPU，token 通过 all-to-all 通信被发送到持有其选中 expert 的 GPU 进行计算。DeepSeek-V3 使用 64-way EP（跨 8 nodes）并引入 Node-Limited Routing 约束（每 token 最多路由到 M=4 个节点），使 IB 跨节点流量可控，配合 DualPipe 和 warp specialization 通信 kernel 实现近乎零通信开销。

术语是什么？
DeepSeek-V3 EP 配置：64 GPUs 承载 256 routed experts + 1 shared expert × 58 MoE layers。每个 GPU 承载 4 个 experts（256/64=4）。训练时使用 16-way PP + 64-way EP + ZeRO-1 DP。Node-Limited Routing：每 token 最多路由到 M=4 个节点，平均每节点选 3.2 experts，实际 K_r=8（理论上可扩展到 13 而通信量不变）。推理时 prefill 使用 EP32，decode 使用 EP320。

从kernel调度角度拆解术语：
```
=== DeepSeek-V3 EP with Node-Limited Routing ===

// Training configuration: 64 GPUs (8 nodes × 8 GPUs/node)
// PP=16, EP=64, ZeRO-1 DP

// Node-Limited Gating (per token)
selected = TopK({s_{j,t} + b_j | j=1..256}, K_r=8)
// Check node constraint:
nodes_used = {node_of(expert_j) for j in selected}
if len(nodes_used) > 4:  // M=4
    // Re-select TopK but limit to at most 4 nodes
    // Each node keeps its top (K_r/M=2) experts by affinity
    for node in nodes_used (keep top 4 by total affinity):
        keep top 2 experts per node

// All-to-All Dispatch flow (warp specialization, 20 SMs)
for token t:
    for expert e in selected[t]:
        target_gpu = expert_to_gpu[e]
        if node_of(target_gpu) != node_of(current_gpu):
            // Cross-node: IB → NVLink pipeline
            IB_send(activation[t], node_of(target_gpu),
                    gpu_idx_within_node(target_gpu))
            // On target node: NVLink forward to expert GPU
        else:
            // Intra-node: NVLink only
            NVLink_send(activation[t], target_gpu)

// Expert FFN Execution
for gpu in 0..63:
    process_batched_tokens_for_4_local_experts()

// All-to-All Combine
// Reverse of dispatch, with FP32 accumulation at aggregation points
```

术语一般如何实现？如何使用？
EP vs TP trade-off：TP 将每个 expert 权重切分到多 GPU，每 GPU 参与每 token 计算但通信开销大；EP 每 GPU 仅处理路由到本地的 token，通信量与激活 expert 数成正比但 GPU 负载可能不均。Node-Limited Routing 是 EP 的关键优化：限制跨节点通信量，配合 IB/NVLink 带宽差异（NVLink 160 GB/s vs IB 50 GB/s ≈ 3.2×）。DeepSeek-V3 的 EP 实现依赖自研 HAI-LLM 框架，而非标准 NCCL all-to-all。NCCL all-to-all 不做 node-limited 优化，浪费 IB 带宽。

涉及论文标题：
- DeepSeek-V3 Technical Report

## Expert-Slicing (专家内张量切分, DeepSpeed-MoE)

术语解释
Expert-Slicing 是 DeepSpeed-MoE 推理系统提出的附加并行维度：将 Tensor-Slicing 应用于 Expert 参数内部，对单个 Expert 的权重矩阵进行行/列切分到多个 GPU，在 Expert Parallelism 的基础上进一步减少每 GPU 的计算量和内存需求。当可用 GPU 数超过 Expert 数时特别有用。

术语是什么？
Expert Parallelism 将不同 Expert 放到不同 GPU，极限是 EP = E（E = Expert 总数，每 GPU 恰好 1 expert）。但当需要更低延迟需要更多 GPU 时（GPU 数 > Expert 数），Expert-Slicing 提供额外的切分维度。

与 Expert Sharding (MoEShard) 的区别：
- **Expert-Slicing (DeepSpeed-MoE)**：Expert Parallelism + Tensor-Slicing within Experts 的组合。每个 GPU 先接收属于其 Expert(s) 的 token（EP 路由），然后 Expert 内部以 tensor-slicing 方式跨多个 GPU 协同计算。本质是 EP 和 TP 的嵌套。
- **Expert Sharding (MoEShard)**：所有 GPU 持有所有 Expert 的部分 shard，所有 GPU 处理所有 token 的 partial computation，完全替代 EP。本质是无 EP 的全 shard 方案。

从kernel调度角度拆解术语：
```
# Expert-Slicing 示例：EP=128, Expert-slicing degree=4 per expert
# Total 512 GPUs, 每 expert 由 4 GPU 协同处理

# 切分方式（per expert）：
W1 [h_in, h_inter] → 列切分为 4 份: W1_g0, W1_g1, W1_g2, W1_g3（每 GPU [h_in, h_inter/4]）
W2 [h_inter, h_out] → 行切分为 4 份: W2_g0, W2_g1, W2_g2, W2_g3（每 GPU [h_inter/4, h_out]）

# Forward（Expert e, on 4 GPUs）:
# Step 1: EP routing - tokens for expert e arrive at GPU group {g0, g1, g2, g3}
# Step 2: Per-GPU computation（所有 4 GPU 并行）:
partial_g0 = tokens @ W1_g0 → GeLU → @ W2_g0
partial_g1 = tokens @ W1_g1 → GeLU → @ W2_g1
partial_g2 = tokens @ W1_g2 → GeLU → @ W2_g2
partial_g3 = tokens @ W1_g3 → GeLU → @ W2_g3
# Step 3: All-Reduce within expert-slicing group
output_expert_e = AllReduce(partial_g0, partial_g1, partial_g2, partial_g3)
```

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理系统（开源：https://github.com/microsoft/DeepSpeed）
- 当 GPU 数超过 Expert 数时自动应用（latency-stringent scenarios）
- 切分策略与 Tensor-Slicing 相同：W1 列切分 + W2 行切分（避免中间同步）
- Expert-Slicing group 内部 All-Reduce 仅限于节点内（NVLink），跨节点使用 Expert Parallelism

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---

## MoE Gating Kernel Fusion (MoE 门控 Kernel 融合)

术语解释
MoE Gating Kernel Fusion 是 DeepSpeed-MoE 推理系统提出的 CUDA kernel 优化：将 Gating 函数的 Top-K 选择、Cumsum（Blelloch scan 并行前缀和）、Scatter 等操作融合为单个 CUDA kernel，使用 dense token-to-expert mapping table 替代 sparse one-hot 表示，消除 sparse einsum 的大量零运算，实现 6x+ MoE kernel 延迟降低。

术语是什么？
传统 MoE 实现中 Gating 相关计算分为多个独立 kernel：创建 sparse one-hot mask（S x E 矩阵，其中 (E-1)/E 为零）、Sparse×Dense Einsum（S x E x M x ce，其中 (E-1)/E 为与零相乘）、Cumsum（计算每 expert 处理多少 token）。这些操作因 sparse 表示导致大量无效计算和 kernel launch overhead。

DeepSpeed-MoE 的关键优化：
1. **Dense Mapping Table**：用 `expert_id[S]`（每 token 目标 expert）和 `local_id[S]`（在同 expert 的 tokens 中的位置）替代 S×E sparse one-hot
2. **Kernel Fusion**：Top-K + Cumsum + Scatter 合并为单个 kernel
3. **Blelloch Scan**：并行前缀和算法实现 GPU 上高效的 Cumsum
4. **Data Layout Transform**：替代 sparse einsum 进行 token 排序/反排序

从kernel调度角度拆解术语：
```
// Fused Gating Kernel (1 CUDA kernel, S threads)
// Input:  gate_logits[S][E],  S=num_tokens, E=num_experts
// Output: expert_id[S], local_id[S], expert_offset[E+1]

__global__ void fused_moe_gate(
    float* gate_logits, int S, int E,
    int* expert_id, int* local_id, int* expert_offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= S) return;
    
    // Phase 1: Top-1 selection
    float max_logit = -INFINITY;
    int best_expert = 0;
    for (int e = 0; e < E; e++) {
        if (gate_logits[tid * E + e] > max_logit) {
            max_logit = gate_logits[tid * E + e];
            best_expert = e;
        }
    }
    expert_id[tid] = best_expert;
    
    // Phase 2: Atomic count per expert
    int pos = atomicAdd(&expert_offset[best_expert + 1], 1);
    local_id[tid] = pos;
}

// After kernel: Blelloch Scan on expert_offset to compute prefix sum
// expert_offset[i] = sum_{j=0}^{i-1} expert_counts[j]
// Gives starting position for each expert's tokens in output buffer

// Data Layout Transform (替代 Sparse Einsum):
// From: O = Softmax(G) ⊙ X  (sparse-dense einsum, S×E×M×ce ops)
// To:   for t in 0..S: output[expert_offset[expert_id[t]] + local_id[t]] = input[t]
// Complexity: O(S×E×M×ce) → O(S×M×ce)
```

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理系统（开源：https://github.com/microsoft/DeepSpeed）
- Blelloch Scan 是 GPU 上高效的并行前缀和算法：up-sweep (reduce) + down-sweep (propagate)
- 数据布局变换融合 gating probability 缩放：在反排序时直接乘以对应的 gate probability
- 组合优化实现 MoE kernel 延迟降低 6x+

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---

## Dense Token-to-Expert Mapping (稠密 Token-Expert 映射, MoE 数据布局变换)

术语解释
Dense Token-to-Expert Mapping 是 DeepSpeed-MoE 推理系统提出的 MoE 优化技术：用稠密的 token-to-expert 映射表（expert_id[S] + local_id[S]）替代传统 sparse one-hot 掩码（S×E）进行 MoE token routing，将 token 排序和反排序实现为显式数据布局变换而非 sparse-dense einsum 乘法。

术语是什么？
传统 MoE 实现中 token routing 通过 sparse einsum 完成：创建 S×E 的 one-hot 掩码矩阵 M（M[t][e]=1 if token t → expert e），执行 M @ X 将 token 分配到各 expert。此操作复杂度 O(S×E×M)，而 one-hot 矩阵中仅 S 个非零元素（top-1 gating 下），(E-1)/E 的运算为与零相乘。

Dense Token-to-Expert Mapping 替代方案：
- 用 expert_id[S]（稠密数组，每元素为 0..E-1 的 expert id）替代 S×E 的 sparse 矩阵
- 用 local_id[S]（每 token 在其目标 expert 中的局部位置）替代 sparse scatter
- 用 expert_offset[E+1]（每 expert 的起始位置，由 cumsum 计算）组织输出缓冲区
- Token 排序 = 直接按 expert_id + local_id 索引 memcpy → 无需矩阵乘法
- Token 反排序 = 逆向索引 memcpy + gate probability 缩放 → 无需矩阵乘法

从kernel调度角度拆解术语：
```
// 传统方法：Sparse Einsum
// M: [S, E] one-hot, X: [S, M]
// O = M^T @ X       // [E, S] @ [S, M] → [E, M], Sparse×Dense, O(S×E×M) ops
// 其中 (E-1)/E 为零乘法

// 优化方法：Data Layout Transform
// Input:  X[S][M], expert_id[S], local_id[S], expert_offset[E+1]
// Output: X_sorted[E][ce][M]

// Sort (by expert_id):
for t in 0..S:
    e = expert_id[t]
    pos = local_id[t]                    // 已由 cumsum 计算
    X_sorted[e][pos] = X[t]              // Direct memcpy, no multiply

// Unsorted (back to original order) with gate probability:
for t in 0..S:
    e = expert_id[t]
    pos = local_id[t]
    output[t] = gate_prob[t] * Y_expert[e][pos]   // 融合 probability scaling

// 复杂度对比：
// Sparse Einsum: S × E × M × ce → O(S·E·M·ce)（立方+零运算）
// Layout Transform: S × M × ce → O(S·M·ce)（仅非零元素）
```

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理系统（开源：https://github.com/microsoft/DeepSpeed）
- 需要 Gating Kernel Fusion 提前计算 expert_id[], local_id[], expert_offset[]
- 内存布局要求：所有 expert 的 token buffer 预分配为 [E, capacity, M]
- 对于 capacity > local_id 的空余位置填充为零（不影响后续 FFN 计算）

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

## Fused GPU Kernel for MoE Coreset Selection

术语解释
一种自定义 CUDA fused GPU kernel，将 MoE dynamic expert sharing 中 coreset selection 的 12 个碎片化算子（softmax、top-k、masked reduction、final ranking 等）融合为仅 2 个 kernel，消除 kernel launch overhead 和冗余 HBM traffic。

术语是什么？
DES 论文（arXiv 2602.00879, 2026）为 Dynamic Expert Sharing 的 coreset selection 阶段开发了自定义 fused kernel。原始 PyTorch 实现需要 12 个独立 kernel（softmax → per-token top-k → mask → cross-sequence reduction → top-k ranking 等），每个 kernel 都需要从 HBM 读写中间结果。

Fused kernel 设计：
- **Primary kernel**：融合 per-token softmax、Top-K filtering 和 weighted expert vote accumulation。使用 register-level computation 保留中间结果（避免 HBM 往返），通过 atomic instructions（atomicAdd）更新全局 expert saliency scores。
- **Second kernel**：基于 threshold-governed ranking 执行 final expert masking。

在 NVIDIA B200 GPU 上实现 **6× speedup** over PyTorch baseline。

从kernel调度角度拆解术语：
```
// Primary Kernel: Per-token fused softmax + topk + vote accumulation
__global__ void fused_vote_kernel(
    float* router_logits,   // [N, M] input
    float* expert_votes,    // [M] output (initialized to 0)
    int N, int M, int K
) {
    int token_id = blockIdx.x;  // one block per token
    if (token_id >= N) return;
    
    // Stage 1: Per-token softmax (register-level, online algorithm)
    float row[M];
    float max_val = -INFINITY, sum_exp = 0.0f;
    float* input = router_logits + token_id * M;
    
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        row[i] = input[i];
        max_val = fmaxf(max_val, row[i]);
    }
    __syncthreads();
    max_val = warpReduceMax(max_val);  // warp-level reduction
    
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        row[i] = expf(row[i] - max_val);
        sum_exp += row[i];
    }
    __syncthreads();
    sum_exp = warpReduceSum(sum_exp);
    
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        row[i] /= sum_exp;  // normalized softmax
    __syncthreads();
    
    // Stage 2: Local top-K mask (register-level)
    // Find K-th largest via parallel selection
    float local_topk_mask[M] = {0};
    // ... topk threshold selection ...
    
    // Stage 3: Weighted vote accumulation (atomic to global)
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        if (local_topk_mask[i] > 0)
            atomicAdd(&expert_votes[i], row[i]);
}

// Second Kernel: Final ranking and masking
__global__ void expert_masking_kernel(
    float* expert_votes,    // [M]
    int* coreset_indices,   // [M_core]
    int M, int M_core
) {
    // Bitonic sort or radix select on expert_votes
    // Output top M_core expert indices
}
```

PyTorch baseline 的 12 kernel 链（内存视角）：
```
logits [N,M] → softmax kernel → gates [N,M] (HBM R/W)
gates → topk kernel → topk_val[N,K], topk_idx[N,K] (HBM R/W)
topk_val → mask kernel → masked_weights [N,M] (HBM R/W)
masked_weights → sum reduce kernel → votes [M] (HBM R/W)
votes → topk kernel → coreset [M_core] (HBM R/W)
coreset → mask/scatter → ... (subsequent routing kernels)
```

Fused 后的 2 kernel 链：
```
logits [N,M] → Primary fused kernel → votes [M] (1 HBM read of logits, atomic writes)
votes [M] → Second kernel → coreset [M_core] (core logic)
```
关键：register-level computation 消除 logits、gates、masked_weights 等中间张量的 HBM 往返（N×M 次 read/write 仅剩 1 次 read），kernel launch 从 12 次降至 2 次。

术语一般如何实现？如何使用？
- 硬件：在 NVIDIA B200 (Blackwell) 上实现，CUDA 13.1。高吞吐架构对算子碎片化更敏感（更受益于融合）
- 实现技术：warp-level reduction、parallel selection for topk、online softmax、atomicAdd for accumulation
- 性能特征：6× speedup over PyTorch baseline in coreset selection；end-to-end GPU kernel time 间接改善 8.2-14.3%
- 适用场景：任何需要跨 token 聚合 routing information 的 MoE optimization（不仅限于 DES）
- 论文未提供开源代码（arXiv 2602.00879）

涉及论文标题：
- DES: Dynamic Expert Selection for Efficient MoE Inference


## Computation-Communication Overlap via SM Control (SM控制的MoE计算-通信重叠)

术语解释
在 MoE 分布式推理中，通过限制 GEMM kernel 占用的 SM（Streaming Multiprocessor）数量，将部分 SM 资源留给 all2all 通信 kernel，使二者在不同 SM 上并行执行，实现计算与通信的 pipeline 重叠。

术语是什么？
NVIDIA GPU 上的 all2all 通信需要 SM 资源来驱动（如 NCCL kernel 在 SM 上执行数据打包/解包和网络操作）。默认情况下，GEMM kernel 倾向于占用所有可用 SM（如 H800 上的 132 SM），导致通信 kernel 只能在 GEMM 完成后才能调度。EPS-MoE 的关键发现：对 GroupGemm，在输入 size 达到一定阈值后，减少其占用的 SM 数不会显著影响计算效率（图5c），因此可以限制 GEMM 的 SM 数，留出 SM 给通信 kernel。

从kernel调度角度拆解术语：

```
=== SM 分区策略 (H800, 132 SMs) ===

# 无重叠（baseline）：
CUDA Stream 0: [GroupGemm 132 SMs] → [all2all 132 SMs]
总时间 = T_geMM + T_comm

# 有 SM 控制（EPS-MoE）：
CUDA Stream 0: [GroupGemm 116 SMs]  
CUDA Stream 1: [all2all 16 SMs]
              └── 并行执行 ──┘
总时间 ≈ max(T_geMM' + Δ, T_comm' + Δ), Δ≈0

# 最佳 SM 配置搜索（表6, H800）：
for gemm_sm in range(92, 133, 8):
    for comm_sm in range(1, 133-gemm_sm):
        # 配置 GEMM kernel SM 数
        # 使用 CUDA stream 实现并行
        GEMM_kernel<<<grid, block, shared_mem, stream_A>>>(..., sm_limit=gemm_sm)
        all2all_kernel<<<grid, block, shared_mem, stream_B>>>(..., sm_limit=comm_sm)
        latency = measure_concurrent_execution()
# 最佳配置：GEMM 116 SM + 通信 16 SM（H800 132 SM total）
# GEMM 计算吞吐损失 < 3%，但通信完全隐藏
```

术语一般如何实现？如何使用？
- 基于 CUDA Stream 实现 GEMM 和通信 kernel 的并发提交
- 通过 NVIDIA Nsight Systems 分析 SM 利用率和 kernel 重叠效率
- EPS-MoE 实验表明：通信 kernel 仅需 10-20 SM 即可跑满 NVLink 带宽，GEMM 在 116 SM 时效率仅比 132 SM 降低 <3%
- 与 FP8 通信正交：FP8 减少通信量，SM 控制减少等待时间，两者结合效果最佳
- H800 上最佳配置：GEMM 116 SM + 通信 16 SM

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference


## Horizontal Split for MoE Pipeline (MoE Pipeline的水平切分)

术语解释
在 MoE Expert Pipeline 调度中，对输入张量按行（token 维度）切分，同时将 MoE 权重按 expert 切分。相对于传统 TP 的垂直切分（按列切分输入+按列切分权重），水平切分避免了参数矩阵的重复内存 I/O，并支持将 GroupGemm 退化为 DenseGemm 以获得更高计算效率。

术语是什么？
MoE 模型的 Expert Pipeline 调度需要将输入数据划分为多个 pipeline stage。两种切分方式：
- **垂直切分（Vertical Split）**：按列切分输入张量，权重也按列切分。每个 stage 需要加载完整权重矩阵，导致重复 I/O。
- **水平切分（Horizontal Split）**：按行切分输入张量（按 token 分组），权重按 expert 切分。每个 stage 只加载对应专家组的权重，利用 MoE 稀疏激活避免重复 I/O。

从kernel调度角度拆解术语：

```
# === 垂直切分 (类似 TP) ===
# 输入 [m, K]，权重 [E*K, N] 完整加载
for pipe_i in range(N):
    x_slice = input[:, pipe_i*K//N : (pipe_i+1)*K//N]  # [m, K/N]
    w_subset = weights[pipe_i*K//N : (pipe_i+1)*K//N, :]
    for expert_j in range(E):  # 仍需 per-expert I/O
        y_j += x_j @ w_subset_j  # GroupGemm, group=E
# I/O: V_vertical = m*P0 + E*W + N*m*P1

# === 水平切分 (EPS-MoE) ===
for pipe_i in range(N):
    tokens_i = tokens_for_experts[pipe_i*E//N : (pipe_i+1)*E//N]  # [m_i, K]
    W_i = expert_weights[pipe_i*E//N : (pipe_i+1)*E//N]  # [(E/N)*K, N]
    # Load-aware GEMM choice
    if m_i >= 4096:
        y_i = DenseGemm(tokens_i, W_i)  # cublas
    else:
        y_i = GroupGemm(tokens_i, W_i, groups=E/N)
# I/O: V_horizontal = m*P0 + E*W + m*P1  (无重复参数I/O)

# 当 N=E 时: GroupGemm 退化为 DenseGemm
# 计算时间比较:
# T_vertical = E*C / R(FLOPS | group=E)
# T_horizontal = E*C / R(FLOPS | group=E/N)
```

术语一般如何实现？如何使用？
- 基于 vLLM 的 MoE FFN 前向传播路径修改输入切分逻辑
- token routing 后按 expert 分组 token，再按 pipeline 数细分为子组
- 权重在初始化时即按 expert 维度切分存储，避免运行时切分开销
- 与 all-to-all 通信流水线配合：第 i 组 all2all 通信与第 i-1 组 GEMM 计算重叠
- 适用于 top-k 较大或专家数较多的 MoE 模型

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

## Incremental Prefilling via KV Cache Reuse (基于 KV Cache 复用的增量 Prefilling)

术语是什么？
一种 kernel 级优化技术，允许在已完成的 prefix prefill 基础上追加新 token chunk 而非重新计算全部。核心原理：Prefix 的 KV blocks 在首次 prefill 后驻留 HBM，后续追加 token 的 KV 计算仅需对新 token 执行 attention 投影（Q/K/V），prefix 部分的 K/V 直接从 HBM 读取。在 Faster-MoA 中，后继 agent 的前缀段先被完整 prefill → KV 驻留 HBM；随后前驱 agent 每产出一个 decode chunk，chunk 被追加到前缀之后 → 发出增量 /prefill_only 更新 → FlashAttention 仅计算新 chunk 的 KV 并追加到 KV cache → prefix 部分的 KV 从 HBM 读取（~100% cache hit rate）。

从kernel调度角度拆解：
增量 prefill 的注意力计算流程（以追加 chunk 为例）：

```
假设已有:
  prefix KV: K_{1..P}, V_{1..P} ∈ R^{P × d_head}   // P 个已 prefilled tokens 的 KV
  HBM 地址: kv_cache_addr_prefix

增量 prefill:

Step 1: 新 chunk input tokens → QKV projection
  x_new ∈ R^{C × d_model}                           // C 个新 token
  Q_new = x_new @ W_Q    ∈ R^{C × d_head}
  K_new = x_new @ W_K    ∈ R^{C × d_head}
  V_new = x_new @ W_V    ∈ R^{C × d_head}

Step 2: 从 HBM 加载 prefix KV (内存复用)
  K_prefix = load_HBM(kv_cache_addr_prefix)          // [P × d_head]
  V_prefix = load_HBM(kv_cache_addr_prefix + offset) // [P × d_head]

Step 3: 拼接
  K_full = concat([K_prefix, K_new])  // [(P+C) × d_head]
  V_full = concat([V_prefix, V_new])

Step 4: Attention 计算 (标准 FlashAttention, P+C < 全 prompt 长度)
  S = Q_new @ K_full^T  / sqrt(d_head)               // [C × (P+C)]
  A = softmax(S, dim=-1, causal_mask=True)            // causal mask 选填
  O = A @ V_full                                       // [C × d_head]

Step 5: 写入新增 KV 到 HBM
  store_HBM(kv_cache_addr_prefix + P*d_head*sizeof, K_new)
  store_HBM(kv_cache_addr_prefix + P*d_head*sizeof + offset, V_new)
  // 完整 KV cache 现在覆盖 [P+C] tokens
```

关键优化：(1) prefix KV 无需重计算，直接从 HBM 读取——O(P·d) 内存带宽 vs O(P·d^2) 重新计算的差距；(2) 新增计算量 ∝ C·(P+C)·d_head，其中 C≪P，即大部分 FLOPs 为 prefix 内存读取所替代；(3) 增量 prefill 是 memory-bandwidth-bound 而非 compute-bound，高效利用 HBM 带宽。

在 Faster-MoA 中的使用场景：Shell Router 每收到 APC 中一个 chunk，调用一次增量 prefill，chunk size 约 16-64 tokens。由于 prefix 部分已在 HBM 中且被上一次 prefill "预热"（近 100% cache hit），每次增量 prefill 延迟约 ~1-2ms（取决于 chunk size 和 HBM 带宽），远小于完整重新 prefill。

术语一般如何实现？如何使用？
- 基于 FlashAttention 或 PyTorch SDPA 实现：将 prefix KV 传入作为额外的 key_value 参数
- 在 SGLang 中通过 /prefill_only API + 已缓存 KV blocks 的 token 追加机制实现
- 要求 PE 维护 prefix KV blocks 在 HBM，不因 /prefill_only 无输出而被回收
- chunk size 权衡：大 chunk → 少请求数但单次延迟高；小 chunk → 多请求数但更高重叠度
- 也适用于其他需要流式/增量构建 prompt 的场景（如 multi-turn dialogue 的 context 积累）

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

## All-gather Dispatch for Expert Parallelism

术语解释
METRO 提出的 Expert Parallelism 通信模式变体，将 MoE layer dispatch 阶段的 all-to-all 通信替换为 all-gather，使每个 GPU 获得全局 token 集合，从而能在每个 GPU 上独立计算全局 top-k 和 token routing 决策。这解决了传统 all-to-all dispatch 下各 GPU 仅有本地 top-k 信息、无法做出全局最优 routing 决策的问题。

术语是什么？
传统 EP dispatch 流程：每个 GPU 对本地 tokens 计算 top-k → all-to-all 将 tokens 发送到对应 expert 所在 GPU。各 GPU 仅知本地 top-k，不知其他 GPU 的 expert 选择情况。METRO all-gather dispatch：每个 GPU 将本地 tokens all-gather 到所有 GPU → 每个 GPU 现在持有全局所有 tokens → 在全局 token 集上计算 top-k → 每个 GPU 独立获得完整的全局 T[1..N]（每个 expert 的全局 token 计数）→ 执行 METRO routing algorithm → 仅计算分配给本 GPU 的 experts → all-to-all combine。

从kernel调度角度拆解术语：
All-gather dispatch 的 kernel 级执行序列：

```
=== METRO All-gather Dispatch（8 GPUs, 32 tokens/GPU）===

// 原 all-to-all dispatch:
// Step A1: 各 GPU local top-k → T_local[1..N]
// Step A2: All-to-all (tokens dispatch) → 每个 GPU 收到目标 expert 的 tokens
// Step A3: FFN compute
// Step A4: All-to-all (results combine)
// 问题: 各 GPU 只有 T_local[1..N]，无法做全局 routing 优化

// METRO all-gather dispatch:
// Step M1: All-gather tokens
//   数据量: 32 tokens/GPU × 8 GPUs × hidden_dim × fp16
//          = 32 × 8 × 1536 × 2 bytes ≈ 768KB → 每 GPU 收到 6MB (after all-gather)
//   NCCL kernel: all-gather 在 NVLink 上 ~3μs bandwidth + ~100μs launch
//   对比: all-to-all 256KB/GPU → ~400ns bandwidth + ~100μs launch
//   差异: bandwidth 增加 ~2.6μs，远低于 NCCL launch overhead

// Step M2: Global Top-K (CUDA kernel)
//   for each GPU g (并行):
//     对全局 256 tokens 计算 router logits
//     for each token t:
//       gate_logits = h[t] @ W_gate
//       probs = softmax(gate_logits)
//       top_k = TopK(probs, K)
//       更新 T[1..N] (每个 expert 的全局 token 计数)
//   开销: 17→20μs (local) vs 原 17→19μs, +3μs max

// Step M3: METRO Routing (CUDA kernel, 单 SM)
//   执行 Algorithm 1: greedy expert-to-GPU assignment
//   见 METRO greedy algorithm entry
//   开销: 17→26μs

// Step M4: FFN Compute
//   仅计算分配给本 GPU 的 activated experts
//   开销: 230→311μs (varies with replication, ~81μs 减少 vs EPLB)

// Step M5: All-to-all Combine
//   同原流程，将 expert outputs 返回各 token 原 GPU
//   开销: 与 EPLB 基线相同
```

术语一般如何实现？如何使用？
- All-gather dispatch 是 METRO 的关键使能技术——没有全局 top-k 信息 T[1..N]，就无法做出最小化 activated experts 的 routing 决策
- 开销分析：在 memory-bound decode 小 batch 下，NVLink latency 主导通信开销（~100μs NCCL launch），bandwidth 差异（all-gather 2MB/GPU vs all-to-all 256KB/GPU = ~2.7μs on 600 GB/s NVLink）可忽略
- 适用条件：decode phase（memory-bound, small batch）；prefill phase（compute-bound, large batch）继续使用 all-to-all + EPLB token routing，因为 prefill 下 bandwidth 开销会放大
- 冗余 top-k 计算：all-gather 后每个 GPU 在全局 token 集上计算 top-k，产生冗余计算（8 GPUs 各算一次），但 top-k 计算量极低（<5% layer time），冗余带来的 overhead <1%

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---

## CUDA Graph for LLM Inference

术语解释
CUDA Graph 是 NVIDIA CUDA 提供的机制，将一系列 CUDA kernel launches 和 memory operations 预录制为一个 graph，后续通过单次 launch 重放，消除重复 kernel launch overhead。在 LLM 推理中，decode 阶段每 step 执行相同的 kernel 序列（仅数据不同），非常适合 CUDA Graph 优化。

术语是什么？
传统 CUDA 编程中，每个 kernel launch 都需 CPU 向 GPU 提交 work，kernel launch overhead 在 decode 阶段（每 step 的计算量非常小）占比显著。CUDA Graph 将一次 decode step 的全过程——attention kernel、top-k、all-to-all dispatch、expert FFN kernels、all-to-all combine 等——录制为一个 graph，后续 decode steps 通过 `cudaGraphLaunch()` 单次提交整个 graph，大幅减少 CPU-GPU 同步和 kernel launch 开销。vLLM 的 compilation framework 支持将 decode phase 编译为 CUDA Graphs。

从kernel调度角度拆解术语：
CUDA Graph 在 METRO vLLM 集成中的应用：

```
=== vLLM CUDA Graph Compilation for Decode ===

编译阶段 (one-time):
  for batch_size in [1, 2, 4, 8, 16, 32]:  // power-of-two
    cudaStreamBeginCapture(stream)
    // 录制以下 kernel 序列:
    Attention_kernel(batch_size, ...)
    AllGather_kernel(tokens)
    TopK_kernel(all_tokens)
    METRO_Routing_kernel(N, G, A, T)    // 单 SM, Algorithm 1
    for each activated expert:
      FFN_GEMM_kernel(expert_weight, tokens)
    AllToAll_Combine_kernel(outputs)
    cudaStreamEndCapture(stream, &graph)
    cudaGraphInstantiate(&graph_exec[batch_size], graph)
    
  // 存储 graph_exec[1,2,4,8,16,32] 供运行时使用

运行时 Decode (each step):
  batch_size = min(next_power_of_two(num_ready_tokens), 32)
  if num_ready_tokens != power_of_two:
    pad_tokens_to(batch_size)  // padding to reuse graph
  update_input_pointers(graph_exec[batch_size])
  cudaGraphLaunch(graph_exec[batch_size], stream)
  // 单次 launch 执行所有 kernel，无中间 CPU-GPU 同步
```

术语一般如何实现？如何使用？
- CUDA Graph 适用于 kernel 序列固定、仅输入数据变化的工作负载——LLM decode 是典型场景
- 限制：(a) 图结构编译后不可变——需要预编译多个 batch size 版本；(b) 不支持动态控制流（kernel 内部的分支可以，但 kernel 选择和 kernel 数量不可变）；(c) 内存地址在录制时固定——需在 relaunch 前更新指针
- vLLM 的 CUDA Graph 集成：预编译 power-of-two batch sizes 的 graph，非 power-of-two 通过 padding 复用最近更大的 graph
- CUDA Graph 不能替代所有 kernel launch overhead——仍存在首次 graph launch 的初始化开销
- METRO 将 routing kernel 嵌入 CUDA Graph 后，Decode 阶段无额外的 kernel launch overhead

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

## Zero-Copy Memory Sharing via Ascend IPC

术语是什么？

Zero-Copy Memory Sharing via Ascend IPC 是 ElasticMoE 在 Ascend NPU 上实现的跨进程零拷贝张量共享机制。它允许两个独立进程（HMM 守护进程和 IMM 推理进程）引用同一块 NPU HBM 物理内存，而无需实际拷贝数据。核心流程：(1) HMM 使用 `IpcSafeAllocator` 分配 IPC 兼容的 HBM 物理内存；(2) 通过 `rtIpcSetMemoryName()` 为内存块注册唯一名称；(3) 通过 `rtSetIpcMemPid()` 将目标进程 PID 加入访问白名单；(4) 通过 ZMQ/UNIX domain socket 将内存句柄名称传递给目标进程；(5) 目标进程通过 `rtIpcOpenMemory()` 导入物理内存指针；(6) 通过 `torch::from_blob()` 将裸指针封装为 PyTorch tensor。

从 kernel 调度角度拆解术语：

```
Zero-Copy 操作伪代码：

// 发送端 (HMM, Process A)
tensor = ipc_safe_allocator.allocate(shape, dtype)  // aclrtMalloc + IPC flag
handle_name = "model_layer_0_attention"
rtIpcSetMemoryName(tensor.data_ptr(), handle_name)
rtSetIpcMemPid(target_pid)  // 白名单 IMM 进程
send_over_socket(target_socket, handle_name, shape, dtype, stride)

// 接收端 (IMM, Process B)
handle_name, shape, dtype, stride = recv_from_socket(source_socket)
physical_ptr = rtIpcOpenMemory(handle_name)
tensor = torch::from_blob(physical_ptr, shape, dtype, stride)
// tensor 现在指向与 Process A 完全相同的物理内存
// 读/写操作直接在 HBM 上进行，无拷贝
```

与 P2P copy 的区别：zero-copy 在两个进程引用同一 NPU 时使用（共享 NPU 的场景），P2P copy 在数据需要跨越不同 NPU 时使用。Zero-copy 速度远快于 P2P copy（无实际数据传输）。

术语一般如何实现？如何使用？

基于华为 Ascend CANN IPC API 实现。在 CUDA 生态中等效为 `cudaIpcGetMemHandle` + `cudaIpcOpenMemHandle`。在 ElasticMoE 中用于共享 attention 权重、KV cache 和 expert 权重。Ablation 表明禁用 ZeroCopy 后 downtime 从 0 升至 67.40s（scale-up 期间无法共享权重和 KV cache）。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

## Peer-to-Peer Weight Transfer via HCCL (p2p-copy)

术语是什么？

p2p-copy 是 ElasticMoE 的跨 NPU 高速 P2P 张量传输原语，在缩放时用于将模型权重从已有 NPU 传输到新增 NPU。与从磁盘加载（最慢链路）不同，p2p-copy 通过 Ascend HCCL 集合通信库经 Unified Bus 或 RDMA 链路直接进行 device-to-device 传输，绕过 host memory，比磁盘 I/O 快约一个数量级。核心 API：HCCL `isend`/`irecv`/`broadcast` + CANN `aclrtMemcpyAsync`。

从 kernel 调度角度拆解术语：

```
p2p-copy 操作伪代码：

// 初始化（一次性）
init_process_group(npus=all_devices, backend="hccl")

// 缩放时 P2P 传输
void p2p_copy_weight(src_npu, dst_npu, tensor_name, partition):
    stream = aclrtCreateStream()  // 可选独立 stream 避免阻塞计算
    src_tensor = hmm.get_tensor(tensor_name, partition, src_npu)
    dst_tensor = aclrtMalloc(shape, dtype, dst_npu)
    aclrtMemcpyAsync(dst_tensor, src_tensor, size,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream)
    // 或使用 HCCL: isend/irecv 异步 P2P
    aclrtSynchronizeStream(stream)
```

传输路径：NPU A HBM → Ascend Unified Bus (intra-node) 或 RDMA (cross-node) → NPU B HBM。整个链路不经过 CPU host memory。

术语一般如何实现？如何使用？

基于 HCCL (Huawei Collective Communication Library) + CANN runtime API。Ablation 表明禁用 HCCL P2P 后 scale-up 延迟从 3.14s 升至 10.42s（约 3.3× 变慢，回退到磁盘 I/O）。在 CUDA 生态中等效使用 NCCL `ncclSend`/`ncclRecv` + `cudaMemcpyDeviceToDevice`。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

## Virtual Memory-based Expert Management (vpage-remap)

术语是什么？

vpage-remap 是 ElasticMoE 的虚拟内存抽象，用于在 MoE EP 重配置时高效管理 expert 权重的物理布局。Expert 权重在 NPU HBM 中以非连续物理页存储（每个 expert 对应独立的 `aclrtMallocPhysical` 物理页），但通过 `aclrtReserveMemAddress` 预留的连续虚拟地址范围映射为逻辑连续张量，满足 GEMM kernel 的对齐要求。当 EP 度变化需要重新分配 expert 时，仅更新虚拟→物理映射表，无需重新分配大缓冲区或全量拷贝 expert 权重。

从 kernel 调度角度拆解术语：

```
vpage-remap 操作伪代码：

// 初始化
void init_expert_vmem(npuid, expert_list):
    total_size = sum(e.size for e in expert_list)
    va_base = aclrtReserveMemAddress(total_size)  // 预留连续 VA
    offset = 0
    for expert in expert_list:
        phys_page = aclrtMallocPhysical(expert.size)  // 非连续物理页
        aclrtMapMem(va_base + offset, expert.size, phys_page)
        offset += expert.size
    // GEMM kernel: torch::from_blob(va_base) → 视为连续张量

// EP 重配置时 (如 EP=4→EP=6)
void remap_experts(npuid, new_expert_list):
    for expert in new_expert_list:
        if expert 新到达:
            received_page = p2p_copy(expert)  // HCCL 接收
            aclrtMapMem(va_base + offset, expert.size, received_page)
        // 旧映射保持活跃直到新实例接管
    // 接管后: aclrtUnmapMem(old_pages) + aclrtFreePhysical(old_pages)
```

术语一般如何实现？如何使用？

基于 Ascend ACL API：`aclrtMallocPhysical` / `aclrtReserveMemAddress` / `aclrtMapMem` / `aclrtUnmapMem` / `aclrtFreePhysical`。在 CUDA 生态中等效使用 CUDA Virtual Memory API：`cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemUnmap`。关键收益：避免 EP 重配置时重新分配大连续缓冲区和全量拷贝，降低 peak memory 和延迟。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

## IpcSafeAllocator

术语是什么？

IpcSafeAllocator 是 ElasticMoE 自定义的 PyTorch 内存分配器，覆盖 PyTorch 默认的 `TorchCachingAllocator`，确保所有模型权重分配使用 CANN 的 IPC 兼容内存标记。标准 PyTorch 分配器使用设备内存池，分配结果通常作为单一内存块管理，无法被跨进程 IPC 共享。IpcSafeAllocator 拦截 `torch.ones()`、`torch.empty()`、`torch.full()` 等核心分配函数，直接调用 IPC 兼容的 `aclrtMalloc`，使分配的张量可被 `rtIpcSetMemoryName`/`rtIpcOpenMemory` 跨进程共享。

从 kernel 调度角度拆解术语：

```
IpcSafeAllocator 伪代码：

class IpcSafeAllocator:
    def allocate(shape, dtype, device):
        size = shape.numel() * dtype.itemsize
        ptr = aclrtMalloc(size, ACL_MEM_MALLOC_HUGE_FIRST | IPC_COMPATIBLE_FLAG)
        return ptr
    def free(ptr):
        aclrtFree(ptr)

// 覆盖 PyTorch 默认分配
torch.ones = patched_ones   // 使用 IpcSafeAllocator 分配
torch.empty = patched_empty
torch.full = patched_full
```

禁用 IpcSafeAllocator 效果：scale-up 延迟 +29%（2.43→3.14s），peak memory +5.4%（275.2→290.0 GB）。延迟增加因回退到非 IPC 内存需额外拷贝步骤；内存增加因无法共享导致两份副本。

术语一般如何实现？如何使用？

基于 Ascend CANN `aclrtMalloc` + IPC 兼容 flag。PyBind11 暴露 C++ allocator。在 CUDA 生态中等效使用 `cudaMalloc` + `cudaIpcGetMemHandle` 兼容的分配策略。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

## HCCL (Huawei Collective Communication Library)

术语是什么？

HCCL (Huawei Collective Communication Library) 是华为 Ascend NPU 生态中的集合通信库，等效于 NVIDIA NCCL。提供 NPU 间高性能通信原语：all-reduce、all-gather、reduce-scatter、broadcast、all-to-all、P2P send/recv 等，通过 Ascend Unified Bus（节点内）或 RDMA（跨节点）链路实现高带宽低延迟数据传输。HCCL API 与 NCCL 概念兼容：`init_process_group` 建立通信域，`isend`/`irecv` 异步 P2P 传输，`broadcast` 广播。

从 kernel 调度角度拆解术语：

```
HCCL 在 ElasticMoE 中：

// 初始化通信域
hccl_init_process_group(world_size, rank, "hccl")
// 底层建立 Unified Bus / RDMA 链路

// P2P 传输 (p2p-copy 原语)
if rank == src:  hccl_isend(tensor, dst_rank, stream)
if rank == dst:  hccl_irecv(tensor, src_rank, stream)

// MoE all-to-all (推理时)
hccl_all_to_all(dispatch_tokens, receive_tokens, expert_counts)
```

术语一般如何实现？如何使用？

开源地址: https://gitee.com/ascend/cann-hccl。ElasticMoE 中 HMM 数据面通过 HCCL 执行 P2P 权重传输和集合通信。Ablation 表明禁用 HCCL（回退磁盘 I/O）导致 scale-up 延迟约 3.3× 变慢。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

---

## EDiT (Elastic Distributed Training)

术语解释
EDiT 是基于 Local SGD 的高效异步分布式训练方法，由 Ling 团队采用并贡献到 ICLR 2025 (Cheng et al.)。结合 layer-wise synchronization、pseudo gradient penalty 和 time-based synchronization 三个机制，解决传统同步 All-Reduce 训练在大规模集群中的 straggler 问题。

术语是什么？
传统同步分布式训练（All-Reduce）面临四个挑战：(1) 高通信开销；(2) straggler 节点瓶颈所有节点；(3) 弹性训练困难；(4) 对数据噪声敏感。EDiT 通过三个核心机制解决：

1. **Layer-wise synchronization**：逐层同步参数（非全局 barrier），prefetch 机制将下一层通信与当前层计算重叠。
2. **Pseudo gradient penalty**：(a) EMA 追踪 pseudo gradient 检测异常 worker 并排除；(b) 按 pseudo gradient norm 加权平均剩余 worker 的梯度；(c) 统一梯度裁剪防止发散。
3. **Time-based synchronization**：按时间阈值而非固定步数触发同步——快节点可执行更多局部更新，动态适应异构环境。

在理想环境中加速比可达 66.1%（baseline 速度降至 5.49e-2 step/s 时）。

从kernel调度角度拆解术语（EDiT 通信-计算 Overlap）：
```
Workers W_0..W_3, layers L_0..L_N:

for step in training:
    for layer in model:
        # Forward + Backward (各 worker 独立)
        hidden = layer.forward(hidden)
        grad = layer.backward(loss)

        # Layer-wise sync + prefetch
        if layer % sync_interval == 0:
            async_broadcast_layer_weights(layer)  # 非阻塞
            # 下一层 forward 与此层 broadcast 并行

    # Pseudo Gradient Penalty（同步时）
    pseudo_grad = (curr_params - prev_params) / lr
    if |norm(pseudo_grad) - EMA(norm)| > threshold:
        exclude_worker(i)                       # 异常排除
    fused = weighted_avg(valid_pseudo_grads)    # 加权平均
    fused = clip(fused, threshold)              # 梯度裁剪

    # Time-based sync
    if elapsed > sync_deadline: sync_all()
```

术语一般如何实现？如何使用？
- 论文链接：https://openreview.net/forum?id=xtlMtbVfWu
- Ling 团队在异构加速器集群上使用 EDiT 训练 300B MoE 模型
- 与 DLRover 框架集成
- 在 straggler 严重的异构环境中加速效果更显著

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---

## Pseudo Gradient Penalty

术语解释
EDiT 的核心组件，一种防止 Local SGD 训练中 loss spike 的梯度质量控制策略。通过三步级联：异常 worker 排除→加权平均→梯度裁剪。

术语是什么？
大规模异构集群中，部分 worker 可能因硬件故障、数据噪声等产生异常梯度。该策略使用 pseudo gradient（当前参数与上一同步步参数的差值除以学习率）替代真实梯度进行 worker 间质量评估：(1) EMA 追踪每个 worker 的 pseudo gradient norm，偏离阈值则排除该 worker；(2) 剩余 worker 按 pseudo gradient norm 反比加权（norm 越小权重越高，表示更稳定的更新）；(3) 超过全局阈值的 fused gradient 被裁剪。

术语一般如何实现？如何使用？
- 集成在 EDiT/DRLover 中
- pseudo gradient 比真实 gradient 更稳定（不受单步数据波动影响）
- EMA 阈值需根据训练规模调整

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---

## XPUTimer

术语解释
Ling 团队开发的轻量级分布式训练性能分析工具（Cui et al. 2025），集成在 DLRover 中。通过 selective tracing + async CUDA event management + data compression 实现 90% 内存节省（~1.5MB/加速器/step）和 O(1) 错误定位。

术语是什么？
传统 profiler（如 NVTX）全量监控产生海量日志难以在生产环境长期使用。XPUTimer 由两大组件构成：(1) Lightweight Selective Tracing——Python 层通过环境变量动态拦截 API，C++/CUDA 层框架无关 kernel 监控（cuBLAS、Flash Attention、NCCL、自定义算子），CUDA event pool 复用+异步后台线程日志+数据压缩仅记录时间戳和 kernel input layout。(2) Diagnostic Engine——multi-layered diagnostic（call stack analysis + in-kernel tracing）将错误定位复杂度从 O(logN) 降至 O(1)，结合宏观 metric (throughput) 和微观 metric (kernel launch latency distribution) 做细粒度异常检测。

从kernel调度角度拆解术语：
```
=== CUDA Event Pool（低开销核心）===
ev_pool = [cudaEventCreate() for _ in range(MAX)]

# 注入 NCCL kernel 后追踪
cudaEventRecord(ev_start, stream)
ncclAllReduce(...)
cudaEventRecord(ev_stop, stream)

# 后台线程异步检查
while training:
    if cudaEventQuery(ev_stop) == cudaSuccess:
        elapsed = cudaEventElapsedTime(ev_start, ev_stop)
        log({kernel: "ncclAllReduce", time: elapsed, layout: dims})

# 压缩: 仅记时间戳+kernel layout，不记完整 tensor
```

术语一般如何实现？如何使用？
- 集成在 DLRover (github.com/intelligent-machine-learning/dlrover)
- Python 层通过 TRACED_PYTHON_API 环境变量动态配置监控目标
- 生产环境长期运行 profiling 无显著性能影响

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

## CoC (Communication over Computation)

术语解释
CoC 是 ETR 论文中用于优化 Ascend NPU 上 MoE 训练的通信-计算融合策略。通过 MTE (Memory Transfer Engine) 的远程内存访问将串行的矩阵乘法和 All-to-All 集合通信融合为统一细粒度 kernel，实现流水线并行执行。

术语是什么？
传统 MoE 训练中 expert FFN 的 MatMul 和 All-to-All 通信串行执行，计算和通信间有显著 idle bubble。CoC 利用 MTE 的远程 DMA 能力，在 AI CORE 执行当前 micro-batch 的 MatMul 时，MTE 同时发起下一 micro-batch 的 token 数据传输，实现计算-通信 pipeline overlap。

从kernel调度角度拆解：
```
# 传统串行
for micro_batch in batches:
    output = MatMul(tokens, expert_weights)     # AI CORE
    dispatch_tokens(output)                      # HCCL All-to-All
    # Total = T_compute + T_comm (idle bubbles)

# CoC 流水线
stream_comp = create_stream(AI_CORE)
stream_comm = create_stream(MTE)
for i, mb in enumerate(batches):
    launch_matmul_on(stream_comp, tokens[i])
    if i+1 < len(batches):
        prefetch_on(stream_comm, tokens[i+1])   # MTE 预取
    synchronize()
    # Total ≈ max(T_compute, T_comm) (重叠)
```

术语一般如何实现？如何使用？
在 Ascend CANN 上通过 MTE 实现。MTE 是 Ascend NPU 内的专用数据传输引擎，支持类似 RDMA 的远程内存直接访问，不占用 AI CORE 计算资源。CANN 编译器将 MatMul 和 All-to-All 标记为可融合算子对生成融合 kernel。论文仅简要描述此优化，未开源实现细节。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Parallel Loading Dequantization Kernel (并行加载反量化 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parallel Loading Dequantization Kernel 是 D2MoE 为 MWQ 专门设计的 CUDA 反量化 kernel，解决动态 bit-width 权重推理时 GPU 存储层级间并行效率不足的问题。传统方法先完整反量化 INT→FP16 写入 global memory，再启动 GEMM 读取——中间 FP16 结果往返 HBM 浪费带宽。D2MoE 的 kernel 同时优化三个维度：

1. **加载并行 (Loading Parallelism)**：从 NVMe SSD 到 GPU global memory 的量化权重传输 (cudaMemcpyAsync) 与激活值从 global memory 到 L2 cache 的移动并发
2. **计算并行 (Computation Parallelism)**：反量化操作（CUDA cores 上拆解 packed INT + 位操作）与矩阵乘法（Tensor cores 上 GEMM）通过独立 CUDA stream 重叠
3. **去量化优化**：使用 Any-Precision LLM 的二进制位操作代替传统 bit-transpose，直接解包 packed 整数到 FP16

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 D2MoE-V1 (b₁=2, b_K=4, group_size=128) 的 MWQ 反量化为例：

```
=== Parallel Loading Dequantization Kernel (CUDA Pseudocode) ===

// 输入: MWQ 量化权重（已在 GPU Global Memory）
//   Q_W_b1:  packed INT2 [s, h/8] (每 byte 存 4 个 INT2)
//   z_b1:    INT8 [s, h/128]  (per-group zero points)
//   s_b1:    FP16 [s, h/128]  (per-group scales)
//   Q_W_bk:  packed INT1 [s, h/8] (binary residual, k=2..K, 每 byte 8 bit)
//   s_bk:    FP16 [s, h/128]  (per-group scales for residual)
// 输出: Ŵ_bK ∈ FP16 [s, h] (dequantized expert weight, 立即送入 GEMM)

__global__ void MWQ_Dequant_Kernel(
    uint8_t* Q_W_b1,        // packed INT2 base
    int8_t*   z_b1,          // per-group zero points
    half*     s_b1,          // per-group scales
    uint8_t* Q_W_bk_packed, // packed binary residuals (k=2..K)
    half*     s_bk,          // residual scales (k=2..K)
    half*     W_deq,         // output: dequantized FP16 weight
    int s, int h, int group_size = 128, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= s || col >= h) return;
    
    int group_id = col / group_size;
    
    // Step 1: 解包 INT2 base
    int byte_idx = (row * h + col) / 4;  // 4 elements per byte for INT2
    int bit_offset = (col % 4) * 2;
    uint8_t byte_val = Q_W_b1[byte_idx];
    int q_val_b1 = (byte_val >> bit_offset) & 0x03;  // extract 2 bits
    
    // Step 2: Asymmetric dequantization (INT2 → FP16)
    float w_acc = (float)(q_val_b1 - z_b1[group_id]) * __half2float(s_b1[group_id]);
    
    // Step 3: 叠加 binary residuals (k=2..K)
    for (int k = 2; k <= K; k++) {
        int residual_byte = (row * h + col) / 8;  // 8 elements per byte for 1-bit
        int bit_shift = col % 8;
        uint8_t residual_byte = Q_W_bk_packed[(k-2) * s * h/8 + residual_byte];
        // 提取 1 bit → 映射到 {+1, -1}
        int q_val_bk = ((residual_byte >> bit_shift) & 0x01) ? 1 : -1;
        w_acc += (float)q_val_bk * __half2float(s_bk[(k-2) * s * h/128 + group_id]);
    }
    
    W_deq[row * h + col] = __float2half(w_acc);  // → L2 cache → Tensor core GEMM
}

=== CUDA Stream Orchestration (Triton-level) ===

// Stream 1 (I/O): cudaMemcpyAsync, disk→global memory
// Stream 2 (Compute): dequantization + GEMM

// Triton 协调:
// 当 Stream 1 加载 batch N 的 Q_W 时
// Stream 2 同时执行 batch N-1 的 dequant + GEMM
// 这种双缓冲策略最大化 GPU 利用率

for batch_id in range(num_batches):
    io_event = cudaMemcpyAsync(Q_W_tensors[batch_id], disk_ptr, size, H2D, io_stream)
    if batch_id > 0:
        wait(comp_event[batch_id-1])  # 等前一批 GEMM 完成
        cudaEventRecord(comp_event[batch_id], comp_stream)
        MWQ_Dequant_Kernel<<<grid, block, 0, comp_stream>>>(...)
        cuBLAS_GEMM<<<..., comp_stream>>>(W_deq, activation, output)
    sync()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 基于 NVIDIA Ampere/Ada Lovelace 架构实现。Kernel 使用 CUDA C++ 编写，与 PyTorch 通过 `torch.utils.cpp_extension` 集成。位操作优化参考自 Any-Precision LLM (Park et al., ICML 2024)，用直接位掩码和移位操作代替传统 int→float bit-transpose 方法，处理速度显著提升。

dequantization kernel 开销分析（Figure 12）：
- 4 requests 时：计算开销 ~20.77%，延迟开销 ~18.56%
- 32 requests 时：计算开销 ~16.77%，延迟开销 ~5.3%（因 MWQ 嵌套结构使 base 反量化结果被更多 request 复用）
- 临时 FP16 中间内存立即释放，对峰值内存影响极小

适用于：端侧设备（RTX 3060/AGX Orin）上需要动态 bit-width 的 MoE 模型推理，KV Cache 量化的反量化 kernel（类似 KIVI 的 Q_MatMul，但 D2MoE 针对的是 expert 权重而非 KV Cache）。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving
