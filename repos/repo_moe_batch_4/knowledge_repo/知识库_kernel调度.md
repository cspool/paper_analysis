## Fast Encode / Fast Decode（MoE 稀疏编码/解码 GPU Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fast Encode 和 Fast Decode 是 TUTEL 为 MoE 层 dispatch（编码）和 combine（解码）阶段设计的 SIMT-efficient 稀疏 GPU kernel。传统 MoE 框架（GShard/Fairseq/DeepSpeed）使用稠密 einsum 实现 encode/decode（Figure 20a），时间复杂度 O(T·E·C_g·D)，包含大量零乘加运算和巨大的中间张量分配。TUTEL 将其替换为基于 warp-level 并行的稀疏实现（Figure 20b），时间复杂度降至 O(T·k·D)，其中 T·k = E·C_g，使得稀疏版本的复杂度仅为稠密版本的 1/T。实现基于三个精心设计的 CUDA kernel：K0（门控处理）、K1（稀疏编码/解码）、K2（布局变换）。

从kernel调度角度拆解：

Fast Encode/Decode 的 kernel 设计和调度流程：

```cuda
// === K0: Gate Processing Kernel (SIMT-efficient) ===
// 每个 warp 处理一个 token，沿 M 维度
__global__ void k0_gate_processing(
    float* gate_probs, int64_t* idxs, float* scores,
    const float* logits, int T, int E, int k) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int token_id = warp_id;
  
  if (token_id >= T) return;
  
  // Softmax (warp-level shuffle reduction)
  float max_val = -INFINITY;
  for (int e = threadIdx.x; e < E; e += 32)
    max_val = max(max_val, logits[token_id * E + e]);
  max_val = warpReduceMax(max_val);  // __shfl_xor_sync
  
  float sum = 0.0f;
  for (int e = threadIdx.x; e < E; e += 32) {
    float p = expf(logits[token_id * E + e] - max_val);
    gate_probs[token_id * E + e] = p;
    sum += p;
  }
  sum = warpReduceSum(sum);
  
  // Normalize (warp shuffle)
  for (int e = threadIdx.x; e < E; e += 32)
    gate_probs[token_id * E + e] /= sum;
  
  __syncwarp();
  
  // Top-K selection (warp-level, one lane only)
  if (threadIdx.x == 0) {
    // Simple partial sort for k << E
    topk_select(gate_probs + token_id * E, idxs + token_id * k, 
                scores + token_id * k, E, k);
  }
}

// === K1: Sparse Encode Kernel ===
// 每个 warp 处理一个 token，稀疏写入 dispatch_input
__global__ void k1_sparse_encode(
    float* dispatch_input,      // (E, C_g, M) output
    int* expert_counters,        // per-expert counter for capacity tracking
    const float* moe_input,      // (T, M) input features
    const int64_t* idxs,         // (T, k) selected expert indices
    const float* scores,         // (T, k) gating scores
    const int* locations,        // (T, k) 1D location within expert's capacity slot
    int T, int k, int C_g, int M, int E) {
  
  int token_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  if (token_id >= T) return;
  
  for (int ki = 0; ki < k; ki++) {
    int expert = idxs[token_id * k + ki];
    int loc = locations[token_id * k + ki];
    float score = scores[token_id * k + ki];
    
    // Sparse scatter: 仅 top-k 专家非零写入
    for (int m = lane_id; m < M; m += 32) {
      float val = score * moe_input[token_id * M + m];
      dispatch_input[(expert * C_g + loc) * M + m] = val;
    }
  }
}

// === K2: Layout Transform (Flexible A2A) ===
// 将 (E, C_g, D) → (E_g, C, D)，消除对 world_size 的依赖
__global__ void k2_flexible_layout(
    float* output, const float* input,
    int E, int C_g, int C, int W, int D, int E_g) {
  // Stride-copy with index remapping
  // input[e][cg][d] → output[eg][c=eg*C_g+?][d]
  // Inline, no intermediate buffer
}
```

关键优化技术：(1) Warp Shuffling——使用 `__shfl_xor_sync` 和 `__shfl_down_sync` 实现 warp 内 reduction，避免 shared memory 开销；(2) Blelloch Scan——用于 prefix-sum 计算每个 expert 的 capacity slot 偏移；(3) Half2 向量化——利用 `half2` 类型同时处理两个 half-precision 元素，double 内存带宽利用率；(4) 无额外数据拷贝——所有 reshape 操作 inline 完成。

术语一般如何实现？如何使用？

TUTEL 通过 CUDA C++ 实现 K0/K1/K2，在 PyTorch 中通过 `torch.autograd.Function` 封装为可微操作：`tutel.fast_encode(input, idxs, scores, capacity)` 和 `tutel.fast_decode(combined_output, idxs, scores)`。前向/反向均使用自定义 CUDA kernel（Figure 21）。与 Fairseq/DeepSpeed MoE 的 einsum dense 实现相比，GPU 内存节省 20%~90%（Table 5），kernel 时间大幅缩短（Figure 15）。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale

## 2DH (Two-Dimensional Hierarchical) All-to-All（二维层次化全交换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

2DH All-to-All 是 TUTEL 提出的新型 All-to-All 通信算法，解决标准 Linear All-to-All 在大规模 GPU 集群中因消息大小 S/n 过小而无法饱和网络带宽的问题。核心思想：将 All-to-All 分解为节点内（intra-node）和节点间（inter-node）两个层次，通过 stride memory copy 将非连续的小 chunks 聚合为大 chunks，在大规模下保持高链路利用率。2DH 算法包含 4 个 phase（Figure 17）：Phase 1 stride memcpy（对齐同目标本地 GPU 的 chunks）→ Phase 2 intra-node All-to-All（m GPUs 交换）→ Phase 3 stride memcpy（对齐同目标远程节点的 chunks）→ Phase 4 inter-node All-to-All（n/m nodes 交换）。

从kernel调度角度拆解：

2DH All-to-All 算法伪代码（Algorithm 2 in paper）：

```cuda
// 2DH All-to-All: 4-phase algorithm
procedure ALL2ALL_2DH(output, input, S, n, m):
    chunksize = S / n
    nnodes = n / m
    
    // === Phase 1: Stride Memcpy (intra-node alignment) ===
    // 重排使同一本地目标 GPU 的数据连续
    strideMemcpy(buffer, input, chunksize, 
                 row=m, col=nnodes)
    // input[i] → buffer[j], j = i%m * nnodes + i/m
    // 源: GPU0 向 GPU0...GPU_n-1 的 chunks 交错排列
    // 目标: 按本地目标 GPU 分组连续排列
    
    // === Phase 2: Intra-node All-to-All ===
    for g = 0; g < m; g++:
        loc = g * nnodes * chunksize
        peer = g + node_rank * m
        ncclSend(buffer[loc], nnodes * chunksize, peer)
        ncclRecv(output[loc], nnodes * chunksize, peer)
    // 节点内 m 个 GPU 交换 S/m bytes 数据
    // chunk 大小: nnodes * chunksize = S/m (不依赖 n!)
    
    // === Phase 3: Stride Memcpy (inter-node alignment) ===
    strideMemcpy(buffer, output, chunksize,
                 row=nnodes, col=m)
    // 重排使同一远程目标节点的数据连续
    
    // === Phase 4: Inter-node All-to-All ===
    for nid = 0; nid < nnodes; nid++:
        loc = nid * m * chunksize
        peer = local_rank + nid * m
        ncclSend(buffer[loc], m * chunksize, peer)
        ncclRecv(output[loc], m * chunksize, peer)
    // 节点间 n/m nodes 交换合并后的大 chunks
    // chunk 大小: m * chunksize = S/nnodes (也大于原始的 S/n!)
end procedure

// Stride Memory Copy
procedure STRIDEMEMCPY(output, input, chunksize, row, col):
    for i = 0; i < row * col; i++:
        j = i % row * col + i / col    // stride index transform
        output[j * chunksize : (j+1) * chunksize] = 
            input[i * chunksize : (i+1) * chunksize]
end procedure
```

关键性能特性：(1) Phase 1-3 的延迟仅取决于 S（总数据量），与 GPU 数 n 无关；(2) Phase 4 的消息大小为 m·chunksize = S·m/n = S/nnodes，比 Linear A2A 的 S/n 大 m 倍；(3) 避免 naive local aggregation 中 O(n/m) 次非连续内存访问问题（当 n=2048, m=8 时延迟从 ~600μs 降至常数级别）。

术语一般如何实现？如何使用？

基于 NCCL 的 `ncclSend`/`ncclRecv` P2P API 实现（Algorithm 2），通过在 nccl-tests 的 `alltoall_perf` benchmark 中集成和验证。额外通过 MSCCL DSL 描述 2DH 算法并编译优化，利用 LL128 协议在低延迟场景进一步提升效率。在 64~4096 GPU 上验证，小消息（1 MiB）和大消息（256 MiB）均有显著加速（Figure 18），且扩展到 4096 GPU 而 Linear All-to-All 在此规模下无法成功运行。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale

## Flexible All-to-All（灵活全交换布局变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Flexible All-to-All 是 TUTEL 对标准 NCCL All-to-All 的 layout 抽象优化。标准 All-to-All 将 tensor layout 从 (E, C_g, D) 变换为 (W, E_g, C_g, D)，其中每个 GPU 的 local capacity C_g 依赖于 world size W（C_g = C/W）。这导致后续 expert FFN 的 matrix multiplication 输入 shape 随 scale 变化，影响计算效率。Flexible All-to-All 将输出 layout 改为 (E_g, C, D)——C 为全局 token capacity（C = E_g × C_g × W / E'），不直接依赖 W。这保证了任意规模下每个 GPU 上 expert matmul 的输入 shape 一致，且更利于 GPU Tensor Core 的高效矩阵乘法。

从kernel调度角度拆解：

Flexible All-to-All 的 layout 变换与标准 A2A 对比：

```
# === Standard All-to-All Layout ===
输入: input[E, C_g, D]  # E experts, C_g local capacity, D hidden
输出: output[W, E_g, C_g, D]  # W GPUs维度引入
# 后续 expert ffn 输入: output[gpu_i] → (E_g, C_g, D)
# 问题: C_g = C/W → matmul shape 随 scale 变化

# === Flexible All-to-All Layout (TUTEL) ===
输入: input[E, C_g, D]
中间All-to-All通信: 标准的跨GPU token交换
输出: output[E_g, C, D]  # 直接合并为全局视角
# 后续 expert ffn 输入: (E_g, C, D)
# C 不依赖 W → matmul shape 恒定

# Inline layout transform (无额外copy):
# output[eg][c][d] = input_gpu[expert][local_c][d]
# 通过索引重映射 inline 完成，无中间buffer分配
```

效果（Figure 11）：Flexible A2A 的 expert computation throughput 高于标准 A2A layout，在 256 GPUs 时额外获得 1.24× 加速（Figure 14, curve 3→4）。

术语一般如何实现？如何使用？

在 TUTEL 中通过定制化的 NCCL All-to-All 封装实现，作为 MoE 层 dispatch/combine 的通信后端。用户透明使用——调用 TUTEL 的 MoE 层 API 时自动应用 Flexible A2A layout。无需额外配置或用户干预。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale

## MoE Adaptive Pipelining（MoE 自适应流水线 / Multi-stream Token Partition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Adaptive Pipelining 是 TUTEL 的运行时优化技术，通过将 MoE 层的 All-to-All 通信与 Expert FFN 计算在不同 CUDA stream 上重叠执行，最大化 GPU 利用率。与传统的 batch-splitting 或 pipeline parallelism（会放大 MoE dispatch 不均衡和破坏 Batch Prioritized Routing 正确性）不同，TUTEL 仅在 All-to-All 和 Expert 区间内沿 capacity 维度分区。每 iteration 通过查字典选择最优的流水线度 d∈{1,2,4,8} 和 All-to-All 算法 a∈{Linear,2DH}。

从kernel调度角度拆解：

Multi-stream Token Partition 的调度时序（以 d=2 为例）：

```python
# Adaptive Pipelining with degree d=2
def moe_layer_pipelined_forward(input, gate_output, d=2, algo='2DH'):
    # input: (E, C_g, D) — token dispatches per expert
    # Step 1: Partition along capacity dimension
    partitions = input.chunk(d, dim=1)  # 2 chunks of (E, C_g/2, D)
    
    streams = [torch.cuda.Stream() for _ in range(d)]
    all2all_outputs = [None] * d
    expert_outputs = [None] * d
    
    # Step 2: Pipeline dispatch across streams
    for i in range(d):
        with torch.cuda.stream(streams[i]):
            # A2A Dispatch (communication stream)
            all2all_outputs[i] = all_to_all(partitions[i], algo=algo)
            # → shape: (E_g, C/2, D) for Flexible A2A
    
    # Step 3: Wait for A2A dispatch, then Expert FFN
    for i in range(d):
        streams[i].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[i]):
            expert_outputs[i] = expert_ffn(all2all_outputs[i])
            # Expert compute on GPU compute stream
            # Stream i+1's A2A overlaps with Stream i's Expert FFN
    
    # Step 4: A2A Combine
    combined = [None] * d
    for i in range(d):
        with torch.cuda.stream(streams[i]):
            combined[i] = all_to_all(expert_outputs[i], algo=algo)
    
    # Step 5: Barrier and merge
    for s in streams:
        s.synchronize()
    output = torch.cat(combined, dim=1)  # (E_g, C, D) or (E, C_g, D)
    return output
```

时序图（Gantt）：
```
Time →     |---- A2A_dispatch_0 ----|         |-- A2A_combine_0 --|
           |    |-- A2A_dispatch_1 ----|       |    |-- A2A_combine_1 --|
           |    |    |-- Expert_FFN_0 --|      |    |    |-- Expert_FFN_1 --|
Stream 0:  [====A2A_dispatch_0====][====Expert_FFN_0====][====A2A_combine_0====]
Stream 1:       [====A2A_dispatch_1====][====Expert_FFN_1====][====A2A_combine_1====]
Compute:        |<-- A2A and FFN overlapped -->|
```

关键设计：(1) 仅对 All-to-All + Expert 区间分区，不影响 gating 和 MoE 层外操作；(2) 不破坏 Batch Prioritized Routing 的语义正确性；(3) 自定义 inline reshape 操作无需额外数据拷贝；(4) 最优度 d 通过预构建字典 O(1) 查找，搜索空间 {1,2,4,8}（度 > 8 几乎不改进 overlap 但增大 A2A 开销）。

术语一般如何实现？如何使用？

基于 PyTorch CUDA Stream API 实现多流调度，All-to-All 操作使用定制的可接受分片输入的通信 kernel。TUTEL 用户通过 MoE 层 API 自动启用自适应流水线，无需手动管理 stream。预构建字典在训练开始前通过少量 profiling（每 key 最多 (log_{1.5}⌈W/E⌉ + 2) × 4 × 2 trials）完成。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale

## MoE Adaptive Parallelism Switching（MoE 自适应并行切换 / Zero-Cost Parallelism Switch）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Adaptive Parallelism Switching 是 TUTEL 的核心机制，允许 MoE 训练在 DP（数据并行）和 EP+DP+MP（专家+数据+模型并行混合）之间**零成本**运行时切换。传统系统中切换并行策略需要：(1) 不同的张量分片布局；(2) 参数/优化器状态迁移（Figure 4）；(3) 框架接口变更。TUTEL 通过单一统一张量布局（基于 ZeRO-DP Stage-3 风格的分片）消除了这些开销——DP 和 EP+DP+MP 共享相同的 weight slicing 格式和 data layout，仅通过控制参数 r 来切换执行流。

从kernel调度角度拆解：

Adaptive Parallelism Switching 的 tensor layout 和调度逻辑：

```
# === 统一张量布局（Single Layout） ===
# 所有 parallel 策略共享此布局:
# - Weights: ZeRO-DP Stage-3 style sharding (每个 GPU 持有 1/W 的权重分片)
# - Data: 每个 GPU 仅持有本地 tokens
# - Optimizer states: 分片存储，与 weight sharding 对齐

# 控制参数 r 的含义:
# r = 0: DP (纯数据并行)
# r = 1: EP+DP (EP+DP, 等价于 EP+DP without MP)
# 1 < r < ceil(W/E): EP+DP+MP (混合)
# r = ceil(W/E): EP+MP (等价, group_size=1 消除 DP all-gather)

# === Switchable DP (r=0) Execution Flow ===
def moe_forward_dp(input, gate_output):
    # All-gather: 收集 W 个 GPU 的完整权重分片
    W_full = all_gather(W_local, group=range(W))  # comm: O(P)
    # 用完整权重计算
    expert_output = expert_ffn(W_full, local_tokens)  # 每个GPU用自己token算所有专家
    # Backward: Reduce-scatter gradients
    grad_W = reduce_scatter(grad_W_local, group=range(W))
    return expert_output

# === Switchable EP+DP+MP (r in [1, ceil(W/E)]) Execution Flow ===
def moe_forward_epdpmp(input, gate_output, r):
    group_size = ceil(W/E) / r  # DP group size
    groups = split_gpus(W, group_size)
    
    # Step 1: LOCAL_REPEAT — 复制 gating 结果 r 份
    gate_replicated = repeat(gate_output, r)  # (T*r, ...)
    
    # Step 2: 基于 replicated gating 进行 All-to-All Dispatch
    dispatched = all_to_all(encode(input, gate_replicated))
    
    # Step 3: Expert FFN —— MP 风格: 各 GPU 计算专家的一部分
    # 仅需要 DP group 内的 all-gather 来获取权重分片
    if group_size > 1:
        W_local_group = all_gather(W_local_shard, group=groups[my_group])
    expert_out = expert_ffn_partial(dispatched, W_local_group)
    
    # Step 4: All-to-All Combine
    combined = all_to_all(expert_out)
    
    # Step 5: LOCAL_SUM — 对 r 份 replica 求和
    output = reduce_sum(combined.reshape(r, T, ...), dim=0)
    
    return output
```

通信复杂度分析（Table 4）：DP 为 O(P)，EP+DP+MP 在 r ≥ W/E 时为 O(C_g · W/E)，在 r < W/E 时为 O(C_g · r + P/E/r)。通过 Ternary Search 在 r ∈ [1, ⌈W/E⌉-1] 中找到凸函数最小值，加上边界 case r=0 和 r=⌈W/E⌉。

术语一般如何实现？如何使用？

实现基于 PyTorch 分布式通信原语（all_gather, reduce_scatter），通过控制 r 值和 group 划分实现不同并行策略的切换。预构建字典 `⌊c/R⌋ → {r*, d*, a*}` 在训练前 profiling 完成（约 (log_{1.5}⌈W/E⌉ + 2) × 4 × 2 次 trial per key）。运行时每 iteration O(1) 查表确定最优 r。TUTEL 用户通过 MoE 层 API 的 `adaptive_r` 参数控制，或设为自适应模式自动选择。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale

## Pipelined Expert Processing (流水线专家处理 / Expert-level Compute-Communication Overlap)

术语是什么？
Pipelined Expert Processing 是 ES-MoE 提出的 MoE 训练优化技术，在 expert-level 粒度上重叠 GPU 计算与 CPU↔GPU 数据传输（expert 参数上传/下载）。当 training iteration 经过 gating network 后，tokens 需要在 GPU 之间交换（token permutation / all-to-all），ES-MoE 利用这个通信窗口异步启动第一个 expert 的 CPU→GPU 上传。后续 experts 顺序处理时，当前 expert 的 GPU kernel（FFN forward/backward）与下一个 expert 的 DMA 传输并行——形成 compute ↔ upload 的细粒度流水线。与传统的 layer-wise pipelining（等整个 layer 所有 experts 完成）不同，ES-MoE 的 expert-level pipelining 在单个 expert 完成后即触发后续操作。

从kernel调度角度拆解：
Pipelined Expert Processing 在一个 MoE layer forward pass 中的调度时序：

```python
# MoE Layer Forward Pass with Pipelined Expert Processing
# K=4 GPUs, num_experts=16 (4 experts per GPU after placement)

def moe_layer_pipelined_forward(tokens, gating_network):
    # Step 1: Gating (GPU kernel)
    gate_output = gating_network(tokens)
    expert_ids = argmax(gate_output)  # top-1
    
    # Step 2: Dynamic Placement (CPU, <2.69us)
    gpu_assignments = greedy_placement(expert_ids)
    # e.g., GPU_0 gets [E3, E7, E12, E1]
    
    # Step 3: Overlapped Permutation + 1st Expert Upload
    # Stream 0: All-to-All token exchange (NVLink, ~few ms)
    # Stream 1: async memcpy E3 (CPU→GPU, PCIe, ~few ms)
    # Both execute concurrently
    cuda_stream_0 = all_to_all_scatter(tokens, gpu_assignments)
    cuda_stream_1 = async_upload_expert(E3_params, CPU→GPU)
    synchronize_both_streams()
    
    # Step 4: PIPELINED Expert Processing
    experts_on_this_gpu = [E3, E7, E12, E1]  # ordered by placement
    for i, expert_e in enumerate(experts_on_this_gpu):
        # Compute expert_e on GPU
        output[expert_e] = expert_e.forward(input_tokens_e)
        # Meanwhile, upload next expert (if not last)
        if i+1 < len(experts_on_this_gpu):
            next_expert = experts_on_this_gpu[i+1]
            async_upload_expert(next_expert, CPU→GPU, stream=upload_stream)
            # upload overlaps with compute of current expert
    
    # Step 5: All-to-All Gather + Weighted Sum
    return combine_expert_outputs_inverse_alltoall(output, gate_output)
```

关键时序约束：expert upload time (TU) 必须 ≤ expert compute time (TC) 才能实现完美 overlap。当 expert 的 token count 很小时，TC 可能 < TU，导致 GPU stall（expert 等待下载完成）。ES-MoE 通过 expert pinning（固定 25% 最热门 experts 在 GPU 上）和 adaptive offloading 来缓解此问题。

术语一般如何实现？如何使用？
- 实现依赖 CUDA streams（计算流 + 拷贝流分离）和 `cudaMemcpyAsync`（非阻塞 DMA）
- 在 PyTorch 中通过 `torch.cuda.Stream` 管理独立的计算和通信流，使用 `stream.record_event()` + `event.wait()` 同步
- Expert 参数需在 CPU 端预加载到 pinned (DMA-able) memory 以实现最高 PCIe 传输带宽
- 对于 backward pass，对称的 pipeline 将 gradient download (GPU→CPU) 与下一 expert 的 backward kernel 重叠

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Dynamic Expert Placement (动态专家放置 / Greedy Expert Scheduling)

术语是什么？
Dynamic Expert Placement 是 ES-MoE 的核心调度算法，在每 iteration 根据 gating network 的输出（各 expert 的 token 数量）动态决定 expert→GPU 的映射，目标是最小化各 GPU 间处理时间的差异（minimize makespan）。这与传统 expert parallelism 中 experts 到 GPU 的**静态映射**根本不同——传统 EP 中 expert 固定在指定 GPU 上，token 负载不均直接导致 GPU 间计算量不均。ES-MoE 借助 expert offloading（experts 从 CPU 按需加载），使 placement 决策在每 iteration 可自由调整，从而将负载均衡决策与 token routing 决策解耦。

从kernel调度角度拆解：
Dynamic Expert Placement 算法（Greedy Scheduling, Graham 1969, 4/3-approximation）：

```python
def greedy_expert_placement(expert_loads, num_gpus):
    """
    Input:
      expert_loads: List[(expert_id, token_count)], 所有 expert 的 token 数量
      num_gpus: 可用 GPU 数量
    Output:
      gpu_assignments: Dict[gpu_id -> List[expert_id]], 每个 GPU 分配的 experts
    
    Algorithm: Greedy Minimum Makespan Scheduling
    Complexity: O(m * log n + m * log m) where m=#experts, n=#GPUs
    Actual runtime: <2.69 us (on CPU), negligible vs ms-scale expert compute
    """
    # Step 1: Model each expert's processing cost
    for expert_id, token_count in expert_loads:
        compute_time = token_count * FLOPs_per_token / GPU_TFLOPS
        upload_time = expert_param_size / PCIe_bandwidth
        processing_time[expert_id] = max(compute_time, upload_time)
    
    # Step 2: Sort experts by processing time (descending)
    sorted_experts = sort_by_processing_time(expert_loads, descending=True)
    
    # Step 3: Greedy assignment
    gpu_loads = [0] * num_gpus
    gpu_assignments = {gpu_id: [] for gpu_id in range(num_gpus)}
    
    for expert_id, _ in sorted_experts:
        # Assign to GPU with minimum accumulated load
        target_gpu = argmin(gpu_loads)
        gpu_assignments[target_gpu].append(expert_id)
        gpu_loads[target_gpu] += processing_time[expert_id]
    
    return gpu_assignments
```

效果（ES-MoE 论文 Figure 6）：MoE-M 64 experts，传统静态 placement 下 GPU 间 token 数差异达 102%（max/min ratio），动态 placement 将差异降至 15%。同时完全消除 zero-padding（不再需要统一 batch size 的 batched MM）。

术语一般如何实现？如何使用？
- 算法运行在 CPU 上（gating network 输出后、token permutation 前），每次 iteration 调用
- 处理时间建模需考虑 expert 参数传输时间 + 计算时间，传输时间取决于 PCIe 带宽和 expert 参数量
- 在 cloud VM 场景下，算法可扩展考虑异构 GPU 能力（不同 GPU model 有不同 TFLOPS）
- 局限性：当 expert offloading 不可用时（GPU only 模式），动态 placement 无法使用

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Expert-wise CPU Optimization (专家级CPU优化器 / Streaming CPU Optimizer)

术语是什么？
Expert-wise CPU Optimization 是 ES-MoE 提出的一种 optimizer 调度策略，将 CPU-based optimizer 的执行粒度从传统 layer 级或 model 级细化到 expert 级。在 MoE 训练中，当 expert 参数和 optimizer states 被 offload 到 CPU 后，CPU 上的 Adam optimizer 更新比 GPU 慢约 31×。传统方案（如 ZeRO-Offload）在整 layer 所有 expert 的 backward 完成后再触发 optimizer，或使用 delayed update（延迟梯度更新到下一 iteration，但引入 staleness 影响精度）。ES-MoE 的方案是：每个 expert 完成 backward pass 后**立即**触发其 CPU Adam step，而不等待同一 layer 的其他 experts——靠近 output 的 layers 的 expert optimizer 执行时间被靠近 input 的 layers 的 GPU backward 计算隐藏。

从kernel调度角度拆解：
Expert-wise CPU Optimization 与 GPU backward 的 overlap 时序（gantt 视角）：

```
Timeline (MoE-L, 24 layers, 4 GPUs, 16 experts):
================================================================================
Layer 23 (closest to output):
  GPU: [E0 backward][E1 bwd][E2 bwd][E3 bwd]
  CPU:              [E0 Adam][E1 Adam][E2 Adam][E3 Adam]  ← 与 GPU 其他 layers 重叠
                          ↓ CPU Adam 与 GPU Layer 22 backward 并行 ↓
Layer 22:
  GPU: [E0 backward][E1 bwd][E2 bwd][E3 bwd]
  CPU:              [E0 Adam][E1 Adam][E2 Adam][E3 Adam]
                          ↓ 与 GPU Layer 21 backward 并行 ↓
...
Layer 0 (closest to input):
  GPU: [E0 backward][E1 bwd][E2 bwd][E3 bwd]
  CPU:              [E0 Adam][E1 Adam][E2 Adam][E3 Adam]
================================================================================
End-to-end: CPU Adam latency ~hidden by GPU backward of preceding layers
```

伪代码：
```python
def expert_wise_cpu_optimization(layer_backward_graph):
    for layer in reversed(model.layers):  # backward: output → input
        for expert in layer.experts:
            # GPU: compute expert gradients
            expert_grads = expert.backward(activations)  # on GPU
            # Immediately trigger CPU Adam (async, non-blocking)
            async_cpu_adam_update(
                expert.params_cpu,      # offloaded params in CPU RAM
                expert_grads.cpu(),     # download gradients GPU→CPU
                expert.opt_states,      # Adam m, v on CPU
                lr, beta1, beta2
            )
            # GPU continues with next expert / next layer
            # CPU Adam runs concurrently
```

与 ZeRO-Offload 的关键区别：ZeRO-Offload 使用 **delayed update**（将 optimizer 延迟到下一 iteration 以隐藏延迟），但引入 staleness——参数更新使用的是上一 iteration 的梯度，可能影响模型精度。ES-MoE 的 expert-wise optimization 使用**当前 iteration 的梯度**即时更新，保持数学等价性。Ablation 结果显示，expert-wise optimizer overlapping 贡献了 8.7% 的 throughput 提升，且不影响精度。

术语一般如何实现？如何使用？
- 实现依赖 CPU 多线程 + GPU streams 的异步执行
- DeepSpeed CPU Adam 基于 AVX2/AVX-512 向量指令优化，单 socket 可达数十 GFLOPS
- 关键挑战：CPU optimizer 的吞吐量受 CPU 核心数限制（ES-MoE 使用 32-core EPYC 7543）
- 当 number of layers 较少或 experts 较少时，overlap 效果减弱（CPU optimizer 可能暴露在 critical path 上）

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Token Permutation in MoE (All-to-All Token Exchange / Token重排)

术语是什么？
Token Permutation 是 expert parallelism 中 token 在 GPU 间重新分配的操作阶段。由于各 GPU 上的 token 需根据 gating network 的路由决策被发送到持有对应 expert 的 GPU 进行计算，token permutation 通过 all-to-all 通信将 token 从"按原始 batch 排列"重排为"按目标 expert 排列"。在 ES-MoE 中，token permutation 阶段被利用为 expert 上传的 overlap 窗口——permutation 通信时间（NVLink all-to-all）通常能覆盖 single expert 的 CPU→GPU 上传时间（PCIe）。

从kernel调度角度拆解：
Token Permutation 的两阶段流程：

```python
# Phase 1: All-to-All Scatter (Dispatch)
# 将 tokens 按 expert routing 发送到目标 GPU
def token_dispatch(tokens, expert_ids, gpu_mapping):
    """
    tokens: (B*S, d_model) on local GPU
    expert_ids: (B*S,), which expert each token goes to
    gpu_mapping: Dict[expert_id -> gpu_id]
    Returns: tokens regrouped by destination GPU
    """
    # Build per-GPU token indices
    gpu_buckets = {gpu_id: [] for gpu_id in range(num_gpus)}
    for token_idx, expert_id in enumerate(expert_ids):
        target_gpu = gpu_mapping[expert_id]
        gpu_buckets[target_gpu].append(token_idx)
    
    # All-to-All: each GPU sends its tokens to target GPUs
    # Uses NCCL alltoallv or similar collective
    recv_tokens = all_to_all_scatter(tokens, gpu_buckets)
    return recv_tokens

# Phase 2: All-to-All Gather (Combine)
# 计算完成后，将 expert outputs 送回 token 原始所在 GPU
def token_combine(expert_outputs, token_origins):
    """
    expert_outputs: per-GPU expert computation results
    token_origins: which GPU each token came from (inverse of dispatch)
    """
    recv_outputs = all_to_all_gather(expert_outputs, token_origins)
    return recv_outputs
```

通信量分析：Token permutation 的通信量与 microbatch size × d_model 成正比，与 expert 数量无关。相比之下，expert 参数的上传/下载通信量与 expert 数量 × expert 参数量成正比。因此当 expert 数量很大时，expert 传输成为主要通信开销——这正是 ES-MoE 将 permutation 时间用于 overlap expert upload 的原因。

术语一般如何实现？如何使用？
- **NCCL alltoall**: NVIDIA 集合通信库，利用 NVLink/NVSwitch 高带宽 GPU-GPU 直连
- **Fairseq/Tutel**: 通过组通信（grouped all-to-all）按 expert parallelism group 交换 tokens
- **FasterMoE**: 提出 expert-centric 通信——交换 experts 而非 tokens（适用于大批量 GPU 场景）
- ES-MoE 将 token permutation 与 expert upload 重叠，利用 permutation 的通信时间窗口进行专家传输

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Dispatch Mask in MoE (MoE调度掩码 / Batched Matrix Multiplication的零填充掩码)

术语是什么？
Dispatch Mask 是 MoE 中 batched matrix multiplication 所需的大型稀疏张量，用于将 token 按 gating 决策重新排列为 per-expert 的连续 batch。其维度为 (num_padded_tokens, num_original_tokens)，是一个二值稀疏矩阵——mask[i,j] = 1 表示 padded token i 对应于 original token j。ES-MoE 指出，该 mask 的内存占用极大：训练 MoE-L batch_size=32, 1024 tokens/batch 时，mask 至少需要 48 GiB 显存。这是 ES-MoE 选择放弃 batched MM、改用 sequential expert computation 的关键动机之一。

从kernel调度角度拆解：
Dispatch Mask 的创建与使用：

```python
# Batched MM 方式（Fairseq, Tutel）:
def batched_moe_forward_with_mask(tokens, gate_output):
    # Step 1: 确定每个 expert 的 token 分配
    expert_token_counts = count_tokens_per_expert(gate_output)
    max_count = max(expert_token_counts)  # 由最热门 expert 决定
    
    # Step 2: 创建 Dispatch Mask
    # Shape: (num_experts * max_count, total_tokens)
    # 每行对应一个 padded expert slot
    dispatch_mask = torch.zeros(
        num_experts * max_count, total_tokens,
        dtype=torch.bool, device='cuda'
    )
    
    # Step 3: 填充 mask
    for token_idx, expert_id in enumerate(gate_output):
        expert_offset = expert_id * max_count
        slot = next_available_slot[expert_id]
        dispatch_mask[expert_offset + slot, token_idx] = True
        next_available_slot[expert_id] += 1
        # 未使用的 slots 保持为 0 → zero-padding
    
    # Step 4: Token 重排 (通过 sparse-dense matmul)
    padded_tokens = dispatch_mask @ tokens  # (E * max_count, d_model)
    
    # Step 5: Batched Expert FFN
    # Reshape + Batched GEMM: (E, max_count, d_model) × (E, d_model, d_ff)
    padded_tokens = padded_tokens.reshape(num_experts, max_count, d_model)
    expert_outputs = torch.bmm(padded_tokens, expert_weights)  # zero-padding 参与计算
    
    return unpermute(expert_outputs)
```

内存消耗详解：`dispatch_mask` 的内存 = `(num_experts * max_count * total_tokens) bits`。MOE-L, batch=32, 1024 tokens/batch, 32 experts → max_count 可能达 few hundred。Mask size = 32 × 300 × 32768 bits ≈ 39 MB... 等等，论文说 "48 GiB"。让我重新看论文：论文说的是 "training MoE-L with a batch size of 32 and 1024 tokens per batch requires at least 48 GiB for the mask"。这应该是指每 device batch size 32 × 1024 tokens per batch = 32768 tokens。如果 `num_experts=128`, `max_count=1024`，则 mask 大小为 `128*1024*32768*4_bytes (int32)` ≈ 17 GB... 具体实现细节论文未完全展开。但核心观点是：dispatch mask 是显存瓶颈，ES-MoE 通过顺序处理避免。

术语一般如何实现？如何使用？
- Fairseq GShard: 通过 `torch.sparse.mm` 或等效的 index-based scatter/gather 实现 token 重排
- Tutel: `sparse_coo_tensor` + 优化的 CUDA kernel 执行 dispatch/combine
- MegaBlocks: 使用 block-sparse matmul 替代传统 dispatch mask，减少 0 值块的内存和计算
- ES-MoE: 完全抛弃 dispatch mask，改用 per-expert 顺序计算——逐个 expert 独立处理其 tokens

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training



术语是什么？
Bit-packed Encoding 是一种利用 Bfloat16 浮点格式中 underutilized exponent bits 存储额外 metadata 的编码技术。PuzzleMoE 观察到 MoE 模型的 expert weights 在 Bfloat16 下 exponent 值集中在 [112, 128] 的窄范围（仅需 5 bits 编码 32 个值），而 Bfloat16 标准分配 8 bits 给 exponent。通过将 exponent 整体减去 112 的 shift 操作（round-up 所有 <112 的值到 112），exponent 可映射到 [0, 31]，释放出 3 bits。这 3 个 bits 用于嵌入：(1) 2 bits 的 binary mask（每个 merged expert pair 中两个 expert 各 1 bit，标记该位置是否属于该 expert）；(2) 1 bit 的 sign（标记原始权重的符号）。

从kernel调度角度拆解术语：
Bfloat16 标准格式：| bit15: sign | bits14-7: exponent (8 bits) | bits6-0: mantissa (7 bits) |
Packed Bfloat16 格式：| bit15: sign_of_expert_1 | bit14: sign_of_expert_0 | bit13: mask_of_expert_1 | bit12: mask_of_expert_0 | bits11-7: shifted_exponent (5 bits) | bits6-0: mantissa |

**Bit-packing 流程：**
1. Input: W_merged（FP32/BF16 merged weight），M_i, M_j, S_i, S_j
2. For each element [p,q]:
   a. Extract raw BF16 fields from |W_merged[p,q]|
   b. exponent ← max(raw_exponent, 112) - 112 → fit in [0, 31]
   c. Pack: packed = (S_i << 15) | (S_j << 14) | (M_i << 13) | (M_j << 12) | (exponent << 7) | mantissa
3. Output: packed_BF16 tensor（视为标准 Bfloat16 tensor，PyTorch 可正常加载，解释为数值时会因 exponent shift 产生偏置，但仅通过自定义 kernel 使用）

**On-the-fly Decoding（CUDA kernel 内）：**
```
mask_bit ← (W ≫ (13 - expert_pos)) & 1
if mask_bit == 0: return 0
sign_bit ← (W ≫ (15 - expert_pos)) & 1
exp ← (W & 0x0F80) + (112 ≪ 7)  // 恢复原始 exponent
W_decoded ← (sign_bit ≪ 15) | exp | (W & 0x007F)
```

术语一般如何实现？如何使用？
- 前提条件：模型权重的 exponent 分布需集中在窄范围（已验证 Mixtral-8x7B, Deepseek-MoE, Qwen1.5-MoE, Qwen3-MoE）。Shift 操作对 perplexity 无影响（Mixtral PPL before=4.37, after=4.37）。
- Packed 数据仍为 16-bit——无需额外 metadata 存储，消除 CSR 等稀疏存储格式的 index overhead。
- 通用性：其他工作如 LEXI（Huffman coding of exponents）、Schrödinger's FP（delta exponent encoding）、Exponent Sharing（LUT-based）也利用了 Bfloat16 exponent 冗余，但 PuzzleMoE 是首个将 freed bits 用于 embedding mask/sign 并配合 custom GPU kernel 做 MoE 推理的。
- 限制：bit 预算受可用 free exponent bits 约束——k=3 合并需 5 bits（3 mask + 2 sign），超出 Bfloat16 的 3 个 free bits，因此不支持 >2-way 合并。

涉及论文标题：
- PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference

## On-the-fly GEMV Decoding CUDA Kernel（即时 GEMV 解码 CUDA 内核）

术语是什么？
On-the-fly GEMV Decoding CUDA Kernel 是 PuzzleMoE 设计的一个自定义 CUDA kernel，在 GPU 上执行矩阵-向量乘法（GEMV）的同时即时从 bit-packed Bfloat16 格式中解码出每个 weight 的实际值。核心思想：解码逻辑（bit shift + mask + exponent 恢复）是计算量极小的 in-place 操作，可以 piggyback 在 kernel 的 data-loading path 上，利用 warp-level scheduling 和 coalesced memory access 使解码延迟被 global memory 读取延迟（~200 cycles）完全隐藏。该 kernel 消除了在 GPU global memory 中创建独立 decoded weight matrix 的需求，避免额外的 memory allocation 和访存开销。

从kernel调度角度拆解术语：
**CUDA Thread Block 执行流程：**
```
__global__ void puzzle_gemv_kernel(
    half* X,          // input activation [B, d]
    uint16_t* W_packed, // packed weights [d, h]
    half* Y,          // output [B, h]
    int expert_pos)   // 0 or 1
{
    int row = blockIdx.x;  // which output dimension
    int tid = threadIdx.x;
    float acc = 0.0f;

    // Coalesced load of input
    for (int k = tid; k < d; k += blockDim.x) {
        half x_val = X[row * d + k];        // load from global memory (~200 cycles)
        uint16_t w_packed = W_packed[k * h + row]; // coalesced load

        // On-the-fly decode (<< 10 cycles, hidden by load latency)
        int mask_bit = (w_packed >> (13 - expert_pos)) & 1;
        if (mask_bit == 0) continue; // weight pruned for this expert

        int sign_bit = (w_packed >> (15 - expert_pos)) & 1;
        int exp = (w_packed & 0x0F80) + (112 << 7);
        uint16_t w_decoded = (sign_bit << 15) | exp | (w_packed & 0x007F);

        half w_val = __uint2half_rn(w_decoded);
        acc += __half2float(x_val) * __half2float(w_val);
    }

    // Warp-level reduction
    acc = warpReduceSum(acc);
    if (tid == 0) Y[row] = __float2half(acc);
}
```

**关键设计决策：**
1. **Decoding on data-load path**：w_packed 从 global memory 加载到寄存器后立即解码——decode 指令（3-4 条 bit ops）与 memory load 的延迟比约为 1:200，因此解码开销被访存完全隐藏。
2. **No materialized decoded matrix**：与传统方法（先解码整个矩阵到 memory 再执行 GEMM）相比，避免了 O(d×h) 的额外 memory 分配和访存。
3. **Zero-value skipping**：mask_bit=0 时直接跳过 FMA——虽然仍在 warp 内（warp divergence），但因 data-load 已发生（mask 在 register 中而非 global memory），整体收益仍为正。
4. **expert_pos parameter**：同一 merged weight 通过 expert_pos=0 和 expert_pos=1 分两次调用 kernel 即可获得两个 expert 的输出——mask/sign 的 bit position 由 expert_pos 动态决定。

术语一般如何实现？如何使用？
- 集成到 PyTorch 推理框架中作为 torch.autograd.Function 的 forward 实现，替换标准 torch.nn.functional.linear。
- Gate network 和 attention 部分使用标准 PyTorch 算子——仅 expert FFN 的 GEMV 使用自定义 kernel。
- 适用于 decode phase（单 token 生成，GEMV 而非 GEMM）——batch_size=1 的自回归解码。
- 也可为 prefill phase（多 token batch, GEMM）扩展为 tiled GEMM kernel，但论文主要 benchmark decode phase 加速。

涉及论文标题：
- PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference

## Tensor-Parallel Expert Loading（张量并行专家加载）

术语是什么？
Tensor-Parallel Expert Loading 是 MoEsaic 为支持大型 MoE 模型（单个 expert 参数 > GPU 显存）的多实例共享而扩展的 tensor parallelism 支持。在 Tensor Parallel (TP) 模式下，每个 expert 被 shard 到多个 GPU 上（如 4-way TP），每个 GPU 仅持有 expert 的一部分 shard。vLLM 原生支持初始模型的 TP 加载，但不支持——向已部署模型中动态添加新 expert 时的 TP 加载。MoEsaic 新增 Ray worker 机制，使新 expert 继承初始模型的 sharding 策略，并在 shard 级别执行去重。

从kernel调度角度拆解术语：
在 MoEsaic 的 4-way TP 配置下（8×A100 40GB），Tensor-Parallel Expert Loading 的 kernel 流程：

```
// Step 1: 初始模型 TP 加载（vLLM 原生）
// Mixtral-8x7B 被 4-way shard 到 GPU 0-3
for gpu in [0, 1, 2, 3]:
    load_model_shard(base_model, gpu_rank=gpu, world_size=4)
    // GPU gpu 持有每个 expert 的 1/4 shard

// Step 2: 新 model instance 的 TP Expert Loading（MoEsaic 扩展）
// 生成 4 个 Ray workers，每个绑定一个 GPU
ray_workers = [RayWorker(gpu=i) for i in range(4)]

// 每个 Ray worker 在对应 GPU 上执行：
for worker in ray_workers:
    // 新 expert 继承初始模型的 sharding 方式
    for new_expert in new_model_instance.experts:
        shard = extract_shard(new_expert, 
                              rank=worker.gpu_rank, 
                              world_size=4)
        // Step 3: Shard 级别的去重
        shard_hash = compute_128bit_hash(shard)  // 对 shard tensor 计算 hash
        
        if shard_hash in hash_dictionary:
            shard.reference(hash_dictionary[shard_hash])  // 共享已有 shard
        else:
            allocate_gpu_memory(shard, gpu=worker.gpu_rank)
            hash_dictionary[shard_hash] = shard

// Step 4: 推理时 fused gate 路由（跨 GPU 的 TP 执行）
// 每个 token 被路由到 merged expert 的 shard
// shard 在各自 GPU 上执行 partial FFN，通过 all-reduce 聚合结果
```

关键区别：
- **完整 Expert 去重 vs Shard 去重**：在单 GPU 模式下，去重是在完整 expert 级别进行（每 expert ~14GB for Mixtral-8x7B）。在 TP 模式下，去重是在 per-GPU shard 级别进行（每 expert shard ~3.5GB for 4-way TP）。
- **继承 Sharding 策略**：新 expert 必须严格继承初始模型的 sharding 方式——若初始模型是 4-way TP 按列切分（column-wise sharding），新 expert 也按同样方式切分。否则去重对象（shard）的语义不一致。
- **Ray Worker 并行加载**：每个 Ray worker 独立在绑定 GPU 上执行 shard 加载和 hash 计算。论文表 3（Table 3）显示 TP 模式下加载速度更快——多个 Ray worker 并行后，Mixtral-4x7B（4 GPUs）加载 4 models 需 135s，比单 GPU 的 Mixtral-4x1B 加载 4 models 需 110s 相对更快（考虑模型尺寸差异）。

术语一般如何实现？如何使用？
- 实现方式：vLLM 的 Ray-based distributed executor + MoEsaic 自定义 `load_weights()` 中的 shard-aware dedup 逻辑。
- 与 vLLM 原生 TP 的关系：vLLM 原生 TP 在初始化时一次性 shard 所有参数；MoEsaic 扩展使其支持增量添加新 expert shard 并在运行时去重。
- 必须性：大型 MoE 模型（Mixtral-8x7B, DeepSeek-V2/V3）因单 expert 参数过大（14GB+），单 GPU 无法容纳完整 expert + runtime state（KV cache），TP 是实际部署的前提。
- 去重粒度权衡：shard 级别 vs expert 级别——shard 级别去重更细粒度（即使完整 expert 不同，其部分 shard 可能相同），但增加了 hash 计算次数和 dictionary 条目数。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts

## Expert Parallelism (EP / 专家并行)

术语是什么？
Expert Parallelism (EP) 是一种 MoE 模型训练/推理中的分布式并行策略。将不同的 expert 的权重分布到不同的计算设备（GPU/加速器）上，每个设备持有若干完整 expert 的权重。Router 在运行时将 token 按 top-k routing 结果发送到持有对应 expert 的设备，计算完成后将结果传回。EP 的核心收益是每个设备只需存储和处理部分 expert，使得总参数量可超过单设备内存限制，同时各 expert 的 matmul 保持了较大的 kernel size。

从kernel调度角度拆解术语：
在 MoE 训练的一个 iteration 中，EP 的 kernel 调度流程为：

```
// 假设 4 个 GPU，每个持有 2 个 expert（共 8 experts, top_k=2）

// Step 1: Router (每个 GPU 独立执行)
logits = Router(local_tokens)                     // (B_local*S, E)
topk_vals, topk_idx = topk(softmax(logits), k=2)

// Step 2: All-to-All Dispatch (通信 kernel)
// 在每个 GPU 上：
expert_tokens = {}  // 按目标 device 分组
for token in local_tokens:
    for expert_id in topk_idx[token]:
        target_device = expert_to_device[expert_id]
        expert_tokens[target_device].append(token)
// All-to-All scatter: 将 token 发送到持有对应 expert 的设备

// Step 3: Expert Compute (计算 kernel)
// 收到来自各 GPU 的 token 后：
for expert_e in my_experts:
    if has_tokens_for(expert_e):
        output = SwiGLU_FFN(expert_e, tokens)
        
// Step 4: All-to-All Combine (通信 kernel)
// 将 expert 输出送回 token 原始所在设备

// Step 5: Token Reorder (reorder kernel)
// 将返回的输出按原始 token 顺序排列
Y = reorder_by_token_index(returned_outputs)
```

BTA 论文指出，在 wafer-scale 处理器上，EP 解决的是跨设备 expert 分布问题，但不能解决同一设备内 attention 与 expert 的 batch size 冲突。BTA 与 EP 是互补的：EP 跨晶圆/设备分布 expert，BTA 在同一设备内解耦 attention 和 expert 的 batch size。

术语一般如何实现？如何使用？
主流框架实现：
- Megatron-Core：通过 `moe_ep_size` 参数配置 EP 并行度，与 TP/DP/PP 混合使用。推荐 Mixtral-8x7B 在 64 GPU 上用 TP=1, EP=8, PP=4。
- DeepSpeed-MoE：基于 DeepSpeed 的 EP 实现，支持 expert 到 GPU 的灵活映射和 All-to-All 通信优化。
- Tutel：自适应 MoE 框架，支持动态 expert 分配和 load balancing。
- FineMoE/MoE-Infinity：使用 hash map 做 expert→GPU 映射，round-robin 确保 GPU 间 expert 数均衡。

主要挑战：(1) Load imbalance — 不同 expert 被选中的 token 数不均衡；(2) All-to-All 通信开销 — 尤其在跨节点场景；(3) 小 top_k 时每个 expert 的有效 batch 过小，计算密度低；(4) Checkpoint 侧——EP 将 expert 分布在不同 ranks 上，现有 baseline（如 Megatron-DeepSpeed）仅用 EP-Group-0 保存所有 expert checkpoint，造成 bottleneck rank 负载过高而其他 EP groups 闲置。MoC-System (ASPLOS '25) 的 Fully Sharded Checkpointing 将 expert checkpoint 按 expert 切分在所有 EP groups 间均分，消除此瓶颈（bottleneck workload 降低 22%-29%）。

涉及论文标题：
- Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
- ReXMoE Reusing Experts with Minimal Overhead in Mixture-of-Experts
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models
- Sparse Upcycling Training Mixture-of-Experts from Dense Checkpoints（TPU v4 上使用 expert partitioning 分布 32 experts；Base/Large 用 64 chips，XL 用 256 chips + 4-way model partitioning）
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
- Upcycling Large Language Models into Mixture of Experts

Sem-MoE 论文揭示了 EP 在 MoE **推理**场景中的严重通信瓶颈：即使在高带宽互联（>400GB/s）下，all-to-all 占 DeepSeek-V2-Lite MoE layer forward latency 的 59.2%。Sem-MoE 通过 semantic-aware model-data collaborative scheduling 提升 Local Activation Rate (LAR) 从 25% 到 62-68%，从而直接减少 all-to-all 通信量 49-57%。

Pre-gated MoE (ISCA '24) 从另一个角度解决 EP 的问题：不使用 multi-GPU EP，而是将所有 expert 参数 offload 到 CPU，仅通过单 GPU 推理。通过 pre-gate function 提前知道下一个 block 需要的 experts，利用 CUDA stream 将 CPU→GPU expert migration 与 GPU expert computation 重叠，避免了 EP 中的 All-to-All 通信开销和 load imbalance 问题。

**ScMoE (ICML '25)** 从架构-调度协同设计角度优化 EP 通信瓶颈：通过shortcut连接使gating和All-to-All dispatch可以基于前一层表示提前启动，与当前层的attention+shared expert计算重叠。当通信时间 ≤ overlap_window（约50%总MoE时间）时实现100%通信隐藏（pipeline策略因其prologue/epilogue bubble无法达到）。在8×A30-PCIe（通信占60%）下实现1.49×训练加速和1.82×推理加速。

ScaleMoE 论文揭示了 EP 中 All-to-All 通信的 zero padding 问题：由于 expert selection 高度不均衡，Tutel/DeepSpeed 等框架为统一 all-to-all message size 而加入大量 zero padding（zero ratio 从训练初期 88% 升至后期 98%），导致通信量膨胀。ScaleMoE 提出 Adaptive All-to-All Communication 通过 all-gather 聚合 per-expert 选择计数后使用精确 slice size 的 NCCL alltoallv，消除 zero padding——all-to-all 通信开销减少 up to 81%。此外，Dynamic Expert Clustering 通过 K-means 聚类 tokens + expert replication + cold expert offload 重新均衡 EP 下的 expert-to-GPU 映射；Topology-aware Expert Remapping 使用遗传算法在异构网络中搜索近最优 cluster-to-GPU 映射。

涉及论文标题：
- Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
- ReXMoE Reusing Experts with Minimal Overhead in Mixture-of-Experts
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
- Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（首次在 Mesh TensorFlow 中系统提出 Expert Parallelism 作为独立并行维度，与 Data/Model Parallelism 组合；Switch-C 1.6T 参数模型使用 2048 experts 纯 EP+DP，Switch-XXL 395B 使用 EP+MP+DP）

**ES-MoE (ICML '24)** 将传统 EP 中 experts 常驻 GPU 的假设打破，通过 expert offloading + dynamic placement 实现 on-demand EP：experts 不再静态分配给 GPU，每 iteration 根据 gating output 动态决定 expert→GPU 映射。GPU 仅持有 non-expert params + active expert params + activations。Expert placement 由 greedy scheduling 算法在 CPU 执行（<2.69us），按 token load 均衡分配。

PopFetcher (USENIX ATC '25) 在 EP 基础上引入 popularity-based expert-wise prefetching：利用滑动窗口（s=10 iterations）预测下一层热门 expert，在 Attention 层（非 MoE 计算）期间通过独立 CUDA stream 异步预取 remote expert 参数到本地 GPU。已预取的 expert 的 token 直接本地计算——消除该部分 token 的 All-to-All dispatch。采用 hybrid push-pull 范式：当 token 传输量 > 2048 tokens（H=1024, ~16MB expert 参数）时 pull expert，否则 push token。在 8×RTX 4090 (100Gbps InfiniBand) 上，token 传输量减少 14.85%（MoE-GPT）、13.46%（MoE-BERT），per-iteration time 加速 1.28-2.4×。

- Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

Pro-Prophet (NUDT) 提出 lightweight expert placement 优化 EP 中的 expert-to-device 映射：每个 expert 仅传输到有其 input 的 device 子集（而非全部 devices），通过 Trans 原语（传输 parameters）和 Agg 原语（聚合 gradients）替代全局 model states 传输，显著降低通信量。Pro-Prophet 的 planner 通过 greedy algorithm + performance model 在 runtime 搜索 communication-efficient placement。

ScheMoE (EuroSys '24) 从任务调度角度重新审视 EP：将 EP 中 MoE layer forward/backward 的 7 类任务（compress、A2A dispatch、decompress、expert compute、compress、A2A combine、decompress）形式化为带数据依赖约束的调度问题，并通过数学证明给出了给定输入分区度 r 下的最优 CompTask 执行顺序（OptSche 算法）。此外，ScheMoE 提出 Pipe-A2A 通信算法——将 EP 中 A2A 的 intra-node SR 和 inter-node SR 分配到两个独立 CUDA stream 并发执行，使 EP 的通信阶段同时利用 intra-node 和 inter-node 带宽。ScheMoE 的 AbsCompressor/AbsAlltoAll/AbsExpert 三层抽象接口使得 EP 中的压缩算法和 A2A 算法可插拔替换而无需修改调度逻辑。

UCCL-EP (2025) 从通信系统可移植性角度解决 EP 的 vendor lock-in 问题：现有 EP 通信系统（DeepEP）通过 IBGDA 实现 GPU-initiated token-level 通信，但每个 (GPU vendor, NIC vendor) 组合需独立开发（O(m×n) 成本）。UCCL-EP 通过 CPU-proxy 架构解耦 GPU 通信发起与 NIC 通信执行——GPU 通过 FIFO channel 将 TransferCmd 传递给多线程 CPU proxy，CPU 通过 libibverbs（可移植 RDMA 库）执行所有 NIC 操作——仅需 O(m+n) 开发成本。在 EFA（无序传输、无硬件 atomics）和 Broadcom NIC 上首次实现 GPU-initiated token-level EP 通信，性能达 DeepEP 可比水平（NVIDIA-only）或更优（EFA 上优于 PPLX 2.1×）。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
- UCCL-EP Portable Expert-Parallel Communication

## Tensor Parallelism (TP / 张量并行)

术语是什么？
Tensor Parallelism (TP) 是将单个 Transformer layer 内的权重矩阵沿特定维度切分到多个 GPU 上的模型并行技术。由 Megatron-LM (Shoeybi et al., 2019) 提出。在 FFN 模块中，第一个 GEMM 的权重沿列切分（column-wise），第二个 GEMM 沿行切分（row-wise），使得中间计算在各设备上独立执行（无需通信），仅在 Dropout 前/后各需一次 all-reduce 同步。TP 的通信为节点内 NVLink all-reduce（高带宽低延迟），因此 TP size 通常限制在单节点 GPU 数（如 8）。

从kernel调度角度拆解术语：
FFN 模块的 TP 切分与通信调度（T 个设备）：
```
输入: X [b*s, h]
权重: A [h, 4h/T] per device (column-wise cut)
      B [4h/T, h] per device (row-wise cut)

// Forward
Y_i = GeLU(X @ A_i) @ B_i    // [b*s, h], 独立计算 T 份
Y = all_reduce(Y_1, ..., Y_T) // inner-node NVLink, 2*(T-1)*bsh/B 数据量

// Backward
∂L/∂Y_i = ∂L/∂Y              // 直接使用（已 all-reduced）
∂L/∂X_i = ∂L/∂Y_i @ B_i^T @ GeLU'(...) @ A_i^T
∂L/∂X = all_reduce(∂L/∂X_1, ..., ∂L/∂X_T)
```

在 PPMoE 中，TP 不仅用于 backbone，还用于 expert parallel——experts 分布在 TP group 内，MoE 层的 all-reduce 与 TP FFN 的 all-reduce 完全一致。

术语一般如何实现？如何使用？
Megatron-LM/Core 通过 `tensor_model_parallel_size` 配置。TP 的通信开销分析（Eq. 5）：t_all-reduce/t_cal = (T-1)TF/(4Bh)，以 V100 (F=125 TFLOPS, B=300 GB/s NVLink, T=8, h=10^3) 为例约 35/6≈6，远低于 DPMoE 的 all-to-all 开销。通常与 DP/PP 组合使用（3D 并行）。Pipeline MoE 论文中 TP=8 是默认配置，保证所有 experts 在单节点内。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

## Pipeline Parallelism (PP / 流水线并行)

术语是什么？
Pipeline Parallelism 是将模型按层（layer）切分为多个 stage，每个 stage 放在不同设备/节点上的模型并行技术。与 TP 在宽度维度切分不同，PP 在深度维度切分模型。前向时前一 stage 完成计算后将中间 hidden states 通过 p2p 通信发给下一 stage；反向时后一 stage 完成 backward 后将梯度发回。常用调度策略为 1F1B（one-forward-one-backward），在稳定阶段每个设备交替执行一个 forward 和一个 backward，通过 micro-batch pipeline 填充 bubble。

从kernel调度角度拆解术语：
PP 的 1F1B 调度（P stages, M micro-batches）：
```
Timeline (device 2 of 4-stage pipeline, M=8):
Warmup:  F0 F1 F2 F3
Steady:  F4 B0 F5 B1 F6 B2 F7 B3
Cooldown:         B4 B5 B6 B7
```
其中 F=forward, B=backward。Bubble 比例 = (P-1)/(P-1+M)，micro-batch 数 M 越大 bubble 越小。

在 PPMoE 中，PP 与 EP+TP 无缝集成——MoE 层的输入/输出格式和通信模式与非 MoE FFN 一致，dense 模型的 TP+PP 框架可直接通过替换部分 FFN 为 MoE 层转化为 PPMoE。

术语一般如何实现？如何使用？
Megatron-LM 的 `pipeline_model_parallel_size` 配置。PP 的通信为 p2p send/recv（节点间 InfiniBand 传输 b*s*h 数据），通信量远小于 TP 的 all-reduce 频率。PP 的缺点是 bubble overhead——bubble ratio = (P-1)/M，小模型更明显（PPMoE 小规模实验中 PP=4, M 较小时 bubble 显著）。适用场景：模型超出单节点内存时，PP 是必需的扩展方式。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

## Data Parallelism (DP / 数据并行)

术语是什么？
Data Parallelism 是最基础且最广泛使用的分布式训练并行技术。每个设备持有完整模型副本，输入数据按 batch 维度切分（micro-batches）分配到各设备，各自独立执行 forward+backward，完成后通过 all-reduce 同步梯度。DP 不切分模型参数，仅切分数据，因此每设备需能容纳完整模型。

从kernel调度角度拆解术语：
```
// DP 训练一个 iteration（D 个 DP ranks）
每个 rank r 独立执行:
    loss_r = model.forward(micro_batch_r)
    loss_r.backward()        // 计算本地梯度
// 梯度同步
all_reduce(gradients)        // 所有 ranks 的梯度求平均
// 优化器更新
optimizer.step()             // 各 rank 独立更新（梯度已同步 → 参数一致）
```

在 DPMoE 中，DP 与 EP 绑定——每 DP rank 持有 E/D 个 experts，EP 的 all-to-all 通信发生在 DP ranks 之间的 MoE layers。在 PPMoE 中，DP 与 EP 解耦——DP 仅用于扩展 global batch size，不影响 expert 分布。

术语一般如何实现？如何使用？
PyTorch DDP (DistributedDataParallel) 或 DeepSpeed ZeRO。PPMoE 实验中，Dense/backbone 使用 DP 扩展（DP=32 时 5120 tok/s/GPU），但 PPMoE 因 PP 已提供足够的 batch scaling 而省略 DP（DP=1）。ZeRO optimizer 可与 DP 结合使用以降低 per-rank 内存占用。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

## Tensor Index Slicing (Index Select) for MoE Dispatch（张量索引切片MoE调度）

术语是什么？
Tensor Index Slicing（index_select）是 PPMoE 中替代 all-to-all 的 token dispatch 机制。由于 PPMoE 将所有 experts 放置在同一 TP group（同一节点）内，且各 TP rank 持有相同的 hidden states（经 TP all-reduce 同步）和相同的 dispatching order（相同的 gating 输入），dispatch 仅需本地 index_select 操作——无通信开销。这是 PPMoE 消除 all-to-all 的关键技术。

从kernel调度角度拆解术语：
```
// PPMoE dispatch: index_select (替代 all-to-all)
// 输入: hidden_states [8, h/T], indices [8]（dispatching order）
// indices = [2, 3, 1, 2, 0, 3, 2, 0]
// N=4 experts on this TP rank (E=64, T=8)

// 按 expert 分组 token
X0 = hidden_states[[4, 7], ...]   // expert 0: tokens 4,7
X1 = hidden_states[[2], ...]      // expert 1: token 2
X2 = hidden_states[[0, 3, 6], ...] // expert 2: tokens 0,3,6
X3 = hidden_states[[1, 5], ...]   // expert 3: tokens 1,5

// 然后串行执行各 expert FFN
```

对比 DPMoE 的 all-to-all dispatch（跨节点传输 b*s*h 数据，走 InfiniBand 12.5 GB/s），PPMoE 的 index_select 是本地 PyTorch 操作，零通信，利用 NVLink 300 GB/s 的 all-reduce 替代 InfiniBand all-to-all 完成 gather。

术语一般如何实现？如何使用？
PyTorch 的 `torch.index_select` 或高级索引 `tensor[indices]` 实现。要求：(1) 所有 experts 在同一节点内；(2) 各 TP rank 持有相同输入（通过 TP 的 copy_to_tensor_parallel_region 保证）；(3) gating network 的参数在 TP ranks 间需同步（仅 h×E 大小，可忽略）。适用场景：PPMoE 或任何将 experts 集中在单节点内的 MoE 并行架构。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

## 1F1B Pipeline Schedule（一前一后流水线调度）

术语是什么？
1F1B (One-Forward-One-Backward) 是 Pipeline Parallelism 中最常用的 micro-batch 调度策略。由 PipeDream (Harlap et al., 2018) 提出。调度分三阶段：(1) Warmup——各 stage 依次执行 forward 填充 pipeline；(2) Steady——各 stage 交替执行 1 个 forward 和 1 个 backward；(3) Cooldown——各 stage 依次完成剩余的 backward。1F1B 相比 GPipe（全 forward 后全 backward）将 activation memory 峰值从 O(M×P) 降至 O(M)，其中 M 为 micro-batch 数，P 为 pipeline stage 数。

从kernel调度角度拆解术语：
4-stage pipeline (P=4), M=8 micro-batches 的 1F1B 调度：
```
Device 0: F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
Device 1:    F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
Device 2:       F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
Device 3:          F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
```
Bubble（空闲时间）占总时间的比例 = (P-1)/(P-1+M)。

在 PPMoE 中，1F1B 与 MoE 层无缝配合——MoE 层的 forward/backward 通信模式与非 MoE FFN 相同（均为 all-reduce），因此 1F1B 调度无需修改即可用于 PPMoE。

术语一般如何实现？如何使用？
Megatron-LM 默认使用 1F1B 调度（可通过 `--num-layers-per-virtual-pipeline-stage` 进一步减少 bubble）。DeepSpeed 的 PipeDream 也支持 1F1B。PPMoE 实验中 PP=4（小规模）和 PP=16（大规模）。bubble overhead 在小模型/micro-batch 少时显著——PPMoE 小规模（6.7B）backbone 仅达 81.4% throughput（vs 90.7% for 143B），部分因 PP bubble。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

## All-Reduce Communication in Distributed Training（分布式训练中的全归约通信）

术语是什么？
All-Reduce 是分布式训练中最核心的集合通信原语之一。将各设备上的数据（通常是梯度或 hidden states）求和/平均后广播到所有设备。在 TP 中用于同步 FFN/attention 的输出和梯度；在 DP 中用于同步梯度。All-Reduce 的通信复杂度为 O(2(N-1)/N · data_size)，在 ring 算法下每设备收/发 2(N-1)·data_size/N 数据。节点内 all-reduce（NVLink）带宽远高于节点间（InfiniBand），因此 TP 的 all-reduce 开销远低于 EP 的 all-to-all。

从kernel调度角度拆解术语：
```
// Ring All-Reduce (N 个设备)
// 分两步：Reduce-Scatter + All-Gather
每个设备 data = local_tensor [size M]

// Step 1: Reduce-Scatter (N-1 步)
for step in 1..N-1:
    send chunk[(rank-step)%N] to (rank+1)%N
    recv chunk[(rank-step-1)%N] from (rank-1+N)%N
    reduce recv_chunk into local accumulator

// Step 2: All-Gather (N-1 步)
for step in 1..N-1:
    send reduced_chunk to (rank+1)%N
    recv chunk from (rank-1+N)%N
```

在 PPMoE 中，MoE 层的 all-reduce 通信量 = 2×b×s×h per global batch（与 TP FFN 完全相同），走 NVLink (300 GB/s)，远低于 DPMoE 的 all-to-all（走 InfiniBand 12.5 GB/s）。PPMoE 将 MoE all-reduce 时间降至仅比 FFN all-reduce 多 1.9% of total forward time。

术语一般如何实现？如何使用？
NCCL (NVIDIA Collective Communications Library) 提供 `ncclAllReduce`，自动选择最优算法（ring/tree/collnet）根据拓扑。PyTorch 通过 `torch.distributed.all_reduce` 调用。PPMoE 中 MoE 层的 all-reduce 通过 Megatron 的 `reduce_from_tensor_parallel_region` 封装。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

## Split-Phase Transmission（分阶段传输）

术语是什么？
Split-Phase Transmission 是 PROBE 中用于在 dual-track 架构下管理 P2P expert weight prefetch 带宽的技术。核心思想是将 expert 传输拆分为两个阶段：(1) Phase 1 — 在当前层 L 的 MoE Compute 期间启动 P2P 传输，利用计算 kernel 不消耗网络带宽的特性；(2) 在 All-to-All Combine 之前暂停传输，释放 NVLink/NVSwitch 带宽给关键通信路径；(3) Phase 2 — Combine 完成后恢复传输，利用下一层 L+1 的 Attention 计算窗口完成剩余传输。

从kernel调度角度拆解术语：
```
Timeline of Split-Phase Transmission on Rank r:

  Main Stream:           |   Aux Track (Prefetch):
                         |
  MoE Compute █████████  |   Prefetch Phase 1 ░░░░░░  (P2P put expert weights)
                         |   ↑ 启动传输
  ── barrier ──          |
  All-to-All Combine ███ |   (PREFETCH SUSPENDED)     ← 释放带宽
                         |   ↑ 暂停传输
  ── barrier ──          |
  Layer L+1 Attention ██ |   Prefetch Phase 2 ░░░░░░  (resume & complete)
                         |   ↑ 恢复传输
```
Split-phase 的关键约束：(1) 传输必须可暂停/恢复——PROBE 将 expert weight 分 chunk，通过 CUDA event 机制控制传输窗口；(2) 暂停时机必须精确对齐 All-to-All Combine 的开始时间——通过 CUDA stream callback 或 pre-recorded event 实现。

术语一般如何实现？如何使用？
PROBE 通过 CUDA stream 管理和 custom Triton P2P kernel 实现。传输在独立 CUDA stream 上进行，主 stream 的 All-to-All launch 前通过 cudaStreamWaitEvent 确保 prefetch stream 的所有 in-flight 传输已完成。适用场景：任何需要在通信密集期（All-to-All）前后进行 bulk 数据传输的 MoE 系统，特别是高带宽 NVSwitch 环境下的 online expert replication。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

## All-to-All Collective in MoE Inference（MoE推理中的全交换集合通信）

术语是什么？
All-to-All Collective 是 Expert Parallelism 下 MoE 层执行的关键通信原语。每次 MoE 层执行两次 All-to-All：(1) Dispatch — 各 rank 将 token hidden states 按 Router 决策发送到持有对应 expert 的 rank；(2) Combine — 各 rank 将 expert 计算结果发回 token 原始所在 rank。与 Sequence Parallelism 中的 All-to-All（swapping sequence/head layout）不同，MoE 的 All-to-All 是 token-level scatter/gather，通信模式由 Router 输出动态决定。

从kernel调度角度拆解术语：
MoE All-to-All 通信 kernel 的调度流程：
```
// Dispatch Phase
每个 rank r:
  send_buf = []  // 按目标 rank 分组
  for token t in local_batch[r]:
    for expert e in topk_indices[t]:
      target = expert_to_rank[e]
      send_buf[target].append((token_data[t], e, t_idx))
  
  All-to-All Scatter: send_buf[target] → rank target
  
每个 rank 接收后:
  recv_tokens = 从各 rank 收到的 tokens
  // 按 expert 分组用于 Grouped GEMM

// Combine Phase (对称反向)
每个 rank r:
  按 t_idx 排序 expert outputs
  All-to-All Gather: expert_outputs → token 原始 rank
```
PROBE 论文揭示了 MoE All-to-All 的 Double Penalty：hotspot rank 同时是最大收发量 rank——Dispatch 时接收最多 unique token，Combine 时发送最多 output。DeepEP 通过 token deduplication 和 topology-aware routing 优化了通信效率，但无法消除 skew 导致的瓶颈 rank。

术语一般如何实现？如何使用？
主流实现：NCCL All-to-All（通用）、DeepEP（MoE 专用，优化 token dedup 和 NVLink 拓扑）、SGLang 集成的 DeepEP normal mode。跨节点时使用 RDMA (InfiniBand/RoCE)。关键优化维度：(1) token deduplication — 同一 rank→同一 remote expert 的多 token 合并为单次 send；(2) 与 GEMM overlap — 利用 CUDA stream 将通信与计算流水线化。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

PopFetcher (USENIX ATC '25) 重点解决 MoE 训练中 All-to-All 占单层总时间 50-60% 的瓶颈：通过非 MoE 计算（Attention 层）期间异步预取热门 expert，使被预取 expert 的 token 本地计算而无需 All-to-All dispatch；在 backward pass 中将 All-to-All 和 All-Reduce 分解为 micro-operations 流水线交错执行，All-to-All 优先级高于 All-Reduce，避免 gradient aggregation 阻塞 token 回传。训练 latency 公式：Lat_w^origin = 3×4B_wαH²/P_w + 4H Σ B_{n,w}^i / W_{n,w}；预取后，token transfer 项变为仅未预取 expert 的 token（即 B_{n,w}^i(1-δ_{n,w}^i)），加上梯度 reduction 开销 2αH² Σ δ_{n,w}^i / W_{n,w}。

## Grouped GEMM in MoE（MoE中的分组通用矩阵乘）

术语是什么？
Grouped GEMM 是 MoE 推理中 expert 计算的核心 kernel 操作。不同于标准批处理 GEMM（所有 token 用同一权重），Grouped GEMM 将不同 expert 的 token 分组，每组用不同权重矩阵执行矩阵乘法。在 CUDA 层面通常通过 CUTLASS grouped GEMM 或 cuBLAS 实现，支持多个独立 GEMM 在单次 kernel launch 中批量执行。

从kernel调度角度拆解术语：
Grouped GEMM 的 kernel 调度与 GEMM efficiency η_g 的关系：
```
// Rank r 上执行 expert e 的 Grouped GEMM
输入: tokens_e ∈ ℝ^{n_e × H}, weights W_e^{gate,up,down}

// SwiGLU FFN 的 3 次 GEMM:
h_gate = tokens_e @ W_e^gate    // GEMM 1: (n_e, H) @ (H, I)
h_up   = tokens_e @ W_e^up      // GEMM 2: (n_e, H) @ (H, I)  
h_act  = SiLU(h_gate) ⊙ h_up   // element-wise
output = h_act @ W_e^down       // GEMM 3: (n_e, I) @ (I, H)

// 性能受 η_g(n_e) 影响:
η_g(n_e) ∝ n_e  // 当 n_e 大时接近 peak FLOPS
η_g(n_e) ≪ 1   // 当 n_e 小时受 padding + low intensity 影响
```
PROBE 分析指出 DP (Data Parallelism) 的 fragmentation penalty 源于每个 replica 处理的 per-expert token 过少，GEMM 效率 η_g 极低。EP 通过聚合全局 token 维持高 η_g，但引入 load imbalance。Expert replication 在两者间 trade off。

术语一般如何实现？如何使用？
CUTLASS 3.x grouped GEMM (支持 Hopper SM90)、cuBLAS grouped GEMM API、Triton 自定义 grouped matmul kernel。在 vLLM/SGLang 中通过 `torch.bmm` 或框架封装的 fused MoE kernel 调用。关键优化：tile size 对齐 Tensor Core 的 M/N/K 维度和 expert 级 padding 策略。

"Who Says Elephants Can't Run" 使用的 CUTLASS Grouped GEMM 方法：在 token routing 完成后（CUB radix sort + permute），为每个 expert 构造子矩阵指针（start offset + token count），将 (sub_matrix_ptr, weight_ptr, bias_ptr) 组队传入 CUTLASS grouped GEMM，单次 kernel launch 并行执行所有 expert 的矩阵乘法。关键特点是 fused dequantize 在 GEMM weight load 阶段进行，对 V100 (Volta, SM70) 优化，使用 FP16 bit-trick 替代原生 I2F 指令加速 int→FP16 转换。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

## CUDA Graph in MoE Serving（MoE服务中的CUDA图）

术语是什么？
CUDA Graph 是 NVIDIA CUDA 12.x 提供的 GPU 工作流图机制，将一系列 CUDA kernel launch 预先录制为静态计算图并重放执行。在 LLM serving 中，decoding phase 的 kernel 序列固定（每个 token 执行相同操作），CUDA Graph 消除了 CPU-side kernel launch overhead，实现 kernel 级 back-to-back 执行。PROBE 面临的挑战：动态 expert replication 和 P2P 传输产生变量控制流，与 CUDA Graph 的静态图要求冲突。

从kernel调度角度拆解术语：
```
// CUDA Graph 录制（静态配置）：
cudaStreamBeginCapture(stream)
  // 录制一系列 kernel launch:
  router_kernel<<<...>>>(...)
  alltoall_dispatch_kernel<<<...>>>(...)
  grouped_gemm_kernel<<<...>>>(...)
  alltoall_combine_kernel<<<...>>>(...)
cudaStreamEndCapture(stream, &graph)

// 推理时重放（避免 per-kernel launch overhead）：
cudaGraphLaunch(graph_exec, stream)  // 单次调用执行全部 kernel
```
PROBE 解决 Graph 兼容性的策略：(1) Planner 运行在 GPU 上（单 SM kernel），消除 host-device sync；(2) Prefetch 传输通过 CUDA event 控制，不引入 host-side 条件分支；(3) Expert slot 通过双缓冲管理，地址在 graph capture 时固定；(4) 动态 routing assignment 通过预分配的 device buffer 传递。

术语一般如何实现？如何使用？
vLLM 和 SGLang 在 decoding phase 使用 CUDA Graph（prefill 因 batch size 变化不使用）。通过 `cuda.graph()` 上下文管理器录制，`graph.replay()` 重放。PROBE 的 planner 和 prefetch 在 graph capture 前预先录制为可重放的 sub-graph，运行时通过 CUDA graph update 或 node-level 参数更新机制适应动态配置。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

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

## IBGDA (InfiniBand GPUDirect Async / GPU直接异步RDMA)

术语是什么？
IBGDA（InfiniBand GPUDirect Async）是 NVIDIA 提供的一种技术，允许 GPU threads 直接向 InfiniBand RDMA NIC 提交网络操作（RDMA write/send/atomic），完全绕过 CPU。GPU SM 通过写入 NIC 的 MMIO doorbell/register 接口直接将 work requests 提交到 NIC 硬件队列，NIC 从 GPU memory 直接 DMA 读取数据并通过网络发送。

从kernel调度角度拆解术语：
IBGDA 使 GPU kernel 可以直接发起和管理网络传输，无需 CPU 参与：
```
// 传统 CPU-initiated RDMA 路径:
// GPU compute → cudaMemcpy(→CPU) → CPU post WR → NIC DMA → network
// 延迟: GPU compute + PCIe read + CPU post + NIC DMA

// IBGDA GPU-initiated RDMA 路径:
// GPU kernel 直接写 NIC MMIO doorbell → NIC DMA from GPU memory → network
// 延迟: GPU compute + NIC DMA (消除 CPU 和一次 PCIe 穿越)

// 在 DeepEP 中的使用:
// GPU SM thread:
//   1. 构造 RDMA work request (addr, length, dest_qp, ...)
//   2. 写入 NIC doorbell register (MMIO write)
//   3. NIC 从 GPU memory 直接读取 token activation data
//   4. NIC 打包 RDMA 包并发送
```

术语一般如何实现？如何使用？
IBGDA 要求：(a) InfiniBand-capable NIC（如 NVIDIA ConnectX-7）；(b) GPU 驱动暴露 NIC MMIO 接口（BAR1 mapping）；(c) GPU 和 NIC 之间基于 PCIe 的直接通信路径。NVIDIA 通过 NVSHMEM 库提供 IBGDA 接口。**核心可移植性问题**：IBGDA 要求 GPU 直接操作 NIC 的特定 MMIO 寄存器，每一个 (GPU vendor, NIC vendor) 组合都需要独立编写和维护集成代码。假设 m 种 GPU、n 种 NIC，需 O(m×n) 开发工作量。DeepEP 官方仅支持 NVIDIA GPU + NVIDIA NICs 组合，无法在 AWS EFA 或 Broadcom NIC 上运行。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## GPU-initiated token-level communication（GPU 发起的 Token 级通信）

术语是什么？
GPU-initiated token-level communication 是 MoE Expert Parallelism 中的一种通信模式：GPU threads 直接在 token 粒度上发起 RDMA 传输（而非先批量打包到 buffer 再统一发起），实现 per-token 或 per-chunk（如 32 tokens）的 fine-grained 通信。这种细粒度通信通过 IBGDA 技术实现，GPU kernel 为每个 token/chunk 独立构造 work request 并直接提交到 NIC。

从kernel调度角度拆解术语：
```
// 对比 coarse-grained vs fine-grained:
// Coarse-grained (NCCL/RCCL):
//   GPU: 将 T 个 tokens 按 dest_rank 打包到连续 buffer (O(T·C·D) memory)
//   CPU: 为每个 dest_rank 构造一个 bulk WR → NIC send
//   问题: packing 开销大，小 T 时吞吐低

// Fine-grained GPU-initiated (DeepEP/UCCL-EP):
//   GPU: 每 token/chunk (如 32 tokens) 独立提交 TransferCmd
//   NIC: 直接从 GPU buffer DMA 数据 (无需 CPU 参与 packing)
//   优势: overlap token dispatch 与 compute, dedup, hierarchical reduce

// 通信量: 典型 7M ops/s/GPU (DeepSeek-V3, 7KB/activation, 400G network)
```

术语一般如何实现？如何使用？
通过 DeepEP (IBGDA-based) 或 UCCL-EP (CPU-proxy-based) 实现。在 HT mode 中，32 tokens 构成一个 chunk，一次提交传输。GPU-initiated 使 token deduplication 和 hierarchical reduce 成为可能：GPU kernel 在提交传输前检查同节点多专家场景，去除冗余传输。在 UCCL-EP 中，GPU 仍负责 token-level 的 initiation 决策（保持 fine-grained overlap），但实际传输执行委托给 CPU proxy（换取可移植性）。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## GPU-CPU FIFO Channel with TransferCmd（GPU-CPU FIFO 通道与传输命令）

术语是什么？
UCCL-EP 设计的 lock-free FIFO 通道，用于 GPU threads 向 CPU proxy 传递 128-bit 的 TransferCmd（传输命令描述符）。通道的 head 元数据在 CPU 内存中、tail 元数据在 GPU 内存中，双方各自访问本地内存侧的头/尾以减少 PCIe 穿越。GPU 侧缓存 tail index 避免跨 PCIe 读取。支持 4 种 TransferCmd：Write、Atomics、Drain、Barrier。

从kernel调度角度拆解术语：
```
// TransferCmd 结构 (128-bit = 16 bytes, 可单条 GPU 指令+MMIO doorbell 写入):
//   type: Write | Atomics | Drain | Barrier
//   dest_rank: 目标 GPU rank
//   src_offset: 源 buffer offset (symmetric memory)
//   dst_offset: 目标 buffer offset
//   length: 传输字节数
//   seq_num: 序列号 (用于 ordering)

// GPU 侧 API:
//   idx = Push(TransferCmd)    // GPU thread 入队命令，返回 index
//   CheckCompletion(idx)       // GPU thread 检查命令是否被 CPU 消费完成

// CPU 侧 API:
//   cmd = Poll()    // CPU proxy 读取但不移除队首命令
//   Pop()           // CPU proxy 移除队首命令 (表示已完成处理)

// 多 FIFO channels per GPU:
//   8 FIFO channels / GPU × 4 CPU threads
//   同 channel 内的命令保证 ordering，跨 channel 不保证
//   GPU kernel 将需要 ordering 的消息映射到同一个 channel

// CPU proxy 背压机制:
//   kMaxInflight: 每个 channel 最大 in-flight 命令数
//   当 channel 满时 GPU thread 阻塞在 Push() 上
//   延迟 Pop() = 延迟 GPU enqueue = rate-limiting GPU sender
```

术语一般如何实现？如何使用？
实现依赖 GPU memory（device buffer）+ CPU memory（host buffer）的共享 FIFO。GPU writes 需 bypass L2 cache（volatile + memory fence），CPU writes 需 flush 到 host memory。在 NVLink-C2C（GH200）等 cache-coherent CPU-GPU 互联上，一致性由硬件保证。FIFO 吞吐量达 8 Mops/s（单 channel），latency 比网络延迟低一个数量级。TransferCmd 使用 offset 而非全局地址（配合 symmetric memory），节省 bits 并使 128-bit 紧凑编码可行。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## RDMA immediate data (ordering emulation / RDMA 即时数据)

术语是什么？
RDMA immediate data 是 RoCEv2 协议包头中的 32-bit 字段，RDMA write/send 操作可在发送数据的同时携带该字段。接收端 CPU 通过 polling completion queue (CQ) 获取该 32-bit 值，无需访问远端 GPU memory。UCCL-EP 利用此字段嵌入 sequence number 和 expert index，在 CPU proxy 中实现不支持硬件 ordering 的 NIC 上的 delivery semantics 模拟。

从kernel调度角度拆解术语：
```
// EFA SRD 协议: 可靠但无序 (unordered delivery)
// 问题: GPU kernel 假设 write→atomic 严格顺序
//       但 EFA 可能让 atomic (用于 ring buffer tail update)
//       先于对应的 data writes 到达，导致 GPU 读到 stale data

// UCCL-EP solution via immediate data:
// 发送端:
//   RDMA_write(dst_addr, data_payload, imm_data = seq_num | expert_idx)
//   RDMA_write_atomic(dst_addr, tail_update, imm_data = atomic_seq)

// 接收端 CPU proxy:
//   cqe = poll_cq()
//   seq = cqe.immediate_data & SEQ_MASK
//   if cqe is atomic and seq > last_applied_write_seq:
//       将 atomic 暂存到 control buffer (unordered arrival)
//   elif cqe is write:
//       标记 write_seq 已到达
//       检查 control buffer 中是否有 pending atomic 现在可 apply
//   if all prior writes done:
//       apply atomic (更新 ring buffer tail)
```

术语一般如何实现？如何使用？
Immediate data 字段在 RoCEv2 标准包头中定义，几乎所有 RDMA NIC 都支持（包括 EFA、ConnectX、Broadcom）。在 UCCL-EP 中：(a) LL mode：接收端 CPU 用 immediate data 中的 expert index 做 conditional check——仅当特定 expert 的 X 个 writes 完成后才 apply atomic；(b) HT mode：per-channel 的 sequence number 用于保证 ring buffer head/tail 更新不引入 race condition；(c) EFA 上模拟 atomics：将 atomic 值打包进 immediate data 的 RDMA write，接收端 CPU 更新 host memory counter。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## NVSHMEM Symmetric Memory for MoE Serving（面向MoE服务的NVSHMEM对称内存）

术语是什么？
NVSHMEM (NVIDIA Shared Memory) 是 NVIDIA 提供的基于 OpenSHMEM 标准的 GPU 间通信库，核心特性是 symmetric memory——在所有 GPU 上分配相同大小、相同虚拟地址空间的内存区域，允许每个 GPU 通过直接 put/get 操作访问远程 GPU 的对称内存。PROBE 使用 NVSHMEM symmetric memory 管理 replicated-expert buffer：每 rank 分配固定大小的对称内存区域存放最多 3 个 expert 副本（双缓冲 6 slots），通过 NVSHMEM put 实现高效 expert weight P2P 传输。

从kernel调度角度拆解术语：
NVSHMEM 在 PROBE 中的使用：
```
// 初始化：在 EP group 内分配 symmetric memory
nvshmem_init()
expert_buffer = nvshmem_malloc(sizeof(expert) × 6 × num_ranks)  // 对称内存
// expert_buffer 在所有 rank 上具有相同虚拟地址

// P2P expert 传输（Triton kernel 内）：
nvshmem_putmem_nbi(
    dst = expert_buffer + rank_dst * 6 * sizeof(expert) + slot_idx,
    src = expert_weights + expert_idx * sizeof(expert),
    size = sizeof(expert),
    pe = rank_dst           // 目标 rank
)

// 全局 All-Gather 聚合预测结果：
nvshmem_allgather(
    output = global_prediction_counts,  // [ep, num_experts]
    input = local_prediction_counts,    // [num_experts]
    size = num_experts × sizeof(int)
)
```
与 NCCL collective 对比：NVSHMEM put/get 是单边操作（不需要目标 rank 参与同步），更灵活；支持 GPU kernel 内直接发起，无需 CPU 线程介入；适合 PROBE 的 P2P expert 传输场景。

术语一般如何实现？如何使用？
需要 NVLink/NVSwitch（单机内）或 InfiniBand（跨机）。安装：`nvshmem 3.3.20+`，配合 CUDA 12.x。使用流程：(1) `nvshmem_init()` 初始化；(2) `nvshmem_malloc()` 分配对称内存；(3) `nvshmem_put/get` 进行单边 RDMA 传输；(4) `nvshmem_barrier_all()` 同步。PROBE 中限制 6 slots per rank 的对称内存开销（双缓冲 3 incoming + 3 outgoing），避免过多侵占 KV cache 空间。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

## HalfRing Algorithm（半环算法）

术语是什么？
HalfRing 是针对环（1-D torus）拓扑上 All-to-All 集合通信的带宽与延迟最优算法。通过利用双向链路的最短路径通信原则，HalfRing 在每个阶段根据收发节点间实际距离选择传输方向（而非固定双方向发送），使得配对阶段可同时利用两条方向相反的链路，消除 Ring 算法的非最短路径带宽浪费。在 N 节点环上，当 N=2k+1 时有 2k 个阶段配对为 k 对同时执行；当 N=2k 时有 2k-1 个阶段，剩余一个未配对阶段将数据等分后双向发送。HalfRing 保证无死锁（仅单跳传输）、无活锁（无绕路）、无网络争用（逐跳显式编排链路分配）。

从kernel调度角度拆解术语：
HalfRing 的逐跳 store-and-forward 调度过程（以 4 节点环为例）：

```
Procedure HalfRing_Generator(Ring_Nodes[N_nodes], Data_Size S):
    // N_nodes 为节点数；每个节点初始持有 S 大小数据，All-to-All 完成时每节点持有所有节点的数据
    if N_nodes % 2 == 1:
        Stage_Num = (N_nodes - 1) / 2   // 奇数节点：Stage_Num 对
    else:
        Stage_Num = N_nodes / 2          // 偶数节点

    comm_size = S / N_nodes
    for stage = 0 to Stage_Num - 1:
        if stage == Stage_Num - 1 and N_nodes % 2 == 0:
            comm_size = comm_size / 2    // 最后一个未配对阶段数据减半

        Sub_Stage_Num = stage            // 该阶段需 stage 次逐跳转发
        for sub = 0 to Sub_Stage_Num - 1:
            for each node in Ring_Nodes:
                // 顺时针方向：node → (node+1)%N_nodes
                Dest_CW[stage][sub] = (node + 1) % N_nodes
                // 逆时针方向：node → (node-1+N_nodes)%N_nodes
                Dest_ACW[stage][sub] = (node + N_nodes - 1) % N_nodes
                Comm_Size[stage][sub] = comm_size
            // 每个子阶段：所有相邻节点对同时执行单跳传输
```

线性成本模型性能分析（S=单节点数据量, N=节点数, B=单向带宽, α=每跳延迟）：
- Ring: 传输时间 = (N-1)/2 · S/(2B)，启动时间 = α
- HalfRing (N 偶数): 传输时间 = N/8 · S/B，启动时间 = α，加速比 1~2×
- HalfRing (N 奇数): 传输时间 = (N²-1)/8 · S/(NB)，启动时间 = α，加速比 1.5~2×

术语一般如何实现？如何使用？
HalfRing 通过离线预计算通信时间表实现——对给定拓扑的所有阶段、子阶段、节点对生成确定的发送方→接收方+转发映射表。运行时将该时间表下发到通信后端（如 MPI 的 Isend/Irecv 或 PyTorch Distributed 的 send/recv），每个节点按其时间表在对应子阶段执行单跳数据传输和转发。适用于 N-D torus 的单维 All-to-All 阶段，需配合 DimRotation 等多维调度使用。PyTorch Distributed 实现中，通信对和传输顺序预先离线计算，CPU 侧 kernel launch 开销显著降低。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

## DimRotation Scheduling（维度轮转调度）

术语是什么？
DimRotation 是针对 N-D torus 网络上 All-to-All 集合通信的多维调度方法。将数据分为恰好 N 个 chunk（N 为拓扑维度数），第 i 个 chunk 按维度 i → i+1 → ... 的循环顺序执行单维 All-to-All。与 Pipeline 调度（所有 chunk 使用相同维度顺序 X→Y→Z）相比，DimRotation 实现零气泡（bubble-free）的全维度链路利用率。

从kernel调度角度拆解术语：
DimRotation 的 chunk 调度逻辑（3D torus 为例）：

```
Procedure DimRotation_Scheduler(S, N):
    Chunk_Num = N              // chunk 数 = 维度数
    Chunk_Size = S / N
    for chunk = 0 to Chunk_Num - 1:
        Schedule[chunk] = []   // 该 chunk 的维度遍历顺序
        for phase = 0 to N - 1:
            dim = (chunk + phase) % N
            Schedule[chunk].append(dim)
    // 3D torus: chunk0→[X,Y,Z], chunk1→[Y,Z,X], chunk2→[Z,X,Y]
    // 3 个 chunk 在 3 个维度上形成完美全覆盖
```

调度时间线优势（以 3D torus 为例）：
- Pipeline（6 chunks, X-Y-Z 顺序）：各 chunk 在不同维度上流水线执行，但固定顺序产生气泡——当 chunk1 在 X-dim 完成需等待 chunk0 释放 Y-dim 链路
- DimRotation（3 chunks, 轮转顺序）：每个时刻恰好有 1 个 chunk 在 X-dim、1 个在 Y-dim、1 个在 Z-dim，链路利用率恒为 100%，零调度气泡

术语一般如何实现？如何使用？
DimRotation 与单维算法（HalfRing/FoldedRing）配合使用——Scheduler 过程（Algorithm 1）先确定每个 chunk 的维度遍历顺序，在每个 phase 调用对应的单维算法生成器（HalfRing_Generator）生成该维度上逐跳传输时间表。对于异构带宽或 mixed-radix torus（如某维度无 wrap-around 链路），总时间受限于性能最差维度的完整数据传输时间。Chunk 数固定为 N（最小充分数量），避免 Pipeline 中 chunk size 选择的困境。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

## FoldedRing Algorithm（折叠环算法）

术语是什么？
FoldedRing 是针对环拓扑上单链路故障的容错 All-to-All 算法。当环上某链路故障时，FoldedRing 利用故障链路两端节点之间的所有反向（逆时针）物理链路构建逻辑补偿连接——将这些反向链路"折叠"为故障链路的替代路径。结合其余健康的顺时针链路，恢复 Ring 算法的逻辑通信模式。FoldedRing 可扩展到其他集合通信（All-Reduce、Reduce-Scatter、All-Gather）。

从kernel调度角度拆解术语：
FoldedRing 的路径构建过程（4 节点环，节点 1-4 间链路故障）：

```
Procedure FoldedRing_Gen(Ring_Nodes, Link_state):
    N_nodes = size(Ring_Nodes)
    for stage = 1 to N_nodes - 1:
        for node = 0 to N_nodes - 1:
            dest = (node + stage) % N_nodes
            if Link[node][dest] exists:     // 直接链路存在
                FoldedRing_Comm[stage][node] = dest
            else:                           // 链路故障，构建折叠路径
                path = []
                curr = node
                while curr != dest:
                    next = (curr - 1 + N_nodes) % N_nodes  // 逆时针绕行
                    if Link[curr][next] exists:
                        path.append(next)
                        curr = next
                    else:
                        break               // 无法到达，失败
                FoldedRing_Comm[stage][node] = path
```

性能特征（单链路故障场景）：传输时间 = (N-1)/2 · S/B（Ring 的传输时间为 (N-1)/2 · S/(2B)，即 FoldedRing 性能为 Ring 的 0.5×）。启动时间 = (N-1)α（因需要建立绕行路径连接，远超 Ring 的 α）。仅能处理单维环上的单一链路故障——对于同一环上两个或更多故障，FoldedRing 无法构建折叠路径。

术语一般如何实现？如何使用？
FoldedRing 作为 MATE 调度中故障环上的基础容错传输机制使用。在正常 phase 中（MATEe 模式），FoldedRing 传输部分数据；在加速 phase 中与 HalfRing（通过健康维度链路构建的逻辑连接）并行执行。实现时需要离线预计算故障环上的 FoldedRing 通信路径表，该路径表取决于故障位置但不受数据量影响（可复用）。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

## MATE/MATEe Scheduling（多维度加速调度）

术语是什么？
MATE（Multi-dimensional Acceleration for Torus with Error）是针对 N-D torus 上链路故障场景的容错 All-to-All 多维调度方法。核心思想：利用 torus 的多维正交特性——其他维度的健康链路可在不冲突的前提下构建故障环上相邻节点的双向逻辑连接，使故障环也能使用 HalfRing 执行高效数据传输。MATE 将通信拆分为正常 phase（仅 FoldedRing 或跳过）+ 加速 phase（利用逻辑连接执行 HalfRing）。MATEe（增强版）在正常 phase 也传输部分数据（按 HalfRing/FoldedRing 性能比静态分配），减少加速 phase 数据量。

从kernel调度角度拆解术语：
MATE 调度结构（2N 个 phase，N=维度数）：

```
Procedure MATE_Scheduler(S, N, D_fault, Torus, Link, mode):
    Chunk_Num = N
    Chunk_Size = S / N
    for chunk = 0 to Chunk_Num - 1:
        for phase = 0 to 2N - 1:
            if phase % 2 == 0:             // 正常 phase
                p = phase / 2
                dim = (chunk + p) % N
                if dim != D_fault:
                    Schedule[chunk][phase] = HalfRing_Gen(Torus[dim], Link)
                else:
                    if mode == MATE:
                        Schedule[chunk][phase] = None  // 跳过
                    else:  // MATEe
                        fraction = perf_ratio(HalfRing, FoldedRing)
                        Schedule[chunk][phase] = FoldedRing_Gen(Torus[dim], Link)
                        // (仅传输 fraction 比例的数据)
            else:                           // 加速 phase
                planes = GetAvailPlanes(D_fault)
                Schedule[chunk][phase] = HalfRing_Planes(planes, Torus, Link)
                // + FoldedRing_Gen(Torus[D_fault], Link) [未在正常 phase 传输的剩余数据]
```

加速原理（2D torus 例，X-dim 故障）：Y-dim 链路将故障 X-dim 环上的相邻节点（如 (0,1)-(0,2)-(1,2)-(1,1)）连接为一组双向逻辑 X-dim 链路，共可构建 N-1 组（每个 Y-dim 平面一组）。加速 phase 在这些逻辑链路上执行 HalfRing，将故障环通信"卸载"到健康环。

术语一般如何实现？如何使用？
MATE 需要离线性能分析以分配各加速平面的数据量（确保并发传输时间一致）。MATE 可处理更复杂故障场景：(1) 同环多故障——FoldedRing 失败，所有通信通过加速 phase 完成；(2) 多环各一故障——为每个故障分配独立加速 phase，无链路冲突时可并行；(3) 异维各一故障——各自独立加速 phase。在 2D torus 上，MATE 性能为 fault-free baseline 的 1.36×（超过 fault-free baseline 1.0×）。MATEe 在小数据量下性能不如 MATE（因静态分配未考虑启动时间差异），但在大数据量下性能更优（加速 phase 数据量更少）。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

## Store-and-Forward in Collective Communication（集合通信中的存储-转发）

术语是什么？
Store-and-Forward 是集合通信中一种逐跳数据传输机制。与硬件路由的多跳直连传输不同，Store-and-Forward 将每个多跳传输分解为多个单跳步骤：中间节点在接收到数据后暂存于本地端点缓冲区，再转发到下一跳。这种细粒度编排方式允许显式控制每步的链路分配，从而完全消除网络拥塞。

从kernel调度角度拆解术语：
以 4 节点环上 Ring 算法 Stage 3（3 跳传输）为例：
```
// 节点 1 发送紫色数据块到节点 4（逆时针 3 跳 = 顺时针 1 跳，但 Ring 固定双向发送）
// 顺时针方向（1→2→3→4，3 跳）:
Sub-stage 3-1: node1 → node2 (Fwd)  // 节点 1 转发给节点 2
Sub-stage 3-2: node2 → node3 (Fwd)  // 节点 2 存储后转发给节点 3
Sub-stage 3-3: node3 → node4 (Fwd)  // 节点 3 存储后转发给节点 4
// 每个子阶段：所有活跃节点对该跳的单跳传输同时执行；无链路共享→零拥塞
```

术语一般如何实现？如何使用？
在本文中，HalfRing/FoldedRing/MATE 均基于 Store-and-Forward 构建通信时间表——算法生成器离线计算每个阶段/子阶段的所有单跳传输对，运行时逐子阶段执行。实现中，store 操作将数据写入接收节点的中间缓冲区，forward 操作从该缓冲区读取并发送到下一跳。相比硬件路由（依赖交换机/路由器处理拥塞控制），Store-and-Forward 在 torus 等直连拓扑上可避免多跳传输的链路争用问题。代价是中间节点的存储开销和额外的转发延迟（每跳增加 α 传播延迟）。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

## Circular Buffer KV Shard Management（循环缓冲区KV分片管理）

术语是什么？
Circular Buffer KV Shard Management 是 PiKV 提出的 Expert-Sharded KV Storage 中的 per-shard 内存管理机制。每个 GPU shard 维护固定容量 S 的 circular buffer（循环缓冲区），通过 hash 函数 s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 将 KV 条目映射到唯一的 (shard, position)，实现 O(1) 时间插入和 O(1) 时间查找。Circular buffer 的固定容量保证无内存碎片和 reallocation，代价是 buffer 满时需驱逐最旧条目（FIFO 语义）。Buffer 大小 S 可根据 PiKV 的 closed-form 优化公式选择：S* = √(L/(KG))，在 sharding granularity 与 reuse coverage 之间求最优平衡。

从kernel调度角度拆解术语：
Circular Buffer KV Shard 的 kernel 级操作：

```
# === Data Structures (per GPU shard) ===
struct KVShard {
    // Circular buffer: fixed-size array + head pointer
    KeyBuffer   K_buf[S][d'];    // S slots, d' compressed dim
    ValueBuffer V_buf[S][d'];    
    Metadata    meta[S];          // {token_id, expert_id, timestamp, ...}
    uint32_t    head;             // next write position (0 ≤ head < S)
    uint32_t    count;            // current valid entries
};

# === O(1) Insert Kernel ===
def kv_shard_insert(shard, K_compressed, V_compressed, token_id, expert):
    pos = shard.head
    # Overwrite oldest entry (FIFO eviction)
    shard.K_buf[pos] = K_compressed        # O(d') memcpy
    shard.V_buf[pos] = V_compressed
    shard.meta[pos] = {token_id, expert, now()}
    shard.head = (pos + 1) % S              # advance head
    shard.count = min(S, shard.count + 1)

# === O(1) Lookup (by position) ===
def kv_shard_lookup(shard, token_id):
    # Hash token_id to buffer position
    pos = token_id % S
    # Verify metadata match (handle hash collisions)
    if shard.meta[pos].token_id == token_id:
        return (shard.K_buf[pos], shard.V_buf[pos])
    else:
        return CACHE_MISS

# === O(S) Scan for Query-Aware Selection ===
def kv_shard_select_pages(shard, q_t, threshold):
    selected = []
    for pos in range(shard.count):
        # Compute utility score for each cached entry
        u = compute_utility(
            shard.K_buf[pos], shard.V_buf[pos],
            shard.meta[pos],
            q_t
        )
        if u >= threshold:
            selected.append((pos, u, shard.meta[pos]))
    # Return top-K by utility
    return sorted(selected, key=lambda x: x[1], reverse=True)[:K]
```

**Buffer 大小优化**：
$$S^* = \sqrt{\frac{L}{KG}}$$
- L 过大 → S* 增大（更多 slot 支持更大 token 容量）
- K 增大 → S* 减小（调度器保留更多 page，减少 per-shard buffer 需求）
- G 增大 → S* 减小（更多 GPU 分担，per-GPU buffer 可缩小）

**Per-GPU Memory Formula**：
$$\mathcal{M}_{\text{total}} = \frac{2d}{\rho}\left(\frac{L}{GS} + KS\right)$$

术语一般如何实现？如何使用？
- PiKV CUDA 实现：`core/cuda/pikv_cuda.py` 中的 `moe_routing` 和 `top_k_experts` CUDA kernel 负责 KV tensor 的 gather/scatter 操作，circular buffer 逻辑在 `core/single/kvcache_centric_system.py` 的 PagedKVCache 中由 Python 管理。
- 与 vLLM block table 的关系：vLLM 的 physical block table 是动态分配的（需要 free list + allocation），PiKV 的 circular buffer 是固定大小（无 allocation 开销），适合 MoE 的 per-expert per-shard 细粒度 KV 存储。
- CUDA kernel 加速：`pikv_cuda.moe_routing(tokens, expert_weights)` → GPU parallel gather of KV from shards, `pikv_cuda.top_k_experts(scores, k)` → warp-level top-k on routing scores。
- 适用场景：MoE 推理的 per-shard KV 存储（S 典型值 256-512），要求低延迟 O(1) 操作的 streaming workload。不适合需要复杂驱逐策略的 workload（此时用 PiKV Scheduling 的多特征评分选择 active pages）。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts

## Backward Stream Scheduling in MoE Training (MoE训练反向流调度 / All-to-All Prioritization)

术语是什么？
Backward Stream Scheduling 是 PopFetcher 提出的一种 MoE 训练 backward pass 中的 CUDA stream 优先级调度机制。由于 expert prefetching 导致同一 expert 可能存在多个副本在不同 worker，backward pass 中需要额外的 All-Reduce 操作来聚合 prefetched expert 的梯度。同时存在三种 CUDA stream：(1) EP All-to-All（token 回传）；(2) non-MoE All-Reduce（梯度聚合）；(3) prefetched expert All-Reduce（副本梯度回主 expert）。在 NCCL 中通信原语在 CUDA stream 被锁定在调用点，无法实时调整优先级。PopFetcher 将 All-to-All 和 All-Reduce 通信分解为 micro-operations 交替流水线执行，All-to-All 优先级高于 All-Reduce。

从kernel调度角度拆解术语：
Backward pass 的 micro-operation 流水线调度：
```
// 传统的 backward pass stream 争抢问题：
// Stream1: All-to-All (token 回传)
// Stream2: All-Reduce (non-MoE gradient)
// Stream3: All-Reduce (prefetched expert gradient)
// 问题：三个 stream 并发竞争 NCCL bandwidth → All-to-All 被非关键 All-Reduce 阻塞

// PopFetcher 的 micro-operation pipelining:
stream_priority = {A2A: HIGH, AR_NON_MOE: MEDIUM, AR_PREFETCH: LOW}

// 将 All-to-All 分解为 micro-ops
a2a_micro_ops = split_into_chunks(all_to_all_data, chunk_size)
// 将 All-Reduce 分解为 micro-ops  
ar_micro_ops = split_into_chunks(all_reduce_data, chunk_size)

// 交错执行，A2A 优先
while a2a_micro_ops or ar_micro_ops:
    if a2a_micro_ops:
        execute_next(a2a_micro_ops)     // 优先 All-to-All
    elif ar_micro_ops:
        execute_next(ar_micro_ops)      // All-to-All 完成后再 All-Reduce
```

效果：减少 backward computation blockage 10.9%（MoE-GPT）、10%（MoE-BERT）。核心原理是保证 token 回传不被 gradient 聚合阻塞——token 回传是下一层计算的依赖，而 gradient 聚合仅影响权重更新（可稍延迟）。

术语一般如何实现？如何使用？
在 C++ 和 CUDA 中实现 pipeline scheduling：将通信操作分解为微操作后，通过 CUDA event 和 stream 管理执行顺序。NCCL 层面无法直接修改优先级，因此通过在应用层控制 micro-operation 的提交顺序来实现效果。适用于 MoE 训练中同时存在 All-to-All 和 All-Reduce 的场景，尤其在采用了 expert prefetching/replication 后 prefetched expert 的额外 All-Reduce 会加剧 stream 竞争。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch

## Expert Prefetch Pruning (专家预取剪枝)

术语是什么？
Expert Prefetch Pruning 是 PopFetcher 提出的优化 expert prefetching 决策空间的方法。由于大集群中 global expert 数量巨大（如 256 GPU × 128 experts/GPU = 32768 experts），每 worker 决策是否预取每个 remote expert（δ_{n,w}^i ∈ {0,1}）的搜索空间为指数级。PopFetcher 通过两重约束剪枝：(1) GPU memory limitation：预取 expert 总参数量 ≤ 可用 GPU memory；(2) Transfer time constraint：预取传输时间 ≤ 非 MoE 计算时间；(3) Popularity filtering：仅 top-k×N 个热门 expert 进入候选集；(4) Efficiency threshold：仅当 ε = P_w/W_{n,w} > 3αH 且 B_{n,w}^i > εαH/2(ε-3αH) 时该 expert 才值得预取。

从kernel调度角度拆解术语：
剪枝与预取决策的数学过程：
```
Input: expert_popularity[p_w^i], gpu_memory[Mem_w^free], bandwidth[W_{n,w}], compute[P_w]
Output: prefetch_plan[δ_{n,w}^i]

// Step 1: Popularity filtering
candidates = top_popularity(experts, k × N)  // 至多 top-k × N 个 expert

// Step 2: Efficiency threshold (Eq. 13)
for each expert E_n^i in candidates:
    ε = P_w / W_{n,w}
    if ε <= 3αH:
        skip                                    // 带宽充足时不值得 prefetch
    threshold = εαH / (2(ε - 3αH))
    if B_{n,w}^i <= threshold:
        skip                                    // 接收 token 太少不划算

// Step 3: Memory + time constraint (Eq. 8-10)
valid = []
for each remaining expert (sorted by popularity desc):
    prefetch_size = 2αH² / W_{n,w}
    if (total_prefetch + prefetch_size <= Mem_w^free) AND
       (total_transfer_time + prefetch_size <= Time^{non-MoE}):
        valid.append(expert)
        total_prefetch += prefetch_size

// Step 4: Solve min-max latency (Eq. 7)
δ^* = argmin_δ max_w Lat_w^{prefetch}(δ)
// 在中后期训练中可固定 δ^* 或降低 replanning 频率
```

术语一般如何实现？如何使用？
实现为 CPU 异步执行的 decision-maker 模块（Python），在 GPU 训练期间后台运行。popularity prediction 通过 All-Gather 聚合各 worker 的 per-expert token 计数（小向量，sync 开销 negligible < 100ms）。剪枝后搜索空间从指数级降至可穷举/贪心求解规模。中后期训练可利用 expert 分布的稳定性降低 replanning 频率。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch

## Non-MoE Computation Overlap (非MoE计算重叠)

术语是什么？
在 MoE 训练中，Non-MoE Computation Overlap 指利用 Transformer block 中非 MoE 组件（如 Attention 层、LayerNorm）的执行时间窗口，将通信或辅助计算与之重叠执行，从而隐藏延迟。PopFetcher 利用 Attention 层计算期间 network link idle 的特点，在此期间异步预取下一 MoE layer 的热门 expert 参数。Attention 层仅使用本地数据（无跨机通信），因此其执行期间 100% 的 network bandwidth 可用于 expert prefetching。

从kernel调度角度拆解术语：
Overlap 的 timeline 调度：
```
// 一个 Transformer block 的执行 timeline（单个 GPU worker）：
Time → 
[Attention Forward]  [MoE Forward: A2A Dispatch → Expert FFN → A2A Combine]  [Attention Backward]  [MoE Backward]
|<-- Non-MoE -->|    |<------------------- MoE ------------------->|       |<-- Non-MoE -->|    |<--- MoE --->|

// PopFetcher 的 overlap 策略：
// 在 Non-MoE 期间的 idle network link 上：
[Attention Forward + Expert Prefetch(l+1)]  [MoE Forward(l) with prefetched experts(l)]
[Attention Backward + Expert Prefetch(l+1)] [MoE Backward(l) with stream pipelining]
```

Overlap 条件：Time^{non-MoE} ≥ Σ expert_prefetch_time，即 Attention 计算时间必须覆盖所有需要预取的 expert 参数的总传输时间。当 bandwidth 有限（compute-to-bandwidth ratio ε 高）时，Attention 计算时间相对充裕，overlap 最有效。

术语一般如何实现？如何使用？
基于 PyTorch CUDA stream 管理：主训练 stream 执行 Attention forward/backward，独立 prefetch stream（torch.cuda.Stream）执行 P2P expert 参数传输。通过 CUDA event 同步确保预取在下一 MoE layer 开始前完成。适用于所有 MoE 训练框架，只要 EP 下 Attention 层计算期间 network link idle。在 Cluster B (8×A10, 32Gbps) 的 bandwidth-constrained 环境下收益尤为显著（加速比 1.18-18.3×）。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch

## Warp-per-Row GPU Dictionary Decoding Kernel (Sub1MatVec / Warp级按行字典解码GPU内核)

术语是什么？
Warp-per-Row Dictionary Decoding Kernel 是 QMoE 设计的自定义 CUDA kernel（命名 Sub1MatVec），用于执行压缩权重矩阵与输入向量的融合解压缩-矩阵乘运算。核心并行策略：每个 GPU warp (32 threads) 处理权重矩阵的独立一行，使用 28/32 threads 同时从字典中执行解码，4 threads 不参与（处理数据格式非均匀性）。该设计的关键约束和决策：(a) UINT16 codewords 映射到 2×UINT32 数据（64-bit），每半 UINT32 由 14 threads 独立解码；(b) 字典 512KB 驻留 GPU L2 cache，按概率降序排列实现 L1 prefetch；(c) ternary dequant 通过 shared memory lookup table 实现，复制 32× 避免 bank conflict。

从kernel调度角度拆解术语：
**Sub1MatVec Kernel 伪代码（简化）**：
```
template<int num_warps, int w_width>
__global__ void Sub1MatVec(
    int* dec,           // 字典 [2^16 * 2] UINT32
    ushort* w_comp,     // 压缩权重 [total_codewords] UINT16
    int* row_off,       // 行偏移 [num_rows+1]
    __nv_bfloat162* ter_minmax,  // 每行 {w_min, w_max}
    __nv_bfloat16* x,   // 输入向量
    __nv_bfloat16* y)   // 输出向量
{
    // === 1. Shared Memory 初始化（全 threadblock 协作）===
    __shared__ float x_shared[w_width];
    for (int i = thread; i < w_width; i += 32*num_warps)
        x_shared[i] = bfloat162float(x[i]);  // 加载输入向量
    
    // === 2. 构建 Ternary Dequant 表（每 warp 独立）===
    __shared__ float deq[3][32 * num_warps];  // 复制 32× 避免 bank conflict
    deq[0][thread] = 0;
    deq[1][thread] = __bfloat162float(ter_minmax[row].x);  // w_min
    deq[2][thread] = __bfloat162float(ter_minmax[row].y);  // w_max
    __syncthreads();
    
    // === 3. 每行独立解码（warp-per-row）===
    __shared__ ushort w_comp_block[32][num_warps];
    int idx = 0;  // 当前输入向量偏移
    float res = 0;  // per-thread 累加器
    
    for (int i = 0; i < row_off[row+1] - row_off[row]; i += 32) {
        // 3a. Coalesced load: 32 UINT16 codewords → shared memory
        w_comp_block[warp][lane] = w_comp[i + lane];
        
        // 3b. 28 threads (lane 0-27) 解码
        if (lane < 28) {
            for (int j = 0; j < 32; j++) {
                int enc = w_comp_block[warp][j];  // UINT16 codeword
                // 线程 0-13 取 UINT32[0]，线程 14-27 取 UINT32[1]
                int wx14 = dec[2 * enc + (lane / 14)];
                // 提取 2-bit ternary 值（shift + mask，无慢速 modulo）
                int ter = (wx14 >> (4 + 2 * (lane % 14))) & 0x3;
                // Dequant via shared memory lookup（无 bank conflict）
                float w = deq[ter][thread];
                // FMA: 连续 shared memory 读（无 bank conflict）
                res += w * x_shared[idx + lane];
            }
            // 偏移推进 = 解码的权重总数（pair_count × 2）
            idx += 2 * (wx14 & 0xf);  // pair_count 存于低 4 bits
        }
    }
    
    // === 4. Warp Reduction ===
    res = warp_reduce_sum(res);  // warp shuffle sum
    if (lane == 0) y[row] = float2bfloat16(res);
}
```
关键设计：每 threadblock 占 1 SM（避免 wave quantization）；超过 32 行时 warp 串行处理多行但每行仍独立；decoding bit ops (~1 cycle) vs global memory read (~200 cycles) → 解码被访存完全隐藏；coalesced memory access (line 3a) + contiguous shared memory reads (line 3b FMA) = 接近内存带宽利用率上限。

术语一般如何实现？如何使用？
- 实现：QMoE 源码 https://github.com/ISTDASLab/qmoe（CUDA kernel 完整版含所有边界条件处理）
- 性能：所有 MoE 矩阵形状下比 cuBLAS bfloat16 GEMV 更快（最高 35% speedup），因压缩后 global memory 读取量仅 ~1/20
- 适用：压缩 MoE 模型的 decode 阶段（memory-bound GEMV）。prefill 阶段（batch >1）建议先完全解压再使用 cuBLAS GEMM
- 限制：当前实现 naively 对同一 expert 的多个 token 分别执行独立 matvec（vs baseline 的 batched matmul 更高效）；可扩展为 kernel 内 token batching

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models

## Fixed-to-Variable Dictionary Code for GPU Decoding（面向GPU解码的定长到可变字典编码）

术语是什么？
Fixed-to-Variable Dictionary Code 是 QMoE 选择的压缩编码方案，与传统的 variable-to-fixed 熵编码（如 Huffman：变长 codeword → 固定长度符号）相反，采用 fixed-length codeword (UINT16) → variable-length sequence of symbols（最多 28 个三元权重）。这种 LZW-style 编码的选择完全由 GPU 解码的硬件约束驱动：(1) 固定长度 codewords 消除变长码的序列解码依赖——每个 codeword 可独立并行查表解码；(2) 一个 warp 的全 32 threads 可联合处理一个 codeword——解决了"二进制字包含不同数量符号导致 warp divergence"的问题；(3) UINT16 codeword 无需慢速 bit-extraction（vs 变长码的 bit-level 操作）。

从kernel调度角度拆解术语：
**为何不用 Huffman（变长编码）？GPU 解码的三个致命挑战**：
```
Challenge 1: 序列依赖
  Huffman: symbol_i 的起始位需要知道前 i-1 个 symbol 的变长码字长度
  → 无法并行解码连续 symbol
  QMoE: codeword_i 独立，32 threads 同时查表解码

Challenge 2: Warp Divergence
  Huffman: 每个二进制字(INT32)可能包含不同数量的解码 symbol
  → 不同线程解码不同数量 symbol → warp divergence → 大量浪费操作
  QMoE: 固定 16-bit codeword, 28 threads 处理 1 codeword/cycle
  → warp 内均匀，无 divergence

Challenge 3: Bit Operations
  Huffman: 大量 bit shifts, masks, variable-length reads
  → GPU 的 bit ops 慢且不友好（vs CPU/ASIC）
  QMoE: 仅 2-bit shift+mask 从 UINT32 提取 ternary 值
  → 极少量 bit ops, 被 memory latency 完全隐藏
```

字典格式设计（硬件约束驱动）：
```
// Codeword: UINT16 (2^16 = 65536 个条目)
// 每个条目映射到 2×UINT32 (64 bit):
//   UINT32[0] → threads 0-13 使用
//   UINT32[1] → threads 14-27 使用
//   每 UINT32 格式:
//     bits[27:0]: 14×2-bit ternary values
//     bits[31:28]: pair_count (0-14)
//   pair_count 存两次确保每半可独立解码
// 总计: 2^16 × 8 bytes = 512KB → GPU L2 cache resident
```
选择 14 对（28 权重）作为最大序列长度：4 bits 存 pair_count（0-14），恰好 fit UINT32 低 4 bits。

术语一般如何实现？如何使用？
- 字典生成：Algorithm 1 (max-priority queue)，以三元值对概率为优先级贪心扩展最高概率序列
- 全局字典：一个字典服务所有 MoE 层/expert（避免 per-expert 存储开销）
- 压缩率 vs 理论极限：20.07× (c2048) vs 25.40× (Shannon 极限, p0=0.886) → ~20% 差距，换取 GPU 快速解码
- 字典按概率降序排列 → 高频 codewords 更可能已在 L1 cache → 自动 prefetch

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models

## Shared Memory Ternary Dequantization Lookup Table（共享内存三元解量化查找表）

术语是什么？
Shared Memory Ternary Dequantization Lookup Table 是 QMoE 在 Sub1MatVec kernel 中使用的一种 GPU shared memory 数据结构，用于将 2-bit 编码的三元值 {0, 1, 2} 快速转换为可计算的浮点权重 {0, w_min, w_max}。核心设计：deq[3][32*num_warps] 的二维 shared memory 数组，其中每个浮点值在列方向（warp dimension）复制 32 次以避免 bank conflict。原因：28 threads 同时执行 deq[ter][thread] 查找，若不复制，不同 threads 访问不同 ter 值（0/1/2）时 smem 的 bank 冲突（同一 bank 被多个 thread 同时访问）会频繁发生；复制 32× 后，thread i 永远访问 deq[ter][i]——每 thread 独占一列，列内连续 32 个 float 分布在 32 banks，无冲突。

从kernel调度角度拆解术语：
```
// 初始化（每 warp 独立，在 kernel 头部执行一次）
// 3 行 × (32 * num_warps) 列，每列 32 个 float 分布在 32 banks
deq[0][thread] = 0;                                        // ter=0 → 0
deq[1][thread] = __bfloat162float(ter_minmax[row].x);     // ter=1 → w_min
deq[2][thread] = __bfloat162float(ter_minmax[row].y);     // ter=2 → w_max

// 运行时查表（per decoded weight）
int ter = (wx14 >> (4 + 2 * (lane % 14))) & 0x3;  // 提取 2-bit ternary index
float w = deq[ter][thread];  // 查表；每 thread 列独立 → 0 bank conflict
```

为何不用 register 或 constant memory？
- Register: 每个 thread 需存 3 个 float = 太少，不影响；但同一 warp 内不同 threads 的 ter 值可能不同——register 无法"共享"访问
- Constant memory: 读延迟 ~constant cache hit (few cycles) / miss (~L1/L2); 但 28 threads 读同一地址 → broadcast 机制可用，但不如 smem 可控
- Shared memory: 1 cycle latency，无 bank conflict (复制 32× 后)，确定性最低延迟
- 代价：deq table 占用 3 × 32 × num_warps × 4 bytes = 384 × num_warps bytes smem——以 num_warps=4 为例仅 1.5KB，可忽略

术语一般如何实现？如何使用？
- QMoE CUDA kernel 源码中实现，配合 `__syncthreads()` 确保所有 warps 完成 deq 初始化后才开始解码
- 通用技术：任何需要在 GPU warp 内频繁执行 small lookup table 操作的 kernel 均可使用（如 low-bit dequantization）
- 限制：仅当 lookup table 极小（≤ few KB）时才实用——每复制 32× 内存乘 32

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models

## mma.sp PTX Instruction (Sparse Matrix Multiply-Accumulate PTX 指令)

术语是什么？
`mma.sp` 是 NVIDIA PTX ISA 中用于调用 Sparse Tensor Core（SpTC）的 warp-level matrix multiply-accumulate 指令，从 PTX ISA 7.0（CUDA 11.0+，SM80 Ampere）开始可用。它允许程序员在 CUDA inline PTX assembly 中直接触发 SpTC 硬件执行 2:4 结构化稀疏矩阵乘法。与标准 `mma.sync` 指令的关键区别在于：(1) A 操作数必须是压缩稀疏格式（data+metadata pair）；(2) metadata 作为独立操作数传入，编码每 4 元素组中 non-zero 值的位置；(3) B 操作数必须是密集矩阵；(4) 计算吞吐为同 shape 密集指令的 2×。

从 kernel 调度角度拆解术语：
`mma.sp` 在 Samoyeds kernel 中的使用流程伪代码：

```
// Samoyeds kernel 中 mma.sp 的使用
// 场景：C[m][n] += A_sparse[m][k] × B_sparse[k][n]
// A 已编码为 data[m][k/2] + metadata[m][k/2] (2-bit each)
// B 通过 SEL 选择有效列，packed in transposed format

// Step 1: 从 SMEM 加载数据到寄存器（使用 ldmatrix）
// ldmatrix 按 SpTC spec 排列：每个线程持有 A 的特定片段
asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(smem_addr));

// Step 2: 加载 metadata 到寄存器
// 每个 32-bit 寄存器包含 16 个 2-bit metadata（Samoyeds 自定义 packing）
ld_metadata_to_reg(metadata_reg, metadata_smem_addr);

// Step 3: 调用 mma.sp 执行稀疏 MMA
// m16n8k32: M=16 rows of A, N=8 cols of B, K=32 (50% sparse → 16 effective)
asm volatile(
    "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, "     // D = C registers (4×f32 = 16×8×f32/32-threads)
    "{%4,%5,%6,%7}, "     // A = compressed sparse data (4×f16×2)
    "{%8,%9}, "           // B = dense matrix fragment (2×f16×2)
    "{%10,%11,%12,%13}, " // C = accumulator (4×f32)
    "%14;"                // metadata operand for A sparsity
    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
    : "r"(a0_reg),"r"(a1_reg),"r"(a2_reg),"r"(a3_reg),
      "r"(b0_reg),"r"(b1_reg),
      "f"(c0_init),"f"(c1_init),"f"(c2_init),"f"(c3_init),
      "r"(metadata_reg)
);

// Step 4: Data stationary shuffle（跨越 Sub-Row 边界时）
// 每 V/k_h 次迭代，按 indices 矩阵 shuffle C 寄存器
if (compute_iter % (V / k_h) == 0) {
    // C 寄存器重新映射到正确的行
    shuffle_C_registers(C_regs, indices);
}
```

Samoyeds kernel 中 `mma.sp` 的调度关键点：
1. **Tiling 对齐**：innermost tile `(m_i, k_i, n_i)` 必须满足 mma.sp 指令形状——`m16n8k32`（FP16→FP32）或 `m16n8k16`（更小 tile，延迟更低但吞吐不如 k32）。
2. **Pipeline overlap**：fetch stage（cp.async GMEM→SMEM）与 compute stage（ldmatrix + mma.sp）通过 CUDA pipeline group 机制 overlap。
3. **Metadata packing**：2-bit metadata 需自定义 packing 方案（Samoyeds 将 16×16 metadata 子矩阵映射为 32-bit 对齐的 memory transaction），否则无法配合 ldmatrix 使用。
4. **Data stationary**：C 保持于寄存器跨越多轮 compute iteration；仅在 Sub-Row 边界处 shuffle。

术语一般如何实现？如何使用？
- **支持架构**：SM80+（Ampere A100/A30/RTX 3090 等）、SM89（Ada Lovelace RTX 4070/4090）、SM90（Hopper H100，使用 `wgmma.mma_async.sp` 替代）。AMD CDNA3 有等效指令但语法不同。
- **支持数据类型**：FP16×FP16→FP32、BF16×BF16→FP32、TF32×TF32→FP32、INT8×INT8→INT32。不支持 FP8（Hopper 新增但 mma.sp 尚无 FP8 变体）。
- **约束**：(1) A 操作数需 16-byte alignment（`layout.aligned`）；(2) 每个 warp（32 线程）协作完成一条 mma.sp 指令，各线程持有矩阵的部分 fragment；(3) metadata 格式为每 4 元素 2-bit，每 32-bit 寄存器存 16 个 2-bit 向量；(4) 稀疏仅支持 A 侧（Ampere），Hopper 后扩展但仍是 A-only。
- **集成方式**：可直接在 .cu 文件中用 inline PTX assembly 调用，或通过 CUTLASS/cuSPARSELt 等高层库间接使用。Samoyeds 的 kernel 用 NVCC 编译为 .so，通过 pybind11 注册为 Python module。

涉及论文标题：
- Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

## VENOM V:N:M Sparse Format

术语是什么？
VENOM（Vectorized N:M format）是 Castro et al.（SC '23）提出的一种灵活结构化稀疏数据格式，旨在利用 NVIDIA SpTC 硬件加速稀疏-密集矩阵乘法，同时突破 cuSPARSELt 固定 50%（2:4）稀疏比的限制。VENOM 的 V:N:M 格式中：M 是 block 大小（列维度），N 是在 block 中保留的 vector 数量，V 是每个 vector 的 element 数量。通过调整 N 和 V，VENOM 支持从 50% 到 90%+ 的灵活稀疏比，而仍然利用 SpTC 的 2:4 硬件加速。其核心技巧是将不同 N:M 比例的向量组合映射到 2:4 pattern，在 metadata 层面"欺骗" SpTC 选择器以正确处理非标准稀疏模式。

从 kernel 调度角度拆解术语：
VENOM 的 V:N:M 编码和 SpTC 映射机制：

```
// VENOM 编码示例: V=2, N=1, M=4 → 75% sparsity
// 原始矩阵 (M=4 columns per block, V=2 elements per vector):
// Column:   0  1  2  3
// Row 0:    a  .  .  .   (非零值 a 在 col 0)
// Row 1:    b  .  .  .   (非零值 b 在 col 0)
// Row 2:    .  .  c  .   (非零值 c 在 col 2)
// ...
// 每 2 行 × 4 列 block 中仅保留 1 个 vector（N=1），每个 vector 含 2 个元素（V=2）
// 有效稀疏比 = 1 - (1×2)/(2×4) = 1 - 2/8 = 75%

// VENOM 编码为 2:4 compatible 格式：
// 将多个 V:N:M block 的 non-zero vectors 拼接成符合 2:4 的 dense rows
// metadata 记录 "原始 column → packed column" 的映射
// SpTC 执行时通过 metadata selection 从 dense B 中选择正确的列参与计算

// VENOM sparse-dense matmul 伪代码：
for each thread block (tile of C[m_b][n_b]):
    for k_tile = 0 to K step K_b:
        load VENOM_A[m_b][K_b] from GMEM → SMEM → register
        load dense_B[K_b][n_b] from GMEM → SMEM → register
        
        for each V:N:M block in A:
            sel_cols = metadata.indices  // 哪些列是活跃的
            
        // Execute via mma.sp (SpTC 利用 metadata 选择 B 列)
        mma.sp(C_tile, A_packed, B_dense[sel_cols], metadata)
```

**VENOM 的关键局限（Samoyeds 论文 Figure 6 揭示）**：
当输入矩阵 B 也是稀疏的（如 MoE token routing），VENOM 的 sparse-dense 设计暴露出三类问题：
1. **I/O amplification（格式②③）**：跳过稀疏 weight column 时，若该 column 对应的 input row 也是稀疏列，则可能加载了不需要的 input 数据或跳过了需要的 weight 行的 input 数据。
2. **Uncoalesced memory access（格式④）**：稀疏 column 导致数据在内存中不连续，GPU 无法 coalesce memory transaction，带宽利用率下降。
3. **Small tile fragmentation**：稀疏 pattern 将数据打散为小 tile，降低 warp 利用率。

这正是 Samoyeds 提出 dual-side sparse format 的动机——解决 VENOM 在"权重稀疏 + 输入稀疏"场景下的退化问题。

术语一般如何实现？如何使用？
- VENOM 代码开源（SC '23 artifact），与 cuSPARSELt 对比，支持灵活 sparse ratio（50%~90%+），在 SC '23 基准上取得 1.38× 加速 over cuSPARSELt。
- VENOM 的适用场景：单端权重稀疏的推理（输入 dense），如传统 LLM 的 FP16 推理。但在 MoE（输入天然稀疏）下性能退化。
- Samoyeds 的改进：(1) 双端稀疏格式——输入端也采用 vector-wise 稀疏 by SEL array；(2) 专门的 sparse-sparse kernel 而非 sparse-dense；(3) customized packing 和 data stationary 避免 VENOM 的 I/O amplification 问题。Samoyeds 在 kernel 级 up to 1.99× vs VENOM，模型级 up to 1.58× vs vLLM。

涉及论文标题：
- Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

## Adaptive All-to-All Communication（自适应全交换通信）

术语是什么？
Adaptive All-to-All Communication 是 ScaleMoE 论文提出的 MoE 分布式训练通信优化技术。传统 MoE 训练框架（DeepSpeed, Tutel）在 expert parallelism 的 all-to-all 通信中，为统一各 GPU 的 message size 而大量使用 zero padding——由于 expert selection 高度不均衡，zero ratio 从训练初期的 88% 迅速升至 98%，导致通信量严重膨胀。Adaptive All-to-All 在运行时监控每个 GPU 的 per-expert token 选择计数，通过一次 all-gather 操作聚合所有 GPU 的计数信息，计算精确的 input slice（第 i 列发给 GPU-i 的 token 数）和 output slice（第 j 行从 GPU-j 接收的输出数），然后使用 NCCL alltoallv 仅传输有效数据，消除所有 zero padding。all-gather 的额外通信开销（44.50ms）相对于被消除的 GB 级 zero 传输可忽略不计。

从kernel调度角度拆解术语：
Adaptive All-to-All 在每个 MoE 层的通信 kernel 调度流程：
```
// 4 GPUs, 4 experts, 每 GPU 10 tokens 为例

// Step 1: Monitoring（本地 GPU kernel）
GPU-1: expert_counts = {E1:4, E2:1, E3:3, E4:2}
GPU-2: expert_counts = {E1:2, E2:6, E3:1, E4:1}
GPU-3: expert_counts = {E1:0, E2:3, E3:7, E4:0}
GPU-4: expert_counts = {E1:1, E2:0, E3:2, E4:7}

// Step 2: All-gather counts（通信 kernel, overhead 44.50ms）
global_counts = all_gather([GPU-1_counts, ..., GPU-4_counts])
// 构建 dispatch matrix: dispatch[i][j] = GPU-i 发往 GPU-j 的 token 数

// Step 3: Adaptive All-to-All dispatch（通信 kernel）
// NCCL alltoallv: 每个 send/recv buffer 大小不同
for each GPU i:
    for each target GPU j:
        send tokens to GPU-j with size = dispatch[i][j]  // 精确大小，无 zero pad

// Step 4: Expert FFN（计算 kernel）

// Step 5: Adaptive All-to-All combine（通信 kernel，对称反向）

// 对比 Baseline: 所有 GPU 按 max(dispatch[i][j] ∀i,j) 统一 buffer size
// e.g., max=7 → GPU-1 发往 GPU-2 仅 1 token 却传输 7-token 等价数据量
// Adaptive: GPU-1→GPU-2 仅传输 1 token 的数据
```

术语一般如何实现？如何使用？
实现于 PyTorch v2.0 + DeepSpeed。在 DeepSpeed 的 MoE dispatcher 中 hook 入监控逻辑，在原有的 all-to-all 调用前插入 all-gather 计数步骤，然后用不等长 buffer 的 `torch.distributed.all_to_all` 变体替换原有等长 all-to-all。与 Megatron-LM 的 alltoallv 功能等价但 dispatcher-agnostic（通过 hook 集成，最小化框架修改）。ScaleMoE 在 32×A100 GPU 上评估：all-to-all 通信开销减少 up to 81%，端到端 speedup 1.71-1.84×（homogeneous）和 2.88-3.31×（heterogeneous）。

- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

## Pipelined Expert Processing（流水线专家处理）

术语是什么？
Pipelined Expert Processing 是 ES-MoE (ICML '24) 提出的 expert-level 计算与通信重叠流水线技术。在 MoE training 中，expert 参数 offload 到 CPU 后需要上传到 GPU 才能计算。传统 layer-wise pipeline 在每个 MoE layer 开始前等待所有 experts 上传完毕，导致 GPU 空转。Pipelined Expert Processing 将 pipeline 粒度从 layer 细化到 individual expert：token permutation 阶段与首个 expert 上传重叠（permutation ~0.05ms 足够上传一个 expert），后续 experts 串行处理时并发上传与计算（expert_N computation || expert_{N+1} upload via PCIe）。

从kernel调度角度拆解术语：
```
# ES-MoE Pipelined Expert Processing per MoE Block (Forward)
# 配置：k GPUs, n experts per layer, PCIe bandwidth B

Gate: x → W_gate·x → softmax → Top-1 expert index per token [GPU]
                      |
Dynamic Expert Placement: greedy schedule expert→GPU mapping [CPU, <2.69us]
                      |
Token Permutation: All-to-All scatter tokens to target GPUs [GPU]
  || (overlapped)
  Expert_0 Upload: cudaMemcpy(W_expert_0, CPU→GPU, stream=copy) [PCIe]
                      |
# Expert Processing Loop (per GPU)
for i in 0..num_local_experts:
  if i > 0:
    Expert_i Upload: cudaMemcpy(W_expert_i, CPU→GPU, stream=copy_i)
  Expert_{i-1} FFN: gate_proj → SiLU⊙up_proj → down_proj  [GPU, stream=compute]
  # Expert_{i-1} computation || Expert_i upload — fully overlapped
  
Token Un-permutation: All-to-All gather expert outputs [GPU]
```

术语一般如何实现？如何使用？
ES-MoE 在 Fairseq 框架上实现。使用 PyTorch CUDA stream 管理：独立的 copy stream 和 compute stream。关键参数：PCIe 4.0 bandwidth ~25 GB/s per direction；expert size 85-170 MB per expert；overlap 效果取决于 per-expert compute time vs upload time。MoE-M 32 experts 下，pipelined processing 使 GPU utilization 从 32% 提升至 39%。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Dynamic Expert Placement on GPUs（GPU上的动态专家放置）

术语是什么？
Dynamic Expert Placement 是 ES-MoE 提出的 per-batch expert→GPU 映射调度策略。传统 expert parallelism 中 expert 静态固定在 GPU 上，导致 GPU 间 load imbalance 和大量 zero-padding。ES-MoE 通过 CPU offloading 使 expert 不再固定于 GPU，每个 batch 根据 gating network 输出的 token 分布动态重新分配 expert 到 GPU，最小化 makespan。

从kernel调度角度拆解术语：
贪心调度算法（Graham 1969, 4/3-approximation）：(1) 建模每个 expert 的处理时间 = max(upload_time + compute_time × token_count)；(2) 按处理时间降序排列 experts；(3) 依次将每个 expert 分配给累积负载最小的 GPU。复杂度 O(m log n + m log m)，CPU 执行 < 2.69μs。实验中 GPU 间 token 差异从 102%（Fairseq static）降至 15%（ES-MoE dynamic），完全消除 zero-padding。

术语一般如何实现？如何使用？
实现于 ES-MoE/Fairseq：gating network 执行后 → all-reduce per-expert token counts → CPU 执行 greedy placement → 输出 expert→GPU 映射 → 各 GPU 按映射上传对应 experts。前提是 experts 已 offload 到 CPU。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

SmartMoE 对该术语的扩展：SmartMoE 将 Expert Placement 从单纯的运行时调度策略升格为并行策略空间中的一个独立可搜索维度。提出三种 placement 搜索算法：(1) **Greedy**：O(NE)——按 per-expert token count 降序排列，依次将 expert 分配到累积负载最小的 GPU（限制 per-GPU expert 数 ≤ E/N）；(2) **DP**：O(N×4^E)——状态 F(i,S) 表示前 i 个 GPU 已放置 expert 集合 S 的最小 makespan，保证最优解；(3) **Hybrid**：O(ME + N×4^M)——先 Greedy 将 E 个 expert 分配到 M 个虚拟设备，再 DP 将 M 个虚拟设备分配至 N 个物理设备（M 可调，如 M = GPUs_per_node）。SmartMoE 将 Expert Placement 作为整个并行化框架中唯一在线可变的维度，而 DP/TP/PP 在离线阶段固定。

## Expert Slot（专家槽位）

术语是什么？
Expert Slot 是 SmartMoE 提出的用于统一表达 MoE 模型混合并行策略的核心抽象。它将每个 GPU worker 上的 expert 子网络存储表示为一个"槽位"（slot），通过三个属性统一描述任意 DP/TP/PP/EP 组合：(1) 每个 slot 的容量（capacity）：0 到 1 的分数，表示存储完整 expert 还是部分 expert；(2) 每个 worker 的 slot 数量（#slots）：正整数；(3) 每个 worker 的 MoE 层数（#layers）。例如纯 EP 配置为 capacity=1, #slots=E/N, #layers=L；EP+TP 配置为 capacity=1/T, #slots=T×E/N, #layers=L；EP+PP 配置为 capacity=1, #slots=E/(N/P), #layers=L/P。

从kernel调度角度拆解术语：
以 (L=2 MoE layers, E=4 experts, N=4 GPUs) 为例的 slot 配置：
```
纯 EP:
  GPU_0: [slot0(E0, cap=1), slot1(E1, cap=1)]
  GPU_1: [slot0(E2, cap=1), slot1(E3, cap=1)]
  #slots=1, #layers=2 per GPU（per-layer 各一个 slot）

DP=2 + EP:
  GPU_0: [slot0(E0, cap=1), slot1(E2, cap=1)]
  GPU_1: [slot0(E1, cap=1), slot1(E3, cap=1)]
  #slots=1, #layers=2 per GPU

TP=2 + EP:
  GPU_0: [slot0(E0_half0, cap=0.5), slot1(E1_half0, cap=0.5)]
  GPU_1: [slot0(E0_half1, cap=0.5), slot1(E1_half1, cap=0.5)]
  #slots=2, #layers=2 per GPU
  cap=0.5 表示每个 slot 存储 expert 参数的一半（另一半在 TP partner GPU）
```

Expert Slot 抽象的关键作用：它使不同混合并行方案之间可以互相比较和转换——相同 slot 配置意味着切换时不需内存分配/释放（只交换参数），这定义了 SmartMoE "pool" 的边界。

术语一般如何实现？如何使用？
SmartMoE 在 FastMoE 框架上实现 expert slot 抽象：运行时，expert 到 slot 的映射由 expert placement 算法动态决定；slot 的 capacity/#slots/#layers 属性由离线阶段搜索的混合并行策略固定。切换时，只有受影响的 slot 内的 expert 参数通过 All-to-All 重新分配。安装：`cd src/fastmoe && USE_NCCL=1 python setup.py install --user`。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

## Workload-Aware Performance Modeling for MoE Training（负载感知的MoE训练性能建模）

术语是什么？
Workload-Aware Performance Modeling 是 SmartMoE 离线阶段使用的性能预测方法。与传统的 data-insensitive 性能模型（仅使用模型结构和硬件信息）不同，它利用 gating network 的设计语义估计训练时的 per-expert 负载分布——在实际训练之前预测执行计划的性能。核心思想：虽然实际 expert selection 在训练前不可获得，但 gating network 的超参数（capacity factor / topology-aware constraints）提供了负载的上界估计，该上界通常接近实际瓶颈，可用于准确预测。

从kernel调度角度拆解术语：
针对两类 gating network 分别估算：
```
类别1: Load-Balanced Gating (GShard gate)
  输入: capacity_factor (e.g., 1.2, 2.4, +∞)
  计算: max_tokens_per_expert = (capacity_factor × batch_tokens) / num_experts
         bottleneck_expert_load = min(max_tokens_per_expert, total_tokens)
  通信量估算: 使用 capacity_factor 控制下的路由分布估算 All-to-All dispatch 量
  
  例子: 4 GPUs, 16 experts, capacity=2.4, batch=1024 tokens
        max_tokens_per_expert = 2.4 × 1024 / 16 = 153.6 → 154 tokens
        最重 expert ≤ 154 tokens, 通信比例 ≤ 154/64 = 240%

类别2: Topology-Aware Gating (Faster Gate)
  输入: 硬件拓扑（node数, GPUs_per_node）+ hierarchical routing 算法
  计算: 按 Faster Gate 的两层路由算法模拟
        优先 intra-node routing → 估计 cross-node all-to-all 量
        max communication per device pair = f(hardware_topology, gate_algorithm)
  
性能模型输出:
  T_layer = T_compute(max_expert_load) + T_comm(max_cross_node_alltoall)
  其中 T_compute 和 T_comm 使用 FasterMoE 的基础性能模型
  R² > 0.5 for all evaluated configurations
```

术语一般如何实现？如何使用？
SmartMoE 在离线池搜索阶段使用该模型：对每个候选池（固定 DP/TP/PP + 可变 expert placement），用 workload-aware 模型评估所有 expert placement 变体的平均/最差性能 → 选性能最优的池。注意原始 FasterMoE 性能模型需要运行时负载数据作为输入——SmartMoE 的关键创新是用 gating 语义估算替代实际数据。该模型也可用于其他需要离线预测 MoE 性能的场景。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

## Enlarged Hybrid Parallelism Space for MoE（面向MoE的扩展混合并行空间）

术语是什么？
Enlarged Hybrid Parallelism Space 是 SmartMoE 提出的概念：在传统 MoE 混合并行空间（DP、TP、PP、EP 的组合）基础上，新增 **Expert Placement** 作为可搜索的并行策略维度。传统自动并行化系统（Alpa、Tofu）和 MoE 训练系统（FasterMoE、Tutel）将 expert 到 GPU 的映射视为固定的（按索引顺序放置），但 SmartMoE 发现 expert placement 的顺序直接影响负载均衡和性能——两个 expert placement 方案即使在"每个 GPU 上有相同数量的 expert"这一粗粒度下等价，其实际性能也可能因动态 token 分布而差异显著。

从kernel调度角度拆解术语：
以 4 experts, 2 GPUs 为例说明 placement 的影响：
```
Workload: E0=200t, E1=300t, E2=200t, E3=100t (极度不均)

方案 A (按索引顺序):         方案 B (按负载交错):
GPU_0: {E0(200), E1(300)}     GPU_0: {E1(300), E3(100)}
  load=500t                     load=400t
GPU_1: {E2(200), E3(100)}     GPU_1: {E0(200), E2(200)}
  load=300t                     load=400t
  Imbalance: 200t                Imbalance: 0t ✓

传统系统: 方案 A = 方案 B（每 GPU 均有 2 experts）
SmartMoE: 方案 B > 方案 A（考虑实际负载）
```

SmartMoE 的 Enlarged Space 将搜索空间从"选择何种 DP/TP/PP/EP 组合"扩展到还包括"expert 如何映射到 expert slot"，使用 expert slot 抽象统一表达。这使得系统能在更大范围内搜索最优并行配置。

术语一般如何实现？如何使用？
在 SmartMoE 中，Enlarged Space 被分解为两阶段处理：离线阶段搜索固定部分（DP/TP/PP/EP 组合），在线阶段动态调整可变部分（expert placement within pool）。实现基于 FastMoE 的 expert slot 抽象，使用 PyTorch。搜索策略：离线穷举候选池 + workload-aware 性能模型评估；在线 Greedy/DP/Hybrid 算法。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

## Expert Pinning（专家钉选/固定）

术语是什么？
Expert Pinning 是 ES-MoE 的 adaptive offloading 优化：将一小部分 hot experts 永久保留在 GPU 显存（不参与 offload），其余 experts 继续动态 offloading。动机：expert 增多时 per-expert compute time 下降，若所有 expert 每次上传，upload time 占比上升导致 pipeline overlap 效果减弱。Pinning hot experts 减少总 upload volume，延长可 overlap 的计算窗口。

从kernel调度角度拆解术语：
每个 iteration：读取上一 iteration 的 per-expert token counts → 排序 → 选择 top 25% → 固定到 GPU → 其余 experts 参与动态 offloading + dynamic placement。MoE-M 32 experts + 4 GPUs: 25% pinning → 22.8% 吞吐量提升（vs no pinning）。

术语一般如何实现？如何使用？
实现为 adaptive offloading 控制器子模块。利用相邻 iteration 间 expert load 的 temporal locality 选择 pinning target。适用于 expert 数量较多场景（>32 experts）。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Dispatch Mask (in MoE Training / MoE训练中的调度掩码)

术语是什么？
Dispatch Mask 是 MoE 训练中使用 batched matrix multiplication 时的核心数据结构——一个巨大的映射表 (N_tokens_padded × N_tokens)，将任意顺序的 tokens 重新排序为 per-expert 连续排列，使每个 expert 的输入形成紧凑矩阵以支持 batched GEMM。

从kernel调度角度拆解术语：
Dispatch Mask 的构造：对每个 token，根据 gating result 计算其在 padding buffer 中的位置（expert_offset + counter），mask[pos][token_i] = 1。使用：input_per_expert = Dispatch_Mask @ input（稀疏矩阵乘法）。内存开销极大：MoE-L (d_model=1536, batch 32, seq 1024) → N_tokens=32768, Dispatch_Mask ~ (32×1024×32768)×4 bytes ≈ 4.3 GB per mask。

术语一般如何实现？如何使用？
在 Fairseq GShard/Tutel 中使用 sparse matmul 实现。megablocks 提出 block-sparse 变体减少存储。ES-MoE 的根本性方案：不使用 batched GEMM 和 dispatch mask，改用 sequential expert processing，tokens 按 gating 结果直接分配给各 expert。节省 >48 GiB GPU memory。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Sequential Expert Processing（串行专家处理）

术语是什么？
Sequential Expert Processing 是 ES-MoE 提出的替代 batched GEMM 的 MoE 计算执行方式。传统 MoE 框架所有 experts 同时在 GPU 上用 batched GEMM + dispatch mask 处理；sequential approach 逐 expert 串行执行：上传 expert → 处理其 tokens → 释放 GPU 内存 → 下一个。

从kernel调度角度拆解术语：
```
for expert_j in assigned_experts:
    mask_j = (gating_result == expert_j)       # boolean mask [N_tokens]
    tokens_j = input[mask_j]                    # [T_j, H] actual tokens
    hidden = silu(gate_proj(tokens_j)) * up_proj(tokens_j)
    output[mask_j] = down_proj(hidden)          # scatter back
```
无 dispatch mask、无 zero-padding。内存仅需 1 expert weight + activations。

术语一般如何实现？如何使用？
ES-MoE 修改 Fairseq MoE layer forward：替换 batched_gemm 为 per-expert loop。当 microbatch 足够大时 per-expert GEMM 可高效利用 Tensor Core。劣势：per-expert tokens 极小时 GPU 利用率下降（如 64 experts, batch 32 → ~16 tokens/expert）。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

## Chunked Prefetching for MoE Expert Loading（MoE专家加载的分块预取）

术语是什么？
Chunked Prefetching 是 ProMoE 提出的一种细粒度 expert 参数传输调度技术，用于解决 CUDA 异步拷贝（cudaMemcpyAsync）不可被中途抢占的问题。由于每个 MoE expert 包含三个线性层（gate_proj, up_proj, down_proj），ProMoE 将每个 expert 的参数按这三个自然边界拆分为 3 个 chunk，以 chunk 为最小单位提交 prefetch 任务。当高优先级（HIGH, precise）prefetch 任务到达时，worker thread 最多只需等待当前 chunk 完成（而非整个 expert），将高优先级任务的阻塞延迟降至原来的 1/3。

从kernel调度角度拆解术语：
Worker thread 循环从双优先级队列取 task（粒度=chunk），每个 task 包含 layer id、expert id、chunk id 和 priority。chunk=0 的任务触发 LRU cache replacement（为新 expert 分配空间）。每个 chunk 通过 cudaMemcpyAsync 传输，完成后更新 ready_chunk 计数器。Without chunking：LOW 任务 = 1 entire expert (~85MB) → HIGH 任务等待最多 ~3.7ms。With chunking（3 chunks）：LOW 任务 = 1 chunk (~28MB) → HIGH 任务等待最多 ~1.2ms，3× faster preemption。

术语一般如何实现？如何使用？
ProMoE 利用 MoE expert 天然的三层结构（gate_proj, up_proj, down_proj）作为 chunk 边界。ready_chunk 计数器跟踪每个 expert 的已完成 chunk 数。Inference 执行 expert FFN 前检查 ready_chunk 确保所需 chunk 已就绪。chunk 粒度权衡：更小 chunk 减少 preemption 延迟但增加任务队列管理开销；3 chunks/expert 在实践中提供良好平衡。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching

## Pipe-A2A (Pipelined All-to-All)

术语是什么？
Pipe-A2A 是 ScheMoE (EuroSys '24) 提出的一种新型 All-to-All 集合通信算法，专为异构 GPU 集群（intra-node 高带宽 + inter-node 较低带宽）上的 MoE 分布式训练设计。其核心思想是将 A2A 中的 Send/Recv (SR) 操作按 GPU 对是否位于同一节点分为 intra-node SR 和 inter-node SR，分配在两个独立的异步 CUDA stream（Intra-Stream 和 Inter-Stream）上并行执行，使得 intra-node 通信可以被 inter-node 通信的时间隐藏。

假设集群有 N 个节点、每节点 M 个 GPU（P = N×M），对所有 GPU i ∈ [0, P-1]，A2A 包含 P 个 SR(i,j) 操作。其中 j 表示目标 GPU。若 i 和 j 同节点，SR(i,j) 为 intra-node 操作（使用 Intra-Stream）；否则为 inter-node 操作（使用 Inter-Stream）。两 stream 并发执行，理论执行时间为：

$$t_{pipea2a} = \max\{M \times t_1, (P - M) \times t_2\}$$

而传统顺序执行的 NCCL-A2A 时间为：

$$t_{nccla2a} = M \times t_1 + (P - M) \times t_2$$

其中 t₁ 为单次 intra-node SR 耗时，t₂ 为单次 inter-node SR 耗时。理论最大加速比 S_max = (M×t₁ + (P-M)×t₂) / max(M×t₁, (P-M)×t₂)。当 t_intra ≈ t_inter 时加速最大（接近 2×）。

从kernel调度角度拆解术语：
以 8 GPU（2 node × 4 GPU）为例，GPU 0 的 Pipe-A2A 执行流程：

```
// GPU 0 的 A2A dispatch（输入 tensor I_0 按 expert 切分为 8 份）
// 同节点 GPU: GPU 0,1,2,3; 跨节点 GPU: GPU 4,5,6,7
// Intra-Stream 和 Inter-Stream 为两个独立的 cudaStream_t

// Intra-Stream (处理同节点 SR):
cudaStream_t intra_stream;
for target in [0, 1, 2, 3]:  // GPU 0 到同节点 GPU 的 SR
    if target != 0:
        cudaMemcpyAsync(send_buf[target], I_0[target], size, 
                        cudaMemcpyDeviceToHost, intra_stream)  // or GPUDirect RDMA
    // Recv 对端数据...
// 总共 M=4 个 intra-node SR 操作

// Inter-Stream (处理跨节点 SR, 并行执行):
cudaStream_t inter_stream;
for target in [4, 5, 6, 7]:  // GPU 0 到跨节点 GPU 的 SR
    // 通过 InfiniBand/NCCL 发送
    ncclSend(I_0[target], size, target, comm, inter_stream)
// 总共 P-M=4 个 inter-node SR 操作

// 两 stream 并发: cudaStreamSynchronize(intra_stream)
//               cudaStreamSynchronize(inter_stream)
// 总耗时 ≈ max(intra_time, inter_time) 而非 intra_time + inter_time
```

ScheMoE 实验表明：在 32 GPU (8×4 RTX2080Ti) 集群上，当消息大小 ≥ 200MB 时 Pipe-A2A 实现 1.4×-2× 加速优于 2DH-A2A 和 NCCL-A2A；小消息时约 3%-5% 提升。在 BERT-Large-MoE (~6.5B params) 上 Pipe-A2A 贡献有限（A2A 输入仅 524KB），说明该算法对大消息 MoE 配置（高 M × B × L）效果最显著。

术语一般如何实现？如何使用？
Pipe-A2A 在 ScheMoE 中通过 C++/CUDA 实现为 AbsAlltoAll 的子类。用户可通过继承 AbsAlltoAll 接口实现自定义 A2A 算法。实现依赖：两个 cudaStream_t（Intra-Stream 和 Inter-Stream），GPU Direct RDMA（同节点 GPU 间通过 PCIe/NVLink 直接访问内存），NCCL（跨节点 InfiniBand 通信）。在 ScheMoE Python 接口中通过 `all_to_all_impl = ScheMoE.PipeAlltoAll` 指定使用 Pipe-A2A。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling

## Hierarchical All-to-All (1DH-A2A / 2DH-A2A)

术语是什么？
Hierarchical All-to-All 是一类利用 GPU 集群异构拓扑（intra-node 高带宽 vs inter-node 低带宽）的 A2A 集合通信算法。基本思想是将全局 A2A 分解为两阶段：第一阶段在节点内执行局部 A2A（利用高带宽 intra-node 连接），第二阶段在节点间执行全局通信（仅传输必要数据到跨节点 GPU），从而减少跨节点通信轮次和利用快速 intra-node 连接。

- **1DH-A2A (1D-Hierarchical A2A)**：由 HetuMoE [31] 提出。Phase 1: intra-node A2A（每个节点内各 GPU 交换数据）；Phase 2: inter-node A2A（节点间交换跨节点目标的数据）。减少的通信轮次：相比完全扁平化的 P×(P-1) 个 Send/Recv 对，1DH-A2A 将跨节点通信轮次从 P×(P-M) 降至更少的层次化轮次。
- **2DH-A2A (2D-Hierarchical A2A)**：由 DeepSpeed-MoE [36] 和 Tutel [16] 提出。在 1DH 基础上进一步优化：Phase 1: local token permutation（节点内按 expert 重排）；Phase 2: local expert computation of shared experts；Phase 3: global cross-node exchange of data for remote experts。2DH-A2A 比 1DH-A2A 更细致地利用 intra-node 带宽。

从kernel调度角度拆解术语：
以 8 GPU (2 node × 4 GPU) 和 2DH-A2A 为例：

```
// 2DH-A2A 在 MoE dispatch 中的执行流程
// 输入: 每个 GPU 持有 B×L tokens，gating 确定每个 token 的 target expert

// Phase 1: Intra-node scatter (node 内)
// GPU 0 持有 tokens for experts on GPU 0,1,2,3,4,5,6,7
// 在 node 0 内部 (GPU 0-3):
for gpu in [0,1,2,3]:
    // PCIe/NVLink 高速传输: GPU i → GPU j (i,j 同节点)
    send_tokens_to_intra_node_target(gpu)

// Phase 2: Inter-node exchange
// 每个 node 将所有跨节点的 token 聚合到一个代表 GPU
// node 0 的 GPU 0 持有所有需要发往 node 1 的 token
// node 0 → node 1: InfiniBand 传输 (仅一次跨节点 A2A per node pair)
ncclGroupStart()
for node_pair in cross_node_pairs:
    ncclSend(node_rep_gpu, tokens_for_other_node, ...)
ncclGroupEnd()

// Phase 3: Intra-node gather (node 内)
// 各 node 内将收到的跨节点 token 分发到目标 GPU
// 再次利用高速 intra-node 连接
```

但 ScheMoE 指出，1DH-A2A 和 2DH-A2A 的共同局限是 Phase 1 和 Phase 2 必须顺序执行——intra-node 和 inter-node 带宽无法同时利用。Pipe-A2A 通过双 stream 并发执行消除了这一瓶颈。

术语一般如何实现？如何使用？
1DH-A2A 在 Hetu 框架（https://github.com/Hsword/Hetu）中实现，2DH-A2A 在 Tutel（https://github.com/microsoft/tutel）和 DeepSpeed-MoE 中实现。ScheMoE 的 AbsAlltoAll 抽象接口同时支持这些算法作为可插拔实现，用户可在初始化时选择 NCCL-A2A、1DH-A2A、2DH-A2A 或 Pipe-A2A。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
- HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System
- Tutel: Adaptive Mixture-of-Experts at Scale
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## Adaptive Operator Scheduling for Communication-Computation Overlap in MoE Expert Parallelism

术语是什么？
Adaptive Operator Scheduling 是 ScMoE 中实现专家并行通信-计算重叠的运行时调度策略。核心思想：在 ScMoE 架构解耦通信与计算的顺序依赖后，将 MoE stream 中的算子（gate routing、encode、All-to-All dispatch、expert computation、All-to-All combine、decode）自适应插入到 shared expert stream（attention → shared expert computation）的适当位置，最大化通信与计算的重叠时间。关键调度决策是为 expert computation 选择最优插入位置——在 shared expert stream 的4个候选位置（①②③④）中选择使总时间最小的位置 K*。

从kernel调度角度拆解术语：
自适应调度器选择 expert computation 位置的优化目标和伪代码：

```
# 性能模型输入（通过profiling获得）
T_disp    = All-to-All Dispatch通信时间
T_comb    = All-to-All Combine通信时间
COMP[1:4] = shared expert stream中4个计算算子各自的执行时间

# 优化目标：选择最优expert computation插入位置K
K* = argmin_K ( |Σ_{i=1}^{K-1} COMP_i - T_disp| + |Σ_{i=K+1}^{4} COMP_i - T_comb| )

# 调度执行（双CUDA stream并行）
# Stream 0 (Shared Expert):  [COMP_1][COMP_2]...[COMP_K-1]...[COMP_K+1][COMP_4]
#                             <-- Overlap with Dispatch -->   <-- Overlap with Combine -->
# Stream 1 (MoE):  [Gate][Encode][Async Dispatch]...[Expert Comp at pos K]...[Async Combine][Decode]

# 总时间边界：
# 下界: T_overall >= |(ΣCOMP) - (T_disp + T_comb)|
# 上界: T_overall <= (ΣCOMP) + (T_disp + T_comb)
# 当通信时间 ≤ overlap_window = ΣCOMP 时，T_overall = ΣCOMP (100%通信隐藏)
```

三种shortcut位置对应的overlap窗口：
- Pos-1: overlap_window = T_Atten + T_SE
- Pos-2: overlap_window = T_Atten + T_SE + T_MLP
- Pos-3: overlap_window = 2*T_Atten + T_SE + T_MLP

Pipeline Augmentation：当 T_disp + T_comb > overlap_window 时，ScMoE 的自适应调度与 pipeline 策略可组合使用。先用 ScMoE 的扩展窗口隐藏部分通信，剩余无法隐藏的部分通过 token 分 chunk 的 pipeline 进一步隐藏。第5条 timeline（图7）展示了这种组合。

与纯 pipeline 策略的核心区别：
- Pipeline: 将tokens等分为M个chunks，chunks间通信与计算交错。但第1个chunk的dispatch（prologue）和第M个chunk的combine（epilogue）无法被隐藏。bubble = T_disp/M + T_comb/M。
- ScMoE自适应调度: 通信的"窗口期"从时间线前端（Block-MLP计算期间）自然延伸到后端（shared expert计算期间），无prologue/epilogue概念。当 T_disp+T_comb ≤ overlap_window 时实现0%通信暴露。

术语一般如何实现？如何使用？
基于 PyTorch CUDA stream API 实现：(1) 通过 `torch.cuda.Stream()` 创建独立的 MoE stream；(2) 在训练开始前 profiling 各算子的执行时间（T_disp, T_comb, COMP_1..4），构建性能模型；(3) 每个 iteration 在 CPU 侧计算 min_K 目标函数确定 K*；(4) 在 MoE stream 上提交 gate → encode → async dispatch 操作；(5) 在主 stream 的 K*-1 位置后插入 expert computation（通过 `torch.cuda.current_stream().wait_stream(moe_stream)` 同步）；(6) expert computation 完成后在主 stream 恢复后续计算，同时在 MoE stream 提交 async combine → decode。ScMoE 在 8×A30-PCIe 场景重叠 70% 通信，8×A800-NVLink 场景完全重叠通信。

涉及论文标题：
- Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts

## Expert Data Parallelism (EDP / 专家数据并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Data Parallelism (EDP) 是 Skywork-MoE 提出的针对 MoE 训练的定制化并行策略，定义为 Size_EP = Size_TP。与 Megatron-LM Core 0.6.0 中已有的 Expert Parallelism (EP, Size_EP = Size_DP * Size_TP) 和 Expert Tensor Parallelism (ETP, Size_EP = Size_DP) 不同，EDP 的核心设计是在 Attention 层使用 Tensor Parallelism (TP)，在 MoE/FFN 层切换为 Expert Parallelism (EP)，同一数据同时穿越 TP Group 和 EP Group。Device mesh 配置：Attention weights 为 [Size_PP, Size_DP, Size_TP]，Expert weights 为 [Size_PP, Size_DP, Size_EP]。EDP 对中等 expert 数量（≤64）的 MoE 模型特别有效，能优化 gating 层 token 路由的 AllToAll 通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Skywork-MoE (1536 A800 GPUs) 的训练中 EDP 配置为 Size_PP=12, Size_DP=32, Size_TP=Size_EP=4：

```
# === Device Mesh 定义 ===
# 总 GPU 数 = PP * DP * TP = 12 * 32 * 4 = 1536
# Attention Device Mesh: [PP=12, DP=32, TP=4]
# Expert Device Mesh:    [PP=12, DP=32, EP=4]

# === 单层前向传播流程 (在 EDP 下) ===

# Phase 1: Attention Layer (TP Mesh)
# TP Group: 4 GPUs, 切分 head 维度
# 每 GPU 处理 36/4 = 9 attention heads
x = LayerNorm(x)                    # 所有 GPU 独立计算
# QKV projection: 每 GPU 计算部分 heads
q, k, v = split_heads(W_QKV @ x)    # TP 切分 head 维度
# Flash Attention: 在 TP Group 内通信
attn_out = flash_attention(q, k, v)  # 需要 TP group 内 all-reduce
attn_out = W_O @ attn_out           # TP 切分 + all-reduce
x = x + attn_out

# Phase 2: MoE Layer (EP Mesh, Mesh 切换)
# EP Group: 4 GPUs, 每 GPU 持有 16/4 = 4 个完整 expert
x = LayerNorm(x)
# Gate: 每 GPU 独立计算 (gate 参数在所有 EP GPU 上复制)
z = W_gate @ x + b_gate
z_tilde = (z - mean(z)) / std(z)    # Gating Logit Normalization
g = softmax(z_tilde)
E_i = topk(g, k=2)

# Token Dispatch: AllToAll 在 4 EP GPUs 间
# 将 token 发送到持有目标 expert 的 GPU
tokens_dispatched = all_to_all(tokens, routing_map)

# Expert FFN 计算: 每 GPU 独立计算其持有的 4 个 expert
For each expert j on this GPU:
    expert_out[j] = SwiGLU_FFN(tokens_for_expert_j)

# Token Combine: AllToAll 将 expert 输出送回原 GPU
tokens_combined = all_to_all(expert_outputs, reverse_routing_map)

# Weighted sum
y = weighted_combine(tokens_combined, g[E_i])
x = x + y
```

EDP 的核心优势：EP 组大小 = TP 组大小，使得 Attention 和 Expert 阶段的通信模式协调，避免了 ETP 中 AllToAll 随 TP 增大而迅速膨胀的问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 Skywork-Megatron 框架（基于 Megatron-LM 23.06），需要在训练框架中支持动态 device mesh 切换。具体实现要点：(1) 在 Attention 层和 Expert 层之间切换通信组（TP group ↔ EP group）；(2) 确保 expert 数量 ≥ Size_EP（才有足够的 expert 分配给每个 EP rank）；(3) Gate 参数在所有 EP ranks 上复制（非分布式）。适用场景：expert 数量 ≤ 64 的 MoE 模型训练，在通信开销和 GPU 利用率之间取得最优平衡。

涉及论文标题：
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

## Expert Tensor Parallelism (ETP / 专家张量并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Tensor Parallelism (ETP) 是 Megatron-LM Core 0.6.0 中的一种 MoE 训练并行策略，定义为 Size_EP = Size_DP。与 EP（每个 expert 完整驻留在单个 GPU 上，受 expert 数量上限限制 GPU 扩展）不同，ETP 允许将单个 expert 的权重切分到多个 GPU 上（通过 Size_TP），从而突破 EP 的 GPU 扩展上限（expert 数量限制）。然而代价是 AllToAll 通信开销随 Size_TP 增大而迅速增加，因为每个 token 的 expert 输出需要在更多的 GPU 间进行集合通信。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ETP 的 device mesh 为 [Size_PP, Size_DP=Size_EP, Size_TP]：

```
# ETP 配置示例: Size_PP=8, Size_DP=Size_EP=16, Size_TP=4 → 512 GPUs
# 每个 expert 被切分到 4 个 GPU 上 (TP)
# 每个 EP group 内 4 个 GPU 共同持有 1 个完整 expert

# Expert FFN 前向 (ETP 模式下):
For each GPU in EP group:
    # Expert 权重在 TP 维度切分
    # W_gate: [d_model, d_ffn/TP] — 列切分
    # W_up:   [d_model, d_ffn/TP] — 列切分
    # W_down: [d_ffn/TP, d_model] — 行切分
    
    # 局部计算
    h1 = W_gate_partial @ x     # [batch, d_ffn/TP]
    h2 = W_up_partial @ x       # [batch, d_ffn/TP]
    h = SwiGLU(h1) * h2
    partial_out = W_down_partial @ h  # [batch, d_model]
    
    # TP group 内 all-reduce 得到完整输出
    expert_out = all_reduce(partial_out)
```

相比 EDP (Size_EP = Size_TP)：
- ETP: Size_EP = Size_DP → EP 组更大，单个 expert 分布在更多 GPU 上，AllToAll 通信量更大，但可用 GPU 不受 expert 数量限制
- EDP: Size_EP = Size_TP → EP 组更小（等于 TP 组大小），AllToAll 通信更高效，但 GPU 扩展受 expert 数量影响

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ETP 在 Megatron-LM Core 0.6.0 中实现。适用场景：expert 数量较少但需要大规模 GPU 集群训练的 MoE 模型。实际选择：当 expert 数量 ≤ 64 时，EDP 因通信效率更高而优于 ETP；当 expert 数量很大（如 128+）时，EP 可能已经足够，ETP 的 TP 切分优势不显著。

涉及论文标题：
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

## Unbalanced Pipeline Parallelism（非均衡流水线并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Unbalanced Pipeline Parallelism 是 Skywork-MoE 提出的一种 Pipeline Parallelism (PP) 优化策略，打破传统的均匀层分割（每 PP stage 分配相同数量的 transformer 层），采用非均匀分割以减少 pipeline bubble time。其核心动机：由于最后一个 PP stage 除了正常的 transformer 层计算外还需要处理 loss calculation（包括 logits projection、cross-entropy 计算等），导致该 stage 成为计算瓶颈，增加 bubble time。通过将最后一 stage 的 transformer 层数减少（例如从 6 层减少到 4 层），可以补偿 loss calculation 的额外计算，实现更好的 stage 间负载均衡。Skywork-MoE 实验显示，将 24 层模型从均匀 4-stage [6,6,6,6] 改为非均匀 5-stage [5,5,5,5,4] 可以减少 pipeline bubble time 约 10%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 24 层 transformer, 4-stage PP 对比 5-stage unbalanced PP 为例：

```
# === 均匀 4-stage PP: [6, 6, 6, 6] ===
# Stage 0: layers 0-5   (6 layers)
# Stage 1: layers 6-11  (6 layers)
# Stage 2: layers 12-17 (6 layers)
# Stage 3: layers 18-23 + Loss (6 layers + loss calc)
# 问题: Stage 3 的 loss calculation 使其成为 bottleneck
# Bubble time 较大

# === 非均匀 5-stage PP: [5, 5, 5, 5, 4] ===
# Stage 0: layers 0-4   (5 layers)
# Stage 1: layers 5-9   (5 layers)
# Stage 2: layers 10-14 (5 layers)
# Stage 3: layers 15-19 (5 layers)
# Stage 4: layers 20-23 + Loss (4 layers + loss calc)
# 优势: Stage 4 少 1 层 → 节省的时间补偿 loss calculation
# Bubble time 减少约 10%

# 配合差异化梯度重计算 (Gradient Checkpointing):
# Stage 0-3: 正常 checkpointing 配置 (buffer 较小)
# Stage 4: 减少 checkpointing (buffer 较大, 因层数少)
# 进一步平衡各 stage 的显存使用
```

Skywork-MoE 在 146B/52 层模型上使用 12-way unbalanced PP。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 Skywork-Megatron 框架：(1) 在 PP 配置中指定非均匀的层分割方案；(2) 为每个 PP stage 差异化配置 gradient checkpointing（activation recomputation）策略，buffer 大的 stage 减少 checkpointing 以平衡显存；(3) 根据实际 profiling 结果调整各 stage 的层数分配。该技术是 PP 的通用优化，可应用于任何使用 PP 的 transformer 训练，不仅仅是 MoE 模型。

涉及论文标题：
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

## Grouped GEMM (Varlen-M / Varlen-K) for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Grouped GEMM 是将多个独立 GEMM（矩阵乘法）作为一个 batch 在 GPU 上并行执行的技术，特别适合 MoE 场景。MoE 的每个 expert 独立处理不同数量的 token（T_e 可能不同），因此需要 varlen-M Grouped GEMM（M 维度=token 数可变，N/K 维度=权重矩阵固定）或 varlen-K Grouped GEMM（K 维度=token 数可变）。具体：(1) Varlen-M Grouped GEMM：用于 forward up-proj/down-proj 和 backward activation gradient，A∈R^{T_e×d}, B∈R^{d×2n}, T_e 各 expert 不同；(2) Varlen-K Grouped GEMM：用于 backward weight gradient，A^T∈R^{d×T_e}, B∈R^{T_e×2n}，在 K 维度 reduction，T_e 不同。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SonicMoE 中 varlen-M Grouped GEMM kernel（H100, per thread block）：

```
// Persistent tile scheduler：所有 expert 的 tiles 进入统一 work queue
while (tile = atomicAdd(&work_counter, 1) < total_tiles):
    expert_id, m_tile, n_tile = decode_work_tile(tile)
    Te = expert_token_counts[expert_id]
    
    // Prologue: TMA load weight + cp.async gather input
    update_tensor_map_for_Te(desc, Te)
    cp.async.bulk.tensor.load(W_smem, W_desc[expert_id])
    cp.async.gather.load(X_smem, X, routing_idx[expert_id], m_tile)
    
    // Mainloop over K dimension
    for k in [0..ceil(d/Ktile)):
        wgmma(acc, X_smem[k], W_smem[k])
    
    // Epilogue: activation or store
    tma_store(output[expert_id], acc)
```

DeepGEMM 的 varlen-M Grouped GEMM 要求每个 expert 的 token 数必须是 M_tile 的倍数（不支持运行时 tensor descriptor 更新），SonicMoE 通过在线 descriptor 更新解决了此限制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：CUTLASS 3.x Grouped GEMM (CuTe-DSL)、DeepGEMM、SonicMoE。在 PyTorch 层面通过自定义 CUDA extension 调用。SonicMoE 提供 PyTorch nn.Module 封装，直接替换 MoE layer。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

## Ping-Pong Scheduling (Warpgroup MMA/IO Overlap)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ping-Pong Scheduling（乒乓调度）是 NVIDIA Hopper GPU 上 warp-specialized GEMM kernel 的线程调度策略：将 2 个 consumer warpgroups 交替分配 MMA 和 epilogue/IO 工作。当 consumer WG 0 执行 WGMMA 进行当前 tile 的矩阵乘法时，consumer WG 1 同时执行上一 tile 的 epilogue（activation、store to HBM）。每 tile 完成后角色互换。此概念最早见于 CUTLASS Hopper warp-specialized kernels，FlashAttention-3 (Shah et al. 2024) 将其用于 attention。SonicMoE 首次将 Ping-Pong 应用于 MoE kernel，特别是针对细粒度 MoE 的 heavy epilogue 场景（如 backward dH kernel 需在 epilogue 中同时执行 dSwiGLU、dS reduction 和 TMA store A'）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SonicMoE forward down-proj Y kernel Ping-Pong 调度（H100）：

```
// 2 consumer warpgroups 交替执行 MMA 和 epilogue
for tile_i in work_tiles:
    if tile_i % 2 == 0:  // Ping phase
        consumer[0].wgmma(tile_i)    // WGMMA compute
        consumer[1].epilogue(tile_i-1) // TMA store previous tile
    else:                 // Pong phase
        consumer[1].wgmma(tile_i)
        consumer[0].epilogue(tile_i-1)
    sync_warpgroups()  // role switch barrier
```

传统无 Ping-Pong：MMA → barrier → epilogue → barrier → next MMA。Ping-Pong 将 epilogue latency 完全隐藏在 MMA 之下。特别对 fine-grained MoE（dH kernel epilogue 需 load H + compute dSwiGLU + dS reduction + store dH/dS/A'），Ping-Pong 维持了高 Tensor Core utilization。

Blackwell 上的等效：利用 TMEM 2-stage（每 stage 256×128 columns of 32-bit cells）和 UMMA 单线程异步指令——MMA warp 写入一个 TMEM stage 时，epilogue warps 并发读取另一个 stage 的结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CuTe-DSL 实现：创建 2 个 consumer warpgroup pipeline stages，使用 `cute::conditional_return` 和 warpgroup barrier 同步。适用条件：epilogue IO 相对 MMA tile 不可忽略（n < 1024 的 fine-grained MoE），且 SMEM 充足支持双缓冲。SonicMoE 动态选择 Ping-Pong vs 普通 scheduling 取决于 intermediate size。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

## Gather Fusion in MoE Kernels

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gather Fusion 是将 MoE 中 token gather 操作（根据 routing mask π 将 token 从不同原始位置按 expert 收集）与 GEMM 的 GMEM-to-SMEM load 融合的技术。传统方式需先 launch gather kernel（X → X_e 连续 buffer），再在 GEMM 中加载。Gather fusion 直接在 GEMM prologue 使用 cp.async 指令按 routing index 从分散位置读到 SMEM，消除 X_e 物化（saves 2TKd bytes IO）和额外 kernel launch。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 传统（含 gather fusion 的 GEMM 需单独 gather kernel）
// Step 1: gather(X, routing_idx) → X_e  // 2TKd bytes HBM traffic
// Step 2: GEMM(X_e, W)                  // 需读 X_e

// SonicMoE gather fusion
for each expert e:
    for m_tile in expert_e_tiles:
        // cp.async 直接从原始 X 分散加载
        for t in m_tile:
            src = routing_idx[e][t]
            cp.async.load(SMEM[t], X[src])
        cp.async.wait()
        wgmma(acc, SMEM, W[expert_id])
    tma_store(output, acc)
// 总 HBM 访问相比无 gather fusion 节省 2TKd bytes
```

ScatterMoE/MoMoE 仅在 forward varlen-M 实现 gather fusion（backward 仍单独 gather）。SonicMoE 在 forward varlen-M 和 backward varlen-K (dW1/dW2) 均实现 gather fusion。Blackwell 2-CTA cluster 的特殊处理：CTA 1 的 relay warp 接收 cp.async completion → 用 mbarrier cluster-scope 转发给 CTA 0。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA PTX `cp.async.ca.shared.global` 指令（Ampere+）。ScatterMoE/MoMoE 仅 forward varlen-M fusion；SonicMoE forward + backward 全覆盖。当输入已 contiguous-packed 时无需 gather fusion。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

## Epilogue Fusion in MoE Kernels (SwiGLU / dSwiGLU / dS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Epilogue Fusion 是将 GEMM 输出后处理与多个原本需单独 kernel launch 的操作融合。SonicMoE 实现：(1) forward up-proj A kernel：WGMMA → SwiGLU activation + TMA store 融合在 epilogue；(2) backward down-proj dH kernel：WGMMA → dA = Broadcast(s)·dA' + dSwiGLU(dA, H) + dS = ⟨dA',A⟩ + A' = Broadcast(s)·A + TMA store。4 个操作融合于同一 kernel epilogue，消除了额外 kernel launch 和输出/输入 HBM 往返。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
dH kernel epilogue（per tile, per expert）：

```
// 输入: dA'_e = dO_e W_{2,e}^T (WGMMA accumulator), s_e, H_e

// 1. dA + SwiGLU recompute + dSwiGLU
for t in tile:
    dA[t,:] = s[t] * dA'[t,:]
    gate = sigmoid(H[t,:n])
    A[t,:n] = gate * H[t,n:2n]
    dH[t,:n] = dA[t,:n] * A[t,:n] * gate * (1-gate)
    dH[t,n:2n] = gate * dA[t,:n]

// 2. dS = sum over n of dA'[t,i] * A[t,i]
dS[t] = dot_product(dA'[t,:], A[t,:])

// 3. A' preparation for dW2
A'[t,:] = s[t] * A[t,:]
// TMA store: dH, dS, A' → HBM
```

SonicMoE 选择 dS=⟨dA', A⟩（vs ScatterMoE 的 dS=⟨dO, Y⟩）：(1) 0 额外 HBM 访问；(2) reduce over n vs d：节省 log₂(d/n) 轮 reduction；(3) 不需缓存 Y（节省 2TKd bytes activation）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CuTe-DSL 中通过自定义 epilogue functor 实现。dH kernel 依赖 Ping-Pong scheduling 在 hopper 上维持 high TFLOPS despite heavy epilogue。Blackwell 上通过 TMEM 2-stage + UMMA 实现更好 overlap。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

## Tile Quantization in Sparse MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tile Quantization 指 GPU GEMM 必须 pad 矩阵维度到 tile size 整数倍时产生的计算浪费。对于 MoE：expert e 收到 T_e 个 token，若 T_e mod M_tile ≠ 0，需 pad 到 ceil(T_e/M_tile)×M_tile 个 token，padding 位置的 GEMM 计算全部浪费。稀疏 MoE 下 T_e 很小（如 E=256, T=16K, K=4 时 T_e≈250, M_tile=128 需 2 tiles=256，waste 6/256≈2.3%），但绝对浪费随稀疏度增加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE forward+backward FLOPs = (6+12)T_e·n·d per expert。当 E 从 32 增至 256（保持 T 和 K 不变），T_e 从 2000 降至 250，tile quantization waste 从 ~2.3% 升至 ~2.3%（比例相近但绝对 tile 数翻倍）。实际影响不仅 FLOPs——小 M tile 降低 SM occupancy 和 TMA efficiency。SonicMoE Figure 8 显示 T=16k, d=4k, n=1k, K=4 下 waste 随 E 增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
解决：(1) Token Rounding（SonicMoE）：routing 阶段将 f_e 舍入到 M_tile 倍数；(2) Token Dropping：丢弃超 capacity token；(3) Dynamic tile shape。SonicMoE TR 在 K/E ≤ 1/64 时带来 16% kernel TFLOPS 提升。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

## Shuffled-Reduce-Scatter (SRS / 混洗规约分散)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shuffled-Reduce-Scatter (SRS) 是 Sem-MoE 在 Attention-TP 场景中实现的融合通信原语，将 speculative token shuffling（基于预测的 token 重排）嵌入标准 reduce-scatter 集合通信操作中。传统 Attention-TP 流程为：attention → allreduce → gate → all-to-all dispatch。SRS 将后三个操作融合：在 reduce-scatter 阶段按预测的 expert device assignment 对 token 进行重排，使每个 device 直接获得应路由给它的 token 子集，消除后续 all-to-all dispatch。Shuffling 逻辑嵌入 ring-based communication schedule，额外 overhead 仅约 1%。配套的 argsort kernel（Triton 实现）比 PyTorch 原生快 25%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SRS Kernel: Fused Shuffle + Reduce-Scatter
Input: X ∈ R^{B×H}          # post-attention hidden states
       T ∈ R^{t×E}           # token-to-device schedule table
       A ∈ R^{E²×E}          # 2-gram inter-layer device transition table
       C_p, A_p              # confidence scores

Step 1: Predict target device per token (两表竞争)
  for each token_id j in batch:
    if C_p[j] > A_p[prev_seq]:
      dev_ids[j] = T[j]          # token-level prediction
    else:
      dev_ids[j] = A[(d_prev1, d_prev2)]  # inter-layer prediction

Step 2: Compute shuffle indices (Triton argsort, 25% faster than PyTorch)
  shuffle_indices = argsort(dev_ids)

Step 3: Group, align, concatenate
  shuffle_indices = concat(align(group_by_key(shuffle_indices)))

Step 4: Shuffle + Ring-based Reduce-Scatter
  X_shuffled = X[shuffle_indices]
  X_local = reduce_scatter(X_shuffled)  # integrated shuffle overhead ≈ 1%

Output: X_local_i per device  # ready for local gate + expert FFN
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Triton 实现，调度表驻留 GPU memory（<12 MB for DeepSeek-V2），O(1) 查表。两表竞争机制确保鲁棒性：token-level 和 inter-layer 预测均低置信度时 fallback 到标准 all-to-all。结合 DeepEP 通信后端。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

## Shuffled-AllGather (SAG / 混洗全收集)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shuffled-AllGather (SAG) 是 Sem-MoE 中与 SRS 配对的融合通信原语。在 MoE expert computation 完成后，将 allgather + reverse shuffle 融合为单次操作。利用 SRS 保存的 shuffle_indices，通过 argsort 计算反向排列，经 GPU tensor indexing 恢复原始 token 顺序，消除标准 EP 的 all-to-all combine。

从kernel调度角度拆解术语：

```
# SAG Kernel
Input: Y_local per device, shuffle_indices (saved from SRS)

Step 1: Ring-based AllGather
  Y_shuffled = allgather(Y_local)  # in shuffled order

Step 2: Reverse shuffle
  reverse_indices = argsort(shuffle_indices)
  Y = Y_shuffled[reverse_indices]  # restored original order

Output: Y ∈ R^{B×H}  # next layer ready
```

SRS+SAG 组合确保 token shuffling 无损——计算位置变化不影响最终结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Triton 实现，与 SRS 共用调度表。Reverse shuffle 仅需 tensor indexing。Overhead ≈ 1%。配合 DeepEP 增强回退时的 all-to-all 性能。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

## Expert Processing with Tensor Parallelism on NMP (NMP上的张量并行专家处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Processing with Tensor Parallelism on NMP 是 Stratum 中将 MoE expert 计算映射到 NMP logic die 多个 PU 上并行执行的核心调度策略。MoE 的每个 expert 执行三个级联 GeMM 操作（projection-up GeMM1: W_1 ∈ R^{K×N}, projection-up GeMM2: W_2 ∈ R^{K×N}, projection-down GeMM3: W_3 ∈ R^{N×K}），Stratum 采用 tensor parallelism 将每个 expert 的权重矩阵分片到所有 PU 上，所有 PU 协作处理一个 expert（sequential across experts），而非并行处理多个 experts。矩阵分区策略：(a) GeMM1/2 沿 N 维（列）垂直分片——W_1[i] ∈ R^{K×(N/P)}；(b) GeMM3 沿 K 维（行）水平分片——W_3[i] ∈ R^{(N/P)×K}。这种分区避免了 expert weight 的跨 PU 复制（因不沿 M 维分片），同时消除了 GeMM2→GeMM3 之间的跨 PU 通信（每 PU 已拥有所需的 W_3 slice 和对应的输入 slice）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MoE Layer Execution on Stratum NMP (16 PUs)
Input: X_t [M×K] (tokens in batch), expert routing IDs, gating weights w_e
Output: MoE layer output [M×K]

# Step 1: xPU sends X_t to Mono3D DRAM, switches to NMP mode
# Step 2: Sub-ring all-gather X_t to all 16 PUs

For each activated expert e (sequential):
  # Step 3-4: GeMM1 + GeMM2 (parallel on all PUs)
  For PU_i in [0..15] (parallel):
    Z_1[i] = X_t @ W_1[i]    # [M × N/P], PE Tensor Core 16×16 MAC
    Z_2[i] = X_t @ W_2[i]    # [M × N/P], parallel with Step 5
  
  # Step 5: Activation (overlapped with GeMM2)
  For PU_i in [0..15] (parallel):
    A[i] = SiLU(Z_1[i])              # Special Function Engine
    X_2[i] = A[i] ⊙ Z_2[i]           # Hadamard, no inter-PU comm needed
  
  # Step 6: GeMM3 (parallel on all PUs)
  For PU_i in [0..15] (parallel):
    Z_3[i] = X_2[i] @ W_3[i]   # [M × K/P], PE Tensor Core
  
  # Step 7: Reduce-scatter Z_3 across PUs via ring network
  # Each PU_i gets slice of final output
  # Overlapped with next expert's GeMM1 (pipeline)

# Step 8: Weighted sum of expert outputs
For PU_i in [0..15] (parallel):
  Y += w_e * Z_3_concat[i]   # Special Function Engine, on-the-fly

# Step 9: Write back to DRAM → exit NMP mode → xPU reads
```

关键 Pipeline 优化：
- GeMM2 || Activation（无数据依赖，并行执行）
- Reduce-Scatter(Expert N) || GeMM1(Expert N+1)（通信与计算 overlap）
- Weighted-Sum 在 expert 输出就绪后立即执行（minimize idle cycles）
- Input token 分片发送到各 DRAM channel → sub-ring all-gather（减少传输延迟）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Stratum 的张量并行专家处理的关键实现考量：(1) Sequential expert execution（非 parallel across experts）——因不同 experts 的 token count 不同会导致 PU 间负载不均，sequential 利用所有 PU 协作处理每个 expert 保证负载均衡；(2) Intra-PU matrix partitioning——PE 间主要按权重矩阵长边分片以最大化 tensor core utilization；(3) Communication-computation overlap——ring network 的 reduce-scatter 延迟被下一 expert 的计算完全隐藏（前提是 reduce-scatter latency ≤ GeMM1 latency）；(4) 输入 token 复制成本——all-gather X_t 到所有 PU 的 cost 被 amortize（因所有 activated experts 共用同一 X_t）。Cycle-level simulation 验证这些 optimizations 使 NMP 的 expert processing 未因为 sequential 而成为瓶颈。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

## Attention Processing with Head-Level Parallelism on NMP (NMP上的头级并行注意力处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Processing with Head-Level Parallelism 是 Stratum 将 Transformer 的 multi-head attention 计算映射到 NMP processor 上的调度策略。由于 attention heads 之间无数据依赖，可以高度并行执行。Stratum 将 16 个 PUs 划分为多个 PU groups（通过 ring topology 上的 neighboring PUs），每个 group 负责处理一组 attention heads。两个 heads 分配给同一 group 以支持 interleaved processing（一个 head 执行 MatMul 时另一个执行 Softmax）。Key/Value 矩阵沿 sequence length 维（而非 head dim）分片到 PU group 内的各 PU，因 sequence length（512-32k）远大于 head dim（64-128），partition along sequence length 提供更好的负载均衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Processing on Stratum NMP (8 heads, 4 PU groups, 2 heads/group)
# Assumes KV cache stored in intermediate-speed Mono3D DRAM tier

# For each PU Group (e.g., PUs 0-3, heads H0, H1):

# --- Head H0 processing ---
# Step 1: Sub-ring all-gather Query Q to all PUs in group
For PU_i in group (parallel):
  Q_i = all-gather(Q_slice)    # ring network, replicate full Q

# Step 2: Score = Q @ K^T (K partitioned along seq_len dim)
For PU_i in group (parallel):
  S_i = Q @ K_i^T              # [1 × S/P], PE Tensor Core (GeMV mode)
  # S_i contains scores for sequence range [i*S/P, (i+1)*S/P)

# Step 3: Softmax (3-stage with inter-PU scalar communication)
# Stage 1: Local max
local_max_i = row_max(S_i)     # Special Function Engine
global_max = ring_scalar_exchange_max(local_max_i)  # scalar only!

# Stage 2: Local exp_sum
local_exp_i = sum(exp(S_i - global_max))  # Special Function Engine
global_sum = ring_scalar_exchange_sum(local_exp_i)

# Stage 3: Normalize
S_soft_i = exp(S_i - global_max) / global_sum

# Step 4: O = Softmax(S) @ V (V partitioned along seq_len dim)
O_i = S_soft_i @ V_i           # [1 × d_head], PE Tensor Core

# Step 5: Reduce-scatter O across PUs in group
O = reduce_scatter(O_i)        # ring network

# --- Head H1 processing (interleaved with H0) ---
# While H0 is in Softmax Stages 1-2, H1 executes Step 2 (Scores = Q@K^T)
# While H0 is in Step 4, H1 executes Softmax Stages 1-3
# H1's reduce-scatter overlaps with: nothing (H0 already done)
# But H0's reduce-scatter overlaps with H1's Step 4 (Attn@V)

# Key: Softmax's inter-PU scalar communication (2 values per PU) is negligible
# compared to tensor data transfer, enabling clean interleaving.
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Stratum 的头级并行注意力的关键实现考量：(1) PU group formation——flexible grouping 使系统能适应不同的模型架构（如 MHA, GQA, MQA），每组大小根据 head count 和 request concurrency 动态调整；(2) K/V round-robin placement——新生成的 KV pairs 按 round-robin 分布到 group 内不同 PUs，避免单个 PU 的 KV cache size 过大；(3) Scalar-only inter-PU communication——Softmax 所需的 global max/sum 仅需标量交换（每 PU 2 个值，总共 8 values per group），通过 ring network 的标量通信通道完成，latency 极小；(4) Head interleaving——2 heads per group 的设计确保一个 head 的 Softmax（低计算强度，高延迟）被另一个 head 的 MatMul（高计算强度）完全隐藏。Multi-head scheduling 的全流程由 compiler 在 offline 时根据模型配置预计算并作为静态调度嵌入 NMP 的 finite state machine。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

## Expert Swap via Row-Swap Buffer (基于行交换缓冲区的专家交换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Swap via Row-Swap Buffer 是 Stratum 中执行 Mono3D DRAM 内部 tier-to-tier expert 参数迁移的硬件机制。当 serving scheduler 从一种 topic batch 切换到另一种 topic 时（如从 "math" 切换到 "code"），不同 topic 对应的 hot expert 集合不同，需要将原来在快 tier 的 expert（现已成为 cold）与原来在慢 tier 的 expert（现已成为 hot）交换物理存储位置。传统方法需通过 xPU→interposer→DRAM→interposer→xPU 路径进行数据搬移，延迟和能耗极高。Stratum 在每 PE 的 local memory controller 中内置 8KB row-swap buffer，支持同一 DRAM bank 内的 row-to-row data swap，无需 traversing interposer 瓶颈接口。交换时间 <0.37%，能量 <0.03‰。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Expert Swap between consecutive topic batches (in NMP mode)

# Before batch dispatch:
Input: old_topic (e.g., "math"), new_topic (e.g., "code")
       old_placement P_old, target_placement P_target from Algorithm 1

For each DRAM bank with conflicting placements:
  # Identify experts that need to swap tiers
  swaps = []
  For expert e in all experts:
    if P_old[e] in fast_tier and P_target[e] in slow_tier:
      # Was hot, now cold → evict to slow tier
      swaps.append((e, "evict"))
    elif P_old[e] in slow_tier and P_target[e] in fast_tier:
      # Was cold, now hot → promote to fast tier
      swaps.append((e, "promote"))
  
  # Execute swaps pair-wise (evict + promote = swap partners)
  For (e_cold, e_hot) in swap_pairs:
    # Step 1: Read row[cold_tier_addr] → Row-Swap Buffer
    local_mem_ctrl.read_row(cold_tier_addr, row_swap_buffer)  # 8KB per PE
    
    # Step 2: Read row[hot_tier_addr] → overwrite cold_tier_addr
    local_mem_ctrl.read_and_write(hot_tier_addr, cold_tier_addr)
    
    # Step 3: Write Row-Swap Buffer → hot_tier_addr  
    local_mem_ctrl.write_row(row_swap_buffer, hot_tier_addr)
    
    # Timing: each swap ~ tRC(cold_tier) + tRC(hot_tier) 
    # + Row-Swap Buffer access (negligible vs DRAM timing)
    # All within same bank → no cross-bank movement needed

# Per-benchmark overhead (Table 4):
# OLMoE: 5.91 swaps/sec, 0.64ms (0.37%), 0.25mJ (<0.02%)
# Mixtral: 2.59 swaps/sec, 0.90ms (0.23%), 0.35mJ (<0.03‰)
# Llama-4: 4.02 swaps/sec, 0.45ms (0.18%), 0.34mJ (<0.02‰)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Row-swap buffer 的关键设计考量：(1) 仅在 bank 内执行 swap（同一 bank 的快 tier 和慢 tier 区间），充分利用 Mono3D DRAM 的内部带宽（19-34 TB/s），避免穿越 interposer；(2) 8KB buffer 大小足够容纳一个 row-buffer 页面（32Kb = 4KB），double-buffered 支持连续 swap；(3) 由 local memory controller 的 programmable state machine 控制，无需 xPU 参与（由 scheduler 通过 command queue 触发）；(4) 仅使用 dedicated buffer（非 shared memory），避免与 tensor core 计算争抢 SRAM 带宽。主要局限：仅支持 bank 内交换（同一 bank 不同地址），若需跨 bank 移动 expert（tensor-parallel sharding 变化），需 traversing interposer；但 Stratum 的 tensor parallelism 策略保持 expert shard placement 在 banks 之间不变，仅调整同一 bank 内的 tier 归属。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving


## FlashMLA

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashMLA 是 DeepSeek 开源的高效 MLA (Multi-head Latent Attention) decode kernel，专为 Hopper 架构（H100/H800/H200）GPU 优化。GitHub: https://github.com/deepseek-ai/FlashMLA。FlashMLA 实现了 MLA 的 Layer Reordering 优化，在 decode 阶段的 Score 和 Context 计算中通过复用压缩 KV Cache ($\mathbf{C}_{\text{KV}}$) 将 ArI 从 ~100 Op/B 提升至 ~200 Op/B（翻倍）。核心优化：Score 层从 HBM 加载 $\mathbf{C}_{\text{KV}}$ 后，Context 层立即复用共享内存中的 $\mathbf{C}_{\text{KV}}$ 而不重新从 HBM 读取，将两次内存访问合并为一次。这与 FlashAttention 的 tiling + recomputation 思路类似但适配 MLA 的计算模式。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === FlashMLA Decode Kernel (简化的计算流程) ===
# 利用 Hopper GPU 的 TMA (Tensor Memory Accelerator) 和 WGMMA 指令

# 输入在 HBM: C_Q (B, 1536), C_KV (B, L, 512), W_DQ_i (1536,128), etc.
# 输出在 HBM: O (B, 16384)

# 分块策略: 将 L 维度分块, 顺序处理
for block_l in range(0, L, BLOCK_L):
    # 1. 异步加载 C_KV[:, block_l, :] 到共享内存 (TMA)
    #    C_KV_block: (B, BLOCK_L, 512)
    load_C_KV_async(C_KV_block)

    # 2. 计算 Score (使用 WGMMA)
    #    S_block = QW_i @ C_KV_block^T → (B, BLOCK_L)
    #    QW_i = Q_i @ W_DK_i^T → (B, 512) 在 kernel 开始时计算
    compute_score_block()

    # 3. Online Softmax (类似 FlashAttention)
    #    更新 running max 和 sum
    softmax_block = online_softmax_update(S_block)

    # 4. 复用 C_KV_block (已在共享内存) 计算 Context
    #    PV_block += softmax_block @ C_KV_block → (B, 512)
    #    *** 关键: C_KV 不需要再加载, 已经在 SMEM ***
    compute_context_block_with_reuse(C_KV_block)

# 5. 最终 Context: O_i = PV @ W_DV_i → (B, 128)
# 6. 合并所有 heads 的输出
```

核心性能增益：传统实现中 Score 和 Context 两层各需加载一次 $\mathbf{C}_{\text{KV}}$，FlashMLA 通过 tiling + fused kernel 将两次 HBM 访问合并为一次，ArI 翻倍。在 DeepSeek-R1 (d_KVco=512, n_hd=128) 上，ArI 从 ~100 提升至 ~200 Op/B，初步逼近 B200 的 Ridge Point (281.25)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashMLA 要求 Hopper 架构 GPU（SM90+），利用 CUDA 12.3+ 和 CUTLASS 库。安装使用：`pip install flash-mla` 或从源码编译。API 类似 FlashAttention：`flash_mla_fwd(q, c_kv, ...)`。已集成到 DeepSeek 官方推理代码和部分 vLLM/SGLang 版本中。仅适用于 decode 阶段（prefill 使用 reordering 无益，反而增加延迟）。论文中通过 FlashMLA 的优化效果验证了 reordered MLA 使 attention 不再需要专用 PIM 硬件的结论。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

## All-to-All Communication in MoE Expert Parallelism（MoE专家并行中的全交换通信）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
All-to-All Communication 是 MoE Expert Parallelism (EP) 中的关键集体通信操作，在两个阶段出现：(1) **Dispatch All-to-All**：在 expert 层执行前，token 按 router 选择的 expert 分配结果被发送到持有对应 expert 的 GPU；(2) **Combine All-to-All**：在 expert 层执行后，各 GPU 上的 expert 输出被发送回 token 原始所在的 GPU 进行 merge（如 Top-K 加权累加）。All-to-All 是 barrier 式通信——所有参与 GPU 必须同时调用通信原语，最慢的 GPU 决定整体通信完成时间。在 MoE 推理中，all-to-all 通信开销可占 DeepSeek-V2-Lite MoE layer forward latency 的 59.2%（Sem-MoE 数据），是 EP 在跨节点场景中的主要性能瓶颈。

从kernel调度角度拆解术语：
以 NCCL all-to-all 为例的 MoE EP 通信流程：
```
// Standard EP All-to-All flow (e.g., SGLang):
// 在每个 expert layer 前后:
//
// [Dispatch Phase]
// 每个 GPU 将 token 按目标 expert 分组:
send_counts[i] = 本GPU要发给GPU_i的token数
recv_counts[i] = 本GPU要从GPU_i接收的token数
// NCCL all-to-all scatter: 
//   GPU_j 发送 send_counts[j] 个 token embeddings 给 GPU_j
//   同时从每个 GPU_i 接收 recv_counts[i] 个 token embeddings
// → Barrier: 所有 GPU 必须完成此操作
//
// [Expert Compute]
// 每个 GPU 对收到的 tokens 执行本地 experts 的 FFN
//
// [Combine Phase]  
// NCCL all-to-all gather: 将 expert 输出送回原 GPU
// → Barrier: 所有 GPU 必须完成此操作
```

在 load skew 存在的情况下，hot expert GPU 处理大量 tokens 耗时远超 cold expert GPU。由于 all-to-all 是 barrier 操作，所有 GPU 必须在两个 barrier 点等待 hot expert GPU 完成计算和通信——这是 AEP 论文的核心动机（GPU stall 可占总时间的 70%）。

AMoE 的替代方案：**取消 barrier all-to-all，改用异步 P2P 通信**：
- Phase 1：ZeroMQ（CPU message queue）交换 metadata（tensor size, sender GPU rank）
- Phase 2：NCCL P2P（ncclSend/ncclRecv）直接 GPU-to-GPU tensor 传输
- CPU 启动 NCCL kernel 后立刻处理下一个传输任务（不等待完成）
- 接收方在将 tensor 交 Scheduler 前按需同步（cudaStreamSynchronize）

术语一般如何实现？如何使用？
主流框架实现：NCCL `alltoall` 或 `alltoallv`（支持不等长消息）；DeepSpeed-MoE 提供 hierarchical all-to-all（在节点内 NVLink all-to-all 和跨节点网络 all-to-all 间拆解）。ScaleMoE 揭示 EP 中 all-to-all 的 zero padding 问题（因 expert 选择不均衡，zero ratio 可高达 98%），提出 Adaptive All-to-All 通过精确 slice size 的 NCCL alltoallv 消除 padding。AMoE 则完全取消 all-to-all，改为异步 P2P，是另一种根本性的解决方案。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
- Sem-MoE: Semantic-Aware Model-Data Collaborative Scheduling for Communication-Efficient MoE Inference

## Asynchronous P2P Communication for MoE Serving（MoE推理中的异步点对点通信）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous P2P Communication 是 AMoE 系统中替代传统 EP 中 barrier all-to-all 的通信机制。不同于要求所有 GPU 同时参与的集体通信，AMoE 使用点对点（P2P）NCCL 传输，通过 ZeroMQ 消息队列协调发送方和接收方，实现完全异步的 token 传输。每个 GPU 可以向任意其他 GPU 独立发送/接收 token batch，无需任何全局同步。

从kernel调度角度拆解术语：
AMoE 两阶段异步 P2P 通信流程（Figure 8）：
```
// Phase 1: Metadata Exchange (CPU, ZeroMQ)
// Sender:
zeromq_send(receiver_rank, {tensor_size, tensor_dtype, src_gpu_rank})
// → Sender CPU 不等待，继续处理下一个任务

// Receiver:
metadata = zeromq_recv()  // 从消息队列消费
// 创建一个 size 匹配的 NCCL receive buffer

// Phase 2: Tensor Transfer (GPU, NCCL P2P)
// Sender GPU:
ncclSend(tensor_data, tensor_size, receiver_rank, comm, cuda_stream)
// → Sender CPU 回到 Phase 1 处理下一个传输

// Receiver GPU:
ncclRecv(recv_buffer, tensor_size, sender_rank, comm, cuda_stream)
// → Receiver CPU 回到 Phase 1 检查 ZeroMQ queue

// Before using received tensor (receiver only):
cudaStreamSynchronize(cuda_stream)  // 确保 NCCL 传输完成
```

关键设计：(1) 单线程 Communicator 可并发管理多个传输——每个传输的 CPU 侧启动 NCCL 后立即返回，GPU 侧异步执行；(2) Sender 不用同步——batch 发出后不再使用；(3) 接收方延迟同步——仅在 Scheduler 需要使用 tensor 前确保传输完成。

术语一般如何实现？如何使用？
AMoE 中 Communicator 在 C++ POSIX thread 中实现（避免 Python GIL），使用 NCCL 作为底层 GPU P2P 传输协议。与 NCCL 标准 P2P API 的挑战：(1) NCCL send/recv 需双方同时调用——通过 ZeroMQ 提前告知；(2) 动态 tensor size——ZeroMQ metadata 包含 size 信息。ZeroMQ 是开源通用消息库（zeromq.org），提供高性能异步消息队列。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

## Dynamic All-to-All in MoE Gating（MoE 门控中的动态全交换通信 / 可变大小 All-to-All）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic All-to-All in MoE Gating 是 Huang et al. (NeurIPS 2024) 在 "Toward Efficient Inference for Mixture of Experts" 中提出的 Dynamic Gating 机制中的通信模式创新。传统的 MoE Expert Parallelism 使用 NCCL all-to-all 在等大小消息的假设下（每个 expert 发送/接收固定 S×C tokens），因为所有 GPU 预知消息大小，可直接分配 buffer。Dynamic Gating 消除固定 capacity 后，消息大小变为可变——需要两轮 all-to-all：(1) 第一轮通知 sizes（每个 GPU 告知其他 GPU 将接收多少 tokens）；(2) 第二轮传输实际 tokens（可变大小，按第一轮获得的 sizes 动态分配 buffer）。

第一轮 all-to-all 仅传输标量整数（每个 expert device 对应一个 int64），平均延迟 20µs，几乎可忽略。第二轮 all-to-all 传输可变大小 token tensors（vs 静态的 zero-padding filled tensors）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// Dynamic Gating 的通信调度 (每个 GPU 执行)

// Phase 0: Local computation (gating + argsort + bincount)
vector<int> sizes(E, 0);  // sizes[e] = 本 GPU 将发送到 expert e 的 token 数
Tensor sorted_tokens = argsort_and_bincount(tokens, assignments, sizes);

// Phase 1: All-to-All Round 1 — Size Notification
// 本 GPU 需告知: "我将在 Round 2 向你的 expert e 发送 sizes[e] 个 tokens"
// 每个 GPU 发送 E 个 int64，接收 E 个 int64
NCCL_AllToAll(sizes.data(), sizes.data(), 1, ncclInt64, comm, stream);
// → 现在 recv_sizes[src_gpu * E + e] = GPU src_gpu 将向本 GPU 的 expert e 发送的 tokens 数
// 总 incoming tokens = sum(recv_sizes)
// → 预分配 recv_buffer[total_incoming_tokens * D]

// [期间可与其他计算 overlap — 论文提到 "latency hidden"]

// Phase 2: All-to-All Round 2 — Token Transfer
// 发送: sorted_tokens split by sizes[i % (E/P)] for each target GPU
// 接收: variable-length tokens into pre-allocated recv_buffer
NCCL_AllToAll(send_tokens, recv_tokens, send_counts, recv_counts, ncclFloat, comm, stream);
// send_counts, recv_counts 按 expert 分组，每 GPU one entry per target GPU

// Phase 3: Expert Computation
vector<Tensor> expert_outputs;
int offset = 0;
for (int e = 0; e < num_local_experts; e++) {
    int n = recv_counts[e];  // 实际收到的 tokens 数
    if (n > 0) {
        Tensor tokens_e = recv_buffer.slice(offset, offset + n);
        expert_outputs.push_back(FFN[e](tokens_e));
        offset += n;
    }
}

// Phase 4: All-to-All Round 3 — Output Collection
// 将 expert outputs 送回原始 GPU（sizes 对称，可用相同模式）
NCCL_AllToAll(output_tokens, returned_tokens, ...);
// → 按 inverse permutation 还原 token 顺序
```

对比 Static Gating 的单轮通信：
```
// Static Gating: 单轮 all-to-all (固定大小)
// 每 GPU 向每 target GPU 发送 exactly S×C tokens (包括 zeros)
NCCL_AllToAll(dispatched_tokens, received_tokens, 
              S*C*sizeof(float), ncclFloat, comm, stream);
// 通信量: E × S×C × D × sizeof(float)
// 其中大量 zeros → waste

// Dynamic Gating: 两轮 all-to-all (可变大小)  
// Round 1 通信量: E × sizeof(int64) ≈ 512 × 8B = 4KB (trivial)
// Round 2 通信量: 2S × D × sizeof(float) (仅实际 tokens, 无 waste)
// → 节约: (ECS - 2S) × D × sizeof(float) bytes
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NCCL `ncclAllToAll` 支持可变大小（通过 sendcounts/recvcounts 参数或 `ncclAllToAllv`）。关键实现注意事项：(1) Round 1 与 Round 2 之间可插入其他计算（如 token split/reorder）实现 overlap；(2) 第一轮的 20µs 开销在 multi-node 场景下依然 trivial（因此 multi-node scaling 表现更好）；(3) `ncclGroupStart/End` 可将 Round 2 的 send/recv 分组以减少 collective launch overhead。多节点时，减少的 all-to-all 通信量抵消增加的一轮通信开销，吞吐提升更显著（11.55× vs static）。

涉及论文标题：
- Toward Efficient Inference for Mixture of Experts
- Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

## Token Reorder via Indexing (vs Batch MatMul) in MoE Dispatch（基于索引的 MoE Token 重排 / 替代 Batch MatMul 的 Dispatch 实现）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Reorder via Indexing 是 Huang et al. (NeurIPS 2024) 提出的用高级索引（advanced indexing）替代 batch matrix multiplication (bmm) 来重排 MoE tokens 的 kernel 优化。在 Static Gating 中，token dispatch 和 reorder 通过构建稀疏 dispatch mask 矩阵并与 token tensor 进行 batch matmul 实现——O(S²EDC) 复杂度，大部分计算浪费在 ×0 操作上。论文提出用 torch.argsort + torch.index_select (或 Python advanced indexing) 直接重排 tokens——O(SD + S log S) 复杂度，纯 memory-bandwidth bound 操作。

从kernel调度角度拆解术语：

```
// === Static Gating: Batch MatMul Dispatch ===
// Input: tokens X ∈ R^{S×D}, mask M ∈ R^{E×S×S×C}
// M 极度稀疏: 仅 S×k 个 1s (k=top-k), 其余为 0
dispatched = torch.bmm(M, X)  // cuBLAS SGEMM kernel
// → GPU: SGEMM tiles, loads both X and M into shared memory
// → M 中大量零值被加载并参与乘加 → 浪费 FMA + bandwidth
// → S=8, E=512, D=1024: ~860M FLOPs, 92%为×0

// === Dynamic Gating: Indexing Dispatch ===
sorted_idx = torch.argsort(expert_ids)          // RadixSort kernel: O(S log S)
sorted_X = X[sorted_idx]                        // Gather kernel: O(SD) BW
sizes = torch.bincount(expert_ids, minlength=E) // Reduce kernel: O(S)
batches = torch.split(sorted_X, sizes)          // View/slice: O(1)

// → 无 matrix multiplication
// → Gather kernel: 仅读取 X[sorted_idx[i]] 写入 sorted_X[i]
// → Memory BW: 2×S×D×4 bytes (read X + write sorted_X)
// → 比较 Static bmm: S²EDC × 4 bytes (包括 zeros)
```

关键 Kernel 分析：
```
Advanced Indexing (X[sorted_idx]) GPU kernel:
  grid: ceil(S*D / block_size) blocks
  每 thread: load sorted_idx[tid/D] → compute X offset → load X[offset] → store
  
  vs Batch MatMul GPU kernel (cuBLAS):
  grid: 2D tiling over (E*S) × D
  每 block: load M tile + X tile → compute M·X → 大量 zeros
  Waste: tile overhead + FMA for zeros
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyTorch 实现：`dispatched = tokens[argsort(assignments[:, 1])]`。底层调用 GPU gather kernel（`index_select` 或 `take_along_dim`），memory-bandwidth bound（vs compute-bound batch matmul）。适用条件：当 S>32 时优势显著；当 S 极小时（<8），gather kernel launch overhead 可能超过 bmm 的固定开销。论文实验表明 batch=80 时 Dynamic Gating 优于 Megablock (BCSR sparse matmul) 1.46×，因 dense matmul + indexing 比 sparse matmul 在 GPU 上更高效。

涉及论文标题：
- Toward Efficient Inference for Mixture of Experts
- Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

## Token Deduplication in MoE Dispatch（MoE Dispatch中的Token去重）

术语是什么？
Token deduplication 是 GPU-initiated token-level EP 通信的优化技术：当 MoE router 将同一 token 分配给位于同一节点的多个 experts 时，通信库仅跨网络（RDMA）发送该 token 的 activation 一次，利用节点内高速互联（NVLink/xGMI）转发给同节点内的其他 expert GPUs，避免多次跨节点重复传输相同数据。

从kernel调度角度拆解术语：
```
// 无 dedup: 每个 (token, expert) pair 独立发送 (N=top_k 次 RDMA)
//   token_A → expert_0 (GPU_0, node_0): RDMA
//   token_A → expert_3 (GPU_3, node_0): RDMA ← 同一节点, 浪费带宽
//   token_A → expert_5 (GPU_4, node_1): RDMA

// 有 dedup (DeepEP/UCCL-EP HT mode GPU kernel):
//   GPU kernel 在提交 TransferCmd 前执行:
//     1. 按 dest_node 分组 topk_indices:
//        {node_0: [expert_0, expert_3], node_1: [expert_5]}
//     2. 每个 distinct dest_node 仅提交 1 个 Write TransferCmd:
//        token_A → node_0 (1 RDMA write)
//        token_A → node_1 (1 RDMA write)
//     3. 目标 node 接收后, intra-node forwarding:
//        GPU_0 → GPU_3 via NVLink ring buffer
//   结果: 2 次 RDMA (vs 3 次无 dedup), NVLink 带宽远高于 RDMA
```

术语一般如何实现？如何使用？
DeepEP 和 UCCL-EP 的 HT mode kernel 实现。GPU SM thread 在构造 TransferCmd 前检查 routing table 去重，将同一 token 去往同一节点的多个 expert destinations 合并为一条消息。UCCL-EP 此功能通过 CPU proxy 透明支持：GPU kernel 仍按去重后的策略提交命令，CPU proxy 按标准流程执行 RDMA 即可。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## Hierarchical Reduce in MoE Combine（MoE Combine中的层次化归约）

术语是什么？
Hierarchical reduce 是 GPU-initiated token-level EP 通信中 combine 阶段的优化：将 expert outputs 的 weighted sum 归约分解为两层——(1) intra-node reduce：在节点内对同一 token 的多份 expert output 先做本地加权归约；(2) inter-node reduce：归约后的结果通过 RDMA 发回原 token GPU 做最终归约。相比所有 expert GPU 各自独立发送 output，大幅减少跨节点网络传输量。

从kernel调度角度拆解术语：
```
// 无 hierarchical reduce:
//   每个选中的 expert GPU 独立 RDMA output 回原 GPU
//   原 GPU 收到 top_k 份 outputs 后做 weighted sum

// 有 hierarchical reduce (DeepEP/UCCL-EP HT mode):
//   Phase 1 (intra-node):
//     node_i 上所有 local expert outputs:
//       同 token 的 outputs 先做 local weighted sum
//       各 node 输出 1 份 intra-node reduced result
//   Phase 2 (inter-node via RDMA):
//     各 node intra-node result → RDMA → 原 token GPU
//   Phase 3 (final reduce on GPU):
//     原 GPU kernel 对 M 份 inter-node results 做加权 sum
//     M = distinct nodes among selected experts (M << top_k)
//   网络传输量: M 份 (vs top_k 份无 hierarchical reduce)
```

术语一般如何实现？如何使用？
在 UCCL-EP HT mode 中，GPU kernel 在 combine 阶段利用 routing metadata 判断哪些 expert outputs 分布在同一 node 内，先在 NVLink/xGMI domain 内完成 intra-node reduce，仅将结果通过 CPU proxy 发起的 RDMA 跨节点传输。此优化依赖 GPU-initiated fine-grained 通信能力：GPU kernel 需在 transfer 前读取并处理 routing 信息。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## CPU Proxy for GPU-initiated Communication（CPU代理驱动的GPU发起通信）

术语是什么？
CPU proxy 是 UCCL-EP 的核心架构组件：一组在 CPU 上运行的多线程代理，接收 GPU 通过 FIFO channel 发送的 TransferCmd（128-bit 紧凑命令），解析后通过 libibverbs（可移植 RDMA 库）发出 GPUDirect RDMA 操作，并负责强制执行 delivery semantics（ordering、completion fence、barrier）。每个 GPU 分配 1 个 CPU proxy（含 4 worker threads），不同 threads 无共享状态、无需同步。

从kernel调度角度拆解术语：
```
// CPU proxy thread 执行流程:
while True:
    // 1. Poll assigned FIFO channels
    for each channel in my_channels:
        cmd = Poll(channel)  // 读取但不弹出

    // 2. 根据 cmd.type 执行
    switch cmd.type:
        Write:
            wr = build_rdma_write(cmd.dst_rank, cmd.dst_offset,
                                  cmd.length, cmd.seq_num)
            imm = pack(cmd.seq_num, cmd.expert_idx)
            ibv_post_send(qp, wr, ibv_send_flags | IBV_SEND_SIGNALED, imm)
            Pop(channel)  // 可靠传输下入队后立即弹出

        Atomics:
            // EFA: 模拟 via immediate data → host memory counter
            // CX7: 使用硬件 RDMA atomics
            ibv_post_send_atomic(qp, ...)

        Drain:
            drain_cq_until(cq, cmd.idx)  // 等待所有 in-flight 完成

        Barrier:
            hierarchical_barrier(shm, leader)  // 节点内→跨节点同步

    // 3. Poll completion queue (non-blocking)
    for cqe in poll_cq():
        if IMM in cqe:
            seq = extract_seq(cqe.imm_data)
            if out_of_order(seq):
                buffer_in_control_buffer(cqe)  // 暂存, 顺序 apply
            else:
                apply_immediately(cqe)
```

术语一般如何实现？如何使用？
UCCL-EP CPU proxy 通过 libibverbs 抽象 NIC 差异，支持 CX7 EFA Broadcom 等 NIC。每 thread 管理一组 QPs（包括 QP load balancing across NICs），负责 polling sender CQ（确认发送完成）和 receiver CQ（处理到达的消息 + ordering enforcement）。CPU 利用率从 8%（无 UCCL-EP）升至 ~22%（4 threads），远低于 GPU 集群中 CPU 的可用核数（128-192 cores）。与 CPU-assisted IBGDA 的区别：UCCL-EP 使用多线程 proxy + multi-FIFO channels 实现 small-message scalability + heterogeneous NIC ordering emulation。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication

## Fused GEMM + Dequantize Kernel (融合GEMM与反量化内核)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused GEMM + Dequantize 是将量化权重的反量化（dequantization）操作融合进 GEMM 矩阵乘法 kernel 内部的技术。传统做法是先启动一个单独的 dequantize kernel 将 INT4/INT8 权重转为 FP16 写入 GPU 全局内存，然后再启动 GEMM kernel 读取 FP16 权重进行计算。这引入了额外的内存读写：write dequantized weights → read for GEMM。Fused 版本在 GEMM 的 weight tile load 阶段直接读取量化权重并在线程寄存器中完成 dequantize，避免中间的全局内存读写。对于 memory-bound 的 MoE GEMM 操作（加载多个 expert weights 受限于内存带宽）尤其重要。

论文 "Who Says Elephants Can't Run" 的核心洞察：profiling 发现 native CUDA IntToFloat (I2F) 是 fused kernel 的性能瓶颈，因此用 FP16 bit-trick 序列取代 I2F。

从kernel调度角度拆解术语：

INT8 Fused GEMM + Dequantize per-tile 计算过程（V100, SM70）：
```
for each K tile:                               // K dimension iteration
    // Load FP16 activations
    A_tile = load_to_smem(A, m_tile, k_tile)
    
    // Load INT8 weights + Fused Dequantize in registers
    w_packed = *reinterpret_cast<uint32_t*>(&W_plus[idx])
    // Extract bytes: e0,e1,e2,e3
    
    // Optimized INT8→FP16 (2 elements at a time):
    // 利用 FP16 性质: 0x6400 | X  = FP16(X+1024)
    R1 = construct_fp16_pair(e0+1024, e1+1024)  // 0x6400 | e1 << 16 | 0x6400 | e0
    R1_fp16 = R1_fp16 - [1152.0, 1152.0]         // = [e0-128, e1-128] as FP16
    // 1152 = 1024 (bias) + 128 (unsigned offset)
    W_deq = R1_fp16 * S[tile_n]                  // per-channel scale
    
    // FP16 Tensor Core GEMM accumulate
    acc += A_tile @ W_deq
```

INT4 优化变体：权重 layout 重排 `[e0..e7] → [e0,e2,e4,e6,e1,e3,e5,e7]`，减少 bit extraction 操作。减去常量从 1152 变为 1032 (=1024+8, offset=8)。

性能（V100, 32 active experts, 40 tokens）：INT8 optimized 1.59× FP16 baseline, INT4 optimized 1.85× FP16 baseline。对比 native I2F（INT8: 1.46×），optimized I2F 序列约 9% 额外加速。

术语一般如何实现？如何使用？

基于 CUTLASS 自定义 kernel，修改 GEMM 的 weight load "prologue" 阶段。需要在 CUTLASS collective builder / CuTe DSL 中定义自定义 weight tile load。关键考虑：(1) dequantize 计算不能成为新瓶颈（FP16 bit-trick 解决）；(2) INT4 需要特殊 weight layout 对齐 32-bit 加载；(3) 与 Grouped GEMM 配合时各 expert token 数不同（varlen-M）。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

## Sequential GeMM for MoE Expert Computation (MoE专家计算的顺序通用矩阵乘)

术语是什么？

Sequential GeMM 是 X-MoE 中用于替代传统 batched matmul 的 expert 计算方式。在 padding-free MoE pipeline 中，dispatch_out 是动态大小的 uneven token buffer（每个 expert 的 token 数量不同，且无 zero-padding）。Sequential GeMM 按 tokens_per_expert 数组将 dispatch_out 切片，依次为每个 expert 独立 launch 一个标准 GeMM。

从kernel调度角度拆解：

```
# dispatch_out: [Bexp, H], tokens_per_expert: [Elocal]
# w1[Elocal]: 每expert的第一层权重 [H, HFFN]
# w2[Elocal]: 每expert的第二层权重 [HFFN, H]

offset = 0
for i in range(Elocal):
    n_tokens = tokens_per_expert[i]  # expert i 的 token 数
    if n_tokens == 0:
        continue
    
    # 切片获取expert i的token
    expert_input = dispatch_out[offset : offset + n_tokens]  # [n_tokens, H]
    
    # 第一层FFN (可选activation)
    inter = matmul(expert_input, w1[i])  # [n_tokens, HFFN]
    inter = activation(inter)
    
    # 第二层FFN
    expert_output = matmul(inter, w2[i])  # [n_tokens, H]
    
    mlp_out[offset : offset + n_tokens] = expert_output
    offset += n_tokens
```

与 Grouped GEMM（Megablocks 方式）的对比：
- **Grouped GEMM**：单 kernel launch 并行计算所有 expert，但要求 padded equal-size blocks → zero-padding 开销
- **Sequential GeMM**：多次 kernel launch（Elocal 次），每次无 padding，expert 间无同步开销但 launch overhead 存在
- X-MoE 在 Small 模型上 expert computation 时间略增（因 sequential launch + data transform overhead），但总体 layer time 减少 62.3%（因消除了 zero-padding 的通信和内存收益）

术语一般如何实现？

在 X-MoE 中，Sequential GeMM 使用 Python for-loop 驱动 rocBLAS（AMD）或 cuBLAS（NVIDIA）的 GEMM 调用。每次 launch 处理一个 expert 的 tokens，GEMM 维度为 [n_tokens, H] × [H, HFFN] 或 [n_tokens, HFFN] × [HFFN, H]，其中 n_tokens 在各 expert 间通常不同。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## Uneven AlltoAll in MoE Training (MoE训练中的不等长全交换)

术语是什么？

Uneven AlltoAll（也称 alltoallv）是 MPI 风格的集合通信原语，允许每个参与方发送和接收不等量的数据。在 X-MoE 的 padding-free MoE pipeline 中，由于消除了 zero-padding，每个 expert 接收的 token 数量不同（由 tokens_per_expert ERI-array 描述），因此必须使用 uneven alltoall 替代传统 MoE 框架中的 even alltoall。

从kernel调度角度拆解：

传统 MoE 的 even alltoall vs X-MoE 的 uneven alltoall：

```
# Traditional (even alltoall):
# 所有expert的buffer固定大小 [E, C, H]
expert_buffers = alltoall(padded_buffers)  # 传输 E*C*H 个元素
# 大量zero-padding随通信传输

# X-MoE (uneven alltoall):
# Step 1: 先交换元数据
tokens_per_expert = alltoall(tokens_per_expert)  # [E] 整数，轻量
# 每rank据此计算inbound token数量 B_in = sum(tokens_per_expert[my_experts])

# Step 2: 交换实际数据
dispatch_out = alltoallv(dispatch_in, tokens_per_expert)  # 传输 B 个有效token
# dispatch_in: [B_out, H], dispatch_out: [B_in, H]
# 总通信量: B_out * H (无padding浪费)
```

通信量对比（Large 模型, EP=64, 256 GPU）：
- Even alltoall: 含大量 zero-padding，X-MoE 实测 alltoall 时间减少 50.7%
- Uneven alltoall: 仅传输有效 token，通信量 = 实际路由 token 数 × H

在 RBD 中进一步分层为 inter-node uneven alltoall（仅 pilot tokens）+ intra-node uneven alltoall（local replica）。

术语一般如何实现？

在 AMD ROCm 平台上通过 RCCL（ROCm Collective Communication Library）+ AWS-OFI-RCCL plugin（映射到 libfabric）实现。在 NVIDIA 平台上通过 NCCL 实现。X-MoE 使用 PyTorch 的 `torch.distributed.all_to_all_single` 配合 `split_sizes` 参数实现 uneven 传输。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## Triton Gather/Scatter Kernel for MoE (面向MoE的Triton收集/分散内核)

术语是什么？

X-MoE 为 padding-free MoE pipeline 实现了两个基于 Triton 的关键 kernel：Gather Kernel（dispatch 阶段，按 token_ids 从 gate_out 收集 token 到 dispatch buffer）和 Scatter Kernel（combine 阶段，按 token_ids 将 MLP 输出分散回原始序列位置并加权）。这两类 kernel 的核心挑战是嵌套索引访问（如 gate_out[token_ids[i], :]）导致的不规则内存访问模式。

从kernel调度角度拆解：

Gather Kernel 的实现策略（Triton）：

```
# 执行: dispatch_in[i, :] = gate_out[token_ids[i], :]
# 
# Launch: B 个 thread-blocks, 每 block 256 threads
# Block bi 负责复制第 bi 个token
#
# 伪代码 (per thread-block bi):
row_idx = token_ids[bi]           # 源token在序列中的位置
for j in range(0, H, 256):        # 沿hidden dimension循环
    thread_id = j + thread_idx    # 当前线程处理的hidden dim位置
    if thread_id < H:
        dispatch_in[bi, thread_id] = gate_out[row_idx, thread_id]
#
# Coalescing: 连续线程处理连续hidden dim位置
# 保证 gate_out[row_idx, :] 的读取是coalesced (同一row, 连续col)
```

Scatter Kernel 的实现策略：

```
# 执行: combine_out[token_ids[i], :] += mlp_out[i, :] * combine_weights[i]
#
# Scatter的不规则性在"写"端: 多个token可能写到同一行不同列
# 但每行不同列之间无依赖 → 仍可并行
#
# 伪代码 (per thread-block bi):
row_idx = token_ids[bi]
weight = combine_weights[bi]
for j in range(0, H, 256):
    thread_id = j + thread_idx
    if thread_id < H:
        # atomic add 或事先检查无冲突
        combine_out[row_idx, thread_id] += mlp_out[bi, thread_id] * weight
#
# Coalescing: 连续线程写入连续hidden dim位置 (同一row的连续列)
```

与 Megablocks 的对比：
- Megablocks 使用 block-sparse primitives，但仍需 padding 到固定 block size 的倍数
- X-MoE 的 Triton kernel 完全 padding-free，且跨平台（ROCm/CUDA 均支持）

术语一般如何实现？

使用 Triton 语言编写，编译为 AMD ROCm 或 NVIDIA CUDA 后端。关键优化：(1) 将线程映射到 model hidden dimension（外层维度），确保 coalesced memory access；(2) 每 token 一个 thread-block，B 个 block 并行。在 X-MoE 中 Gather kernel 加速 dispatch buffer 填充 35.7×（Small 模型 vs DeepSpeed-MoE 的 einsum dispatch）。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
