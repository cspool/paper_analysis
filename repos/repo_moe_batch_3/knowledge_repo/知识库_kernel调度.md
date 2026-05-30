## Kernel Fusion for Manifold-Constrained Hyper-Connections (mHC Fused Kernels)

术语是什么？

mHC 的 Kernel Fusion 是为缓解 n-stream 残差设计引入的显存带宽瓶颈而设计的专用融合 GPU kernel 集合。由于 HC/mHC 将残差流宽度扩展 n 倍，标准实现下每个 token 的显存 I/O 增加约 $(8n+2)C$ 个元素读/写（n=4 时约 33C vs 标准残差连接的 3C）。mHC 通过 5 个融合 kernel 将显存带宽利用优化到 n=4 时仅 6.7% 额外时间开销。

五个融合 kernel 分别处理不同的计算阶段：
1. **融合线性投影+Norm kernel**（Eq.14-15）：将两次对 $\vec{\mathbf{x}}_l$ 的扫描（RMSNorm r 的计算 + 线性投影 $\vec{\mathbf{x}}_l \varphi_l$）融合为单一 kernel，利用 MMA 单元最大化显存带宽。反向两个矩阵乘法同样融合为单 kernel。
2. **融合后处理 kernel**（Eq.16-18）：将小系数上的轻量操作（RMSNorm 归一化 + gating factor 乘法 + bias 加法 + Sigmoid 激活）融合为单一 kernel，减少 kernel launch 开销。
3. **Sinkhorn-Knopp kernel**（Eq.19）：20 次交替行列归一化在单 kernel 内完成。反向实现自定义 kernel，片上重计算中间结果。
4. **Pre 映射应用 kernel**：计算 $\mathcal{H}_l^{\text{pre}} \mathbf{x}_l$ 聚合 n-stream → 1-stream。
5. **Post+Res 融合应用 kernel**：将 $\mathcal{H}_l^{\text{post}}$ 和 $\mathcal{H}_l^{\text{res}}$ 的应用与 residual merge 融合——读取元素从 $(3n+1)C$ 降至 $(n+1)C$，写入从 $3nC$ 降至 $nC$。

从kernel调度角度拆解：

mHC kernel 执行流程（前向）：
```
// Kernel 1: Fused Linear Projection + Norm
// Input: x_l (n, C) bf16, phi (nC, n^2+2n) tf32
// Output: H_tilde (1, n^2+2n) f32, r f32
// Grid: (1,), Block: single wave
// Pipeline: load bf16 → cast f32 → MMA tf32 → store f32
x_flat = flatten_to_1d(x_l)         // in-register
r = norm(x_flat) / sqrt(n*C)        // fused into same kernel
H_tilde = x_flat @ phi               // MMA on tensor cores

// Kernel 2: Fused Post-processing
// Input: H_tilde, alpha scalars, bias vector
// Output: H_pre, H_post (1,n), H_res_raw (n,n)
H_scaled = (1/r) * [alpha_pre*H_pre, alpha_post*H_post, alpha_res*H_res] + bias
H_pre = sigmoid(H_pre_part); H_post = 2*sigmoid(H_post_part)

// Kernel 3: Sinkhorn-Knopp (single kernel)
// Input: H_res_raw (n,n); Output: H_res (n,n) ~doubly stochastic
M = exp(H_res_raw)
for t=1..20: M = col_norm(row_norm(M))

// Kernel 4: Pre Mapping Application
// Input: H_pre (1,n), x_l (n,C); Output: layer_in (C,)
layer_in = H_pre @ x_l  // reduction: n streams → 1

// [Standard layer computation: F(layer_in, W_l)]

// Kernel 5: Post+Res Fused Application with Residual Merge
// Input: H_res (n,n), x_l (n,C), H_post (1,n), layer_out (C,)
// Output: x_next (n,C)
// Fused: eliminates separate read/write of intermediate results
x_next = H_res @ x_l + H_post.T * layer_out
// I/O: reads (n+1)C, writes nC (vs separate: reads (3n+1)C, writes 3nC)
```

术语一般如何实现？如何使用？

大部分 kernel（除 Kernel 1 的 MMA 融合核外）使用 TileLang 框架实现，TileLang 简化了复杂计算过程 kernel 的实现。混合精度策略：输入 bfloat16、权重 tfloat32、计算 float32。精细的 load→cast→compute→store 流水线处理混合精度。反向 pass 中 mHC kernel 被选择性重计算（recompute）而非保存所有中间激活。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---

## Selective Recomputing / Gradient Checkpointing for mHC

术语是什么？

Selective Recomputing（选择性重计算）是在训练反向传播中，丢弃前向 pass 的中间激活并在反向 pass 中重新计算它们的策略，以显存换计算。mHC 由于 n-stream 设计导致中间激活量约为标准残差连接的 n 倍（每层额外存储 nC 元素的 stream + C 元素的层输入），选择性重计算将这些中间激活丢弃并在反向需要时重新执行 mHC kernel（不含沉重的层函数 $\mathcal{F}$）。

mHC 的重计算策略优化：对于 $L_r$ 连续层，仅需持久化首层输入 $\mathbf{x}_{l_0}$（nC 元素），中间层的 stream 和映射系数在反向中重计算。最优块大小：$L_r^* \approx \sqrt{nL/(n+2)}$，在实践中与 pipeline stage 中的层数对齐。总持久化存储：$\lceil L/L_r \rceil$ 个 $\mathbf{x}_{l_0}$，瞬态峰值：$(n+2)C \times L_r$。

从kernel调度角度拆解：

```
# Forward pass storage strategy
for each block of L_r consecutive layers:
    store x_{l_0}       # first layer input, nC elements, persistent
    store F_outputs     # every layer's F output, C elements each
    # DISCARD: intermediate x_l (nC), H_pre*x_l (C), RMSNorm result (C)
    
# Backward pass recomputation
for each block (reverse order):
    load x_{l_0}
    for l in l_0 .. l_0+L_r-1:
        # Re-execute mHC kernels (without heavy F)
        x_flat = flatten(x_l); r = norm(x_flat)/sqrt(nC)
        H_tilde = x_flat @ phi
        H_pre, H_post, H_res = manifold_project(H_tilde, r, alpha, bias)
        layer_in = H_pre @ x_l
        # Now compute gradients using stored F_output and recomputed intermediates
        ...
        x_l = H_res @ x_l + H_post.T * F_output  # recover next x_l
```

术语一般如何实现？如何使用？

PyTorch 中通过 `torch.utils.checkpoint` 实现，但 mHC 的自定义重计算策略更精细：重计算边界与 pipeline stage 对齐；仅重计算 mHC kernel 不含 $\mathcal{F}$（因为 $\mathcal{F}$ 的输出已存储）；重计算过程与 pipeline 通信解耦（首层激活已在本地缓存）。在 DualPipe schedule 中进一步将重计算与通信重叠。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

## Hierarchical AlltoAll Communication (层次化AlltoAll通信 / Resource-Aware Communication)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical AlltoAll 是 MoESys 针对 MoE 模型中 Expert Parallelism 的 AlltoAll 通信提出的两阶段网络拓扑感知通信优化。传统 AlltoAll 中，不同 rank 的 GPU 跨节点通信时经过 spine switch，造成路由冲突和带宽浪费。Hierarchical AlltoAll 将一次全交换拆为两个阶段：(1) **Intra-node AlltoAll**——利用 NVSwitch/NVLink（900GB/s）在单节点 8 GPU 间完成数据重排，将跨 rank 的数据搬运到同节点内同 rank 的 GPU；(2) **Inter-node AlltoAll**——将各节点中同一 rank（如所有 node 的 GPU0 组成一组、所有 node 的 GPU7 组成另一组）的 GPU 分组做跨节点 AlltoAll，同 rank GPU 的 NIC 接入同一 leaf switch，不经过 spine switch。

从kernel调度角度拆解术语：
Hierarchical AlltoAll 的通信调度流程（以 2 nodes, 16 GPUs 为例，单节点 8 GPU，目标是从 GPU0 Node1 发送数据到 GPU7 Node2）：
```
# Baseline AlltoAll 路径:
GPU0(Node1) → NIC1(rank0) → LE1 → SPq → LE1 → NICn(rank7) → GPU7(Node2)
# 经过 spine switch SPq，高延迟 + 带宽竞争

# Hierarchical AlltoAll 路径:
# Phase 1: Intra-node via NVSwitch
GPU0(Node1) --NVSwitch 900GB/s--> GPU7(Node1)  # 数据搬运到同 rank
# Phase 2: Inter-node via NIC grouped by rank
GPU7(Node1) → NIC7(rank7) → LE7 → NIC7(rank7) → GPU7(Node2)
# 仅经过 leaf switch LE7，不经过 spine switch
```

通信调度伪代码：
```
function HierarchicalAlltoAll(tokens_per_expert):
    # Phase 1: Intra-node
    for each GPU g in node:
        data_to_rank_r = tokens destined for expert on GPU with rank r
        NVSwitch_AlltoAll(data_to_rank_r)  # 在本节点内按 rank 重排
    
    # Phase 2: Inter-node (grouped by rank)
    for each rank r in 0..7:
        comm_group = all GPUs with rank r across nodes
        NIC_AlltoAll(comm_group, data_for_rank_r)  # 同 rank 组跨节点通信
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现的硬件前提：单节点内 GPU 通过 NVSwitch 全互联（900GB/s per GPU），节点间通过 NIC（100G/200G/400G Mellanox ConnectX）+ leaf/spine 交换机互联。
- 与 DeepSpeed 的 AlltoAll 优化的区别：DeepSpeed 通过 tensor fusion 将小 packet 合并为大 packet 解决 per-port 通信量小的问题，是通信 payload 层面的优化；Hierarchical AlltoAll 是利用网络拓扑的物理层次做路径选择优化，是通信 routing 层面的优化。两者正交互补。
- 性能提升：peer-to-peer 通信效率提升 p 倍（p=单节点 GPU 数）；80.7B model / 4 nodes 32 GPUs 下通信阶段加速 15.5%，端到端训练加速 10.3%。
- 局限性：该方案与网络拓扑强耦合——如果 cluster 改用 rail-optimized 或 fat-tree 拓扑，方案需要重新设计。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

## CUDA Pinned Memory for H2D/D2H Transfer (CUDA页锁定内存加速主机-设备传输)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA Pinned Memory (页锁定内存) 是 CUDA 提供的一种 host memory 分配方式（通过 cudaHostAlloc / cudaMallocHost），分配的内存被操作系统锁定在物理内存中（不会被 swap 到磁盘），使得 GPU DMA 引擎可以直接访问该内存区域进行数据传输，无需经过中间 bounce buffer 的额外 copy。默认的 malloc/new 分配的 pageable memory 在 H2D/D2H transfer 时，CUDA driver 需要先将数据 copy 到临时的 pinned staging buffer，再通过 DMA 传输，导致双倍内存带宽消耗和额外延迟。Pinned Memory 可以消除这一次额外 copy，并通过 cudaMemcpyAsync 在独立的 CUDA stream 上与 GPU kernel 执行重叠。

从kernel调度角度拆解术语：
MoESys 中使用 Pinned Memory + Async Copy 实现 H2D/D2H overlap 的 kernel 级调度：
```
// GPU Stream 0: Default compute stream
for layer i in model.layers:
    // 计算第 i 层
    launch_attention_kernel<<<grid, block, 0, stream0>>>(input_i)
    launch_moe_ffn_kernel<<<grid, block, 0, stream0>>>(expert_input_i)
    cudaStreamSynchronize(stream0)

// GPU Stream 1: Async copy stream
for layer i in model.layers:
    // 异步预取第 i+1 层 expert 参数
    cudaMemcpyAsync(
        dst = gpu_expert_params[i+1],
        src = cpu_pinned_expert_params[i+1],
        size = expert_param_size,
        kind = cudaMemcpyHostToDevice,
        stream = stream1  // 独立 stream，与 stream0 并行
    )
// stream0 和 stream1 并发执行：compute(stream0) || H2D_copy(stream1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 分配方式：`cudaHostAlloc(&ptr, size, cudaHostAllocDefault)` 或 `cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined)`（后者牺牲 read 性能换取更高的 PCIe write 带宽，适合纯 H2D 场景）。
- 注意：过多 pinned memory 会减少 OS 可用的 pageable memory 并可能导致系统不稳定，通常限制在物理内存的 25-50% 以内。
- 在 MoESys 中，pinned memory 用于 Ring Memory Offloading 和 2D Prefetch 中的 CPU→GPU 参数传输，是 computation-communication overlap 的底层支撑。
- 类似优化广泛应用于其他 MoE serving 系统：DeepSpeed-Inference、MoE-Infinity、Klotski 等均使用 pinned memory + async copy。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

## CPU Attention Kernel with AVX Intrinsics (for MoE Offloading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU Attention Kernel with AVX Intrinsics 是 MoE-GEN 实现的高性能 CPU 端 self-attention 计算内核，用于将 MoE 推理中的 attention mechanism（$QK^T$ 和 score-V 乘法）卸载到 CPU 执行。该 kernel 使用 Intel AVX（Advanced Vector Extensions）SIMD 指令集实现 Grouped Query Attention（GQA），采用 BF16 数据格式。关键设计点：(1) BF16 数据在 FP32 中表示，显式清零低 16 位尾数，所有计算和累加在 FP32 精度，每次点积累加后按 BF16 舍入规则舍入，保证与 PyTorch GPU attention 数值一致；(2) 优化 CPU cache 局部性，类似 FlashAttention CPU 版的设计思想；(3) 针对 GEMV（matrix-vector multiplication，解码阶段 attention 的算术特征）的算术强度进行优化，使 CPU 处理速度达到与 PCIe 4.0 传输 KV-cache + GPU 计算时间的可比水平。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU Attention Kernel 的伪代码（Grouped Query Attention, BF16 with AVX）：

```
// BF16 数据表示: 使用 FP32，尾数低16位置零
function bf16_mul_add(A_bf16[], B_bf16[], accum_fp32):
    // 加载 BF16 数据到 AVX 寄存器（256-bit YMM = 8×FP32）
    for i in 0..len step 8:  // AVX 一次处理 8 个 FP32
        a = _mm256_loadu_ps(&A_bf16[i])   // BF16 as FP32 (low 16 bits zero)
        b = _mm256_loadu_ps(&B_bf16[i])   
        accum_fp32 = _mm256_fmadd_ps(a, b, accum_fp32)  // FMA: a*b + accum
    
    // 舍入到 BF16: 保留高16位，低16位清零
    result_bf16 = accum_fp32 & 0xFFFF0000

// GQA Self-Attention（解码阶段，单 token，CPU path）:
function cpu_attention_gqa(Q, K_cache, V_cache, num_kv_heads, num_q_heads):
    // Q:    [num_q_heads, head_dim]    — 当前 token 的 single query
    // K_cache: [seq_len, num_kv_heads, head_dim]  — 历史 KV-cache
    // V_cache: [seq_len, num_kv_heads, head_dim]
    
    heads_per_kv = num_q_heads / num_kv_heads
    
    for g in 0..num_kv_heads:  // 每组 GQA group
        // Step 1: QK^T — score computation (GEMV)
        scores[0..seq_len] = 0
        for h in 0..heads_per_kv:
            q_head = Q[g*heads_per_kv + h]  // [head_dim]
            for pos in 0..seq_len:
                // AVX dot product: q_head · K_cache[pos, g]
                scores[pos] += bf16_mul_add(q_head, K_cache[pos,g], scores[pos])
        scores /= sqrt(head_dim)  // scaling
        
        // Step 2: Softmax
        max_score = max(scores)
        exp_sum = 0
        for pos in 0..seq_len:
            scores[pos] = exp(scores[pos] - max_score)
            exp_sum += scores[pos]
        for pos in 0..seq_len:
            scores[pos] /= exp_sum
        
        // Step 3: Score-V multiplication (GEMV)
        for h in 0..heads_per_kv:
            output[g*heads_per_kv + h] = 0  // [head_dim]
            for pos in 0..seq_len:
                // AVX dot-add: output += scores[pos] * V_cache[pos, g]
                output[g*heads_per_kv + h] = bf16_mul_add(
                    scores[pos], V_cache[pos,g], output[g*heads_per_kv + h])
    
    return output  // [num_q_heads, head_dim]
```

关键调度特性：
- **数据局部性**：KV-cache 在 host memory 中连续布局，CPU 顺序访问，L2/L3 cache 命中率高。
- **零拷贝**：CPU kernel 直接读取 host memory 中的 KV-cache，无需 PCIe HtoD copy。
- **与 GPU 并行**：GPU 处理 $(1-\omega) \cdot b_a$ 个 token 的 attention（需等 HtoD KV-cache copy），CPU 同时处理 $\omega \cdot b_a$ 个 token（无需等待 copy）。结果在 Post-Attention 阶段 concatenate。
- **Overlap 收益**：CPU attention 节省的 HtoD 带宽被 expert weight prefetch 利用，减少 GPU idle time。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实际实现和使用：
- **MoE-GEN 实现**：C++ 编写，AVX intrinsics（`_mm256_*` 指令），BF16 格式。约 3000 行 C++ 后端代码的一部分。当前仅支持 greedy decoding。
- **类似工作**：
  - **PowerInfer**：使用 CPU 执行部分 attention 计算，利用 consumer GPU + CPU 混合推理。
  - **Fiddler**：CPU-GPU orchestration for MoE，CPU 执行 attention/experts 以缓解 I/O 瓶颈。
  - **KTransformers**：Intel AMX（Advanced Matrix Extensions）CPU kernel，在 Sapphire Rapids 及以上 Xeon 上实现 21.3 TFLOPS（3.9× PyTorch native），相比 AVX 支持的 BF16 更高效。
  - **llama.cpp**：通过 BLAS backend 支持 CPU attention，但未针对 MoE 解码的 GEMV 特性优化 cache 行为。
- **何时使用**：CPU attention kernel 适用于 memory-bound 的 MoE offloading 场景（PCIe 带宽饱和时）。MoE-GEN 的 search procedure 自动决定最优 $\omega$：若 CPU kernel 执行时间 < GPU attention + KV-cache HtoD copy 时间，则 $\omega > 0$；若 CPU 计算能力弱（如 C3 的 16-Core CPU），$\omega$ 调低或为零。
- **数值精度**：BF16 在 FP32 中模拟（低 16 位清零），保证与 GPU BF16 attention 数值一致性，无需特殊 CPU 硬件支持（兼容旧 CPU）。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

**MoE-Lens 补充**：MoE-Lens 同样实现了手工优化的 CPU decode attention kernel，使用 **AVX512 SIMD intrinsics**（512-bit ZMM registers，一次处理 16 个 BF16 元素，upconvert 到 FP32 后为 8 elements per register）。相比 MoE-GEN 的 AVX（256-bit YMM），AVX512 提供双倍寄存器宽度。MoE-Lens 的优化包括：(1) manual vectorization 替代编译器自动向量化；(2) loop unrolling 减少分支和循环开销；(3) data prefetching 指令提前将下一轮 KV cache 数据加载到 CPU cache；(4) BF16→FP32 upconvert 和 FP32→BF16 rounding 每一步显式处理。在 Intel Platinum 8380 CPU 上，单线程 throughput 是 auto-vectorized baseline 的 4.7×，全线程为 3.1×（>20 threads 后因 memory controller contention 饱和）。该 kernel 在 VSLPipe 的 CPU Task (C) 阶段执行，与 GPU Task B 的 GEMM 并行。Kernel 的 throughput requirement 来自 Equation 6：$T_{CPU} = 2 \cdot s \cdot I_{CPU\_attn} \cdot B_{KV}$，需达到数百 GFLOPs 以满足 system target（当 KV cache = 2× model size 时）。

## Contiguous Data Mover (for MoE Weight Transfer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Contiguous Data Mover 是 MoE-Lens 提出的专用 CPU→GPU 权重传输模块，以独立线程运行（C++ PyTorch extension），负责在 CPU-GPU 混合 MoE 推理系统中高效搬运模型权重。其设计动机是解决 head-of-line blocking：如果将所有权重传输 API 调用嵌入执行流水线中，大批量 weight transfer 会阻塞延迟敏感的 compute transfer（如 PyTorch 操作、attention 同步数据），导致 GPU stall。Contiguous Data Mover 将 weight transfer 从执行流水线中解耦——执行引擎以 layer-wise granularity 推送传输请求，data mover 内部以 fine-grained 小 packet（100MB）分批执行传输，避免与其它 CPU-GPU transfer 竞争。100MB packet size 是 trade-off：足够大以最大化 PCIe bandwidth utilization，又足够小以避免 head-of-line blocking 和过长 latency。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Contiguous Data Mover 的调度伪代码：

```
// Data Mover 线程（独立于 compute 线程）
shared queue<TransferRequest> request_queue;  // layer-wise requests
const PACKET_SIZE = 100 * 1024 * 1024;  // 100MB

function data_mover_thread():
    while not done:
        // 1. 从执行引擎接收 layer-wise 传输请求
        Request req = request_queue.pop()
        
        // 2. 将 layer weights 分区为 100MB packets
        packets = partition(req.weight_data, PACKET_SIZE)
        
        // 3. 逐个 packet 异步传输
        for packet in packets:
            // cudaMemcpyAsync 到 GPU Weight Buffer
            event = cudaMemcpyAsync(
                dst: gpu_weight_buffer[req.layer_slot],
                src: cpu_pinned_weight[req.layer_id][packet.offset],
                size: packet.size,
                stream: mover_stream  // 独立 CUDA stream
            )
            record_event(event)
    
    // 4. 在每个 stage boundary 同步（不等待每个 phase）
    synchronize_stream(mover_stream)
```

关键调度特性：
- **独立 CUDA stream**：data mover 使用独立 stream，不与 compute stream 的 PyTorch operation 竞争。
- **100MB packet 粒度**：在 bandwidth utilization 和 interference minimization 之间折中。论文实证 100MB 为最优值。
- **Stage-boundary synchronization**：data mover 仅在 stage boundary（而非 phase boundary）同步——VSLPipe 中每个 stage 有 CPU phase + GPU phase，data mover 在整个 stage 期间异步运行，仅在 stage 结束时同步确保下一 stage 的 weights 就绪。
- **Weight Buffer 管理**：GPU 端 Weight Buffer 大小为 $2 \times$ per-layer weight size（通常仅为 model size 的 2-3%），实现双缓冲——一块用于当前 computation，另一块用于 prefetch 下一 layer/group 的 weights。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：C++ PyTorch extension，约数百行 C++ 代码。使用 CUDA Runtime API 的 `cudaMemcpyAsync` 和 `cudaStream_t`。
- **与执行引擎的交互**：执行引擎（VSLPipe）在每个 stage 开始时，通过 request_queue 推送下一 stage 所需的 layer weights 和 expert weights 的传输请求。Data mover 内部调度所有 packets，在 stage 结束前完成全部传输。
- **与 CPU Attention 的带宽竞争**：当 KV cache 较大（210GB）且 generation length 较大（256 tokens）时，CPU attention 需要扫描大量 KV cache blocks，与 data mover 竞争 CPU memory bandwidth，导致 weight transfer 时间从 ~5s 增加到 ~6s（论文 §8.2）。这是 MoE-Lens 当前的主要性能瓶颈之一。
- **Weight 存储**：所有权重存储在 pinned CPU memory 中，分为 layer-wise weights（attention projection matrices + normalization）和 expert weights（MoE layer 的 expert-specific 参数）。Transfer 以 layer-wise granularity 请求，内部分区为 packets。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

## Expert Weight Prefetching with HtoD Overlap（专家权重预取与 HtoD 重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Weight Prefetching with HtoD Overlap 是指在 MoE offloading 场景中，利用 GPU 计算当前 expert 的时间窗口，通过 HtoD engine 异步将下一个 expert 的权重从 host memory 预取到 GPU 的 expert buffer（$S_{Expert}$）中，使 GPU 计算和 PCIe 数据传输完全重叠，消除 GPU idle time。关键前提是 expert batch size 足够大（图 3 Right：需要 $>2^{11}$ tokens per expert），使得 GPU 计算时间 ≥ HtoD copy 时间，才能完全隐藏传输延迟。

从kernel调度角度拆解术语：

```
Expert 顺序执行的 kernel 调度时间线 (gantt):

section Expert 1
Compute Expert 1 :a1, 0, 5ms
section Expert 2  
HtoD Prefetch Expert 2 :a2, 1ms, 4ms
Compute Expert 2 :a3, 5ms, 5ms
section Expert 3
HtoD Prefetch Expert 3 :a4, 6ms, 4ms
Compute Expert 3 :a5, 10ms, 5ms

HtoD Prefetch Expert 2 与 Compute Expert 1 重叠 (时间 1-5ms)
HtoD Prefetch Expert 3 与 Compute Expert 2 重叠 (时间 6-10ms)

GPU 无 idle time —— 计算和传输完全 overlap
```

调度伪代码（MoE layer, expert 阶段）：
```
function execute_experts_sequential(B_tokens, experts[E]):
    // B_tokens: accumulated batch after router
    // S_Expert: GPU buffer for one expert's weights
    token_map = router.dispatch(B_tokens)  // token → expert mapping
    
    // Prefetch first expert
    async_htod_copy(experts[0].weights, S_Expert)
    
    for e in 0..E-1:
        wait_htod_copy()           // 等待当前 expert weights 传输完成
        tokens_e = token_map[e]    // 分配给 expert e 的 tokens
        
        if e < E-1:
            // 在计算当前 expert 时，同时预取下一个
            async_htod_copy(experts[e+1].weights, S_Expert2)  // double buffer
        
        // GPU compute: expert e forward pass
        gate_weight = experts[e].gate  // router 输出的 weight
        output[tokens_e] += gate_weight * experts[e].ffn(input[tokens_e])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **Double buffering**：使用两个 expert buffer（$S_{Expert}$ 和 $S_{Expert2}$），当前 expert 用 buffer A，HtoD 预取下一个 expert 到 buffer B，交替使用。
- **Buffer size trade-off**：$S_{Expert}$ 越大，可预取更多 expert weights（减少 on-demand wait），但挤压 attention batch size 和 KV-cache buffer。MoE-GEN 的 DAG scheduler 搜索最优 $S_{Expert}$。
- **与 Dense module 的区别**：Dense module（attention weights, shared expert）token 数固定，buffer 只需大小为单层 dense module。MoE-GEN 实证验证更大的 dense buffer 不会增加 throughput（Section 4.2）。
- **其他系统的预取策略**：
  - **MoE-Infinity**：使用 activation-aware predictor 预测将被激活的 expert，提前预取。
  - **MoE-Lightning**：profiling-based 的 memory movement schedule，通过 CUDA stream 实现 overlap。
  - **Pre-gated MoE**：在 gating 之前使用轻量级 predictor 提前预取 expert weights。
- **大 batch 下的简化**：大 batch（$B$ 大）下 token 均匀分配到各 expert，MoE-Gen 无需 predictor，直接顺序执行 experts 并预取下一个。这避免了 prediction miss 带来的 on-demand fetch 延迟。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

从kernel调度角度拆解术语：
以 MoDES 中 Group GEMM 执行一个 token 的 multi-expert 计算为例：

```
# 输入
x: token hidden state, shape [1, d]                     # 单 token (decoding)
kept_experts: 保留的 expert indices, 数量 k' <= k
W_experts: shape [k', d, d_ff]                          # k' 个 expert 权重矩阵

# === Group GEMM kernel (单次 launch) ===
grid_dim = num_SMs                                    # 或按 workload 调优
block_dim = (tile_M, tile_N) 的 thread block

for each SM in parallel:
    # 每个 SM 处理一组 sub-tile
    expert_idx = assigned_expert_range[sm_id]
    token_range = assigned_token_range[sm_id]

    for tile_m in token_range step tile_M:
        for tile_n in [0..d_ff] step tile_N:
            for tile_k in [0..d] step tile_K:
                # K 维度 reduce
                acc[tile_m][tile_n] +=
                    x[tile_m, :] @ W_experts[expert_idx][:, tile_n:tile_n+tile_N]

# 输出: y = sum(pi[i] * expert_out_i for each expert i)
# shape [1, d]
```

与 naive 逐 expert 执行的对比：
- **Naive (k 次 kernel launch)**: for each expert → launch GEMM(expert_i, x) → sync → 总耗时 = k × (kernel_time + launch_overhead)
- **Group GEMM (1 次 launch)**: 所有 k' 个 expert 在一个 kernel 内以 group sub-task 形式并发执行 → 总耗时 = max(kernel_time) + 1 × launch_overhead

MoDES 的关键实现细节：
1. **离线 profiling**：对不同代表性 activation pattern（不同 token 数 × 不同保留 expert 数组合），做 grid search 确定最优 tile size（tile_M/N/K），保证动态 workload 下的最高计算吞吐。
2. **Skipped expert 过滤**：在 dispatch 阶段，哨兵 ID（M+1）的 expert 被过滤，不进入 Group GEMM 的 sub-tasks 列表，减少无效计算。
3. **Decoding 阶段限制**：Decoding 仅处理 text token 且为 memory-bound（auto-regressive, seq_len=1），加速比低于 prefilling（batch_size=8, seq_len>1, compute-bound）。

术语一般如何实现？如何使用？
- **DeepGEMM** (DeepSeek)：CUDA JIT 编译的 FP8 Group GEMM 库，~300 行核心 kernel。支持 contiguous layout（训练/prefill，M 维分组）和 masked layout（decoding，CUDA Graph 兼容）。H100 上可达 1346 TFLOPS。
- **Triton persistent kernel** (PyTorch 官方博客)：grid 设为 SM 数量实现 persistent kernel（避免 wave quantization），配合 TMA（Hopper）和 grouped launch ordering 优化 L2 cache reuse。比手动 PyTorch loop 快 2.62×。
- **PyTorch `torch.nn.functional.grouped_mm`**：官方 API，支持 jagged token counts per expert，CUDA SM≥80 (BF16)。
- **MoDES 用法**：Group GEMM 与 GMLG/DMT 正交——GMLG/DMT 决定跳过哪些 expert，Group GEMM 加速保留 expert 的执行。离线 profiling 确定 tile size，在线推理无额外开销。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

## Fused AR-A2A Communication Algorithm

术语是什么？
Fused AR-A2A Communication Algorithm 是 MixServe 提出的将 All-Reduce (AR) 和 All-to-All (A2A) 两种 collective communication 算子融合并异步重叠执行的通信算法。核心思路是将 AR 分解为 Reduce-Scatter (RS) + All-Gather (AG) 两个子操作，将 A2A 分解为 Dispatch + Combine 两个子操作，然后重组为 RS→A2A→AG 三段式通信流程，利用异步机制使 intra-node 通信（RS/AG，高带宽）与 inter-node 通信（A2A Dispatch/Combine，低带宽）在时间上重叠，从而隐藏低带宽链路的延迟。该算法包含两个变体：(1) Fused RS-Combine（Alg 1），将 intra-node RS 与 inter-node A2A Combine 异步重叠；(2) Fused AG-Dispatch（Alg 2），将 intra-node AG 与 inter-node A2A Dispatch 异步重叠。

从kernel调度角度拆解术语：
以 4 节点、每节点 m 个 GPU/NPU、n_node=4 为例的 Fused RS-Combine 伪代码：

```
Require: n-node cluster, m GPUs/NPUs per node; input X [bs/d_EP, h] per node; global rank r
Ensure: output Y [b/d_DP, s, h] per node
1: Y = empty(b/d_DP, s, h)
2: [X_1,...,X_m] = split(X, m, dim=-1)  // 沿 hidden dim 切 m 份
3: r_TP = r mod m  // 计算 TP group 内 rank
4: S_1 = X_{r_TP}  // 暂存本地分片
5: for i = 1 to n-1 do async:  // inter-node A2A pairwise (异步)
6:     r_to = (r_TP + i*m) mod mn
7:     isend(X_{r_TP}, r_to)   // 发送到下一节点同 TP rank
8:     r_from = (r_TP - i*m) mod mn
9:     S_{i+1} = irecv(r_from)  // 从上一节点同 TP rank 接收
10: for i = 1 to n do async:  // intra-node RS + top-k 加权 (异步)
11:     S_i = await reduce_scatter(S_i, TP_group)
12:     Y_i = Y_i + topk_weights(S_i)  // 累加加权结果
13: Y = all_gather(Y_{r_TP}, TP_group)  // 最终 intra-node AG 汇总
```

Fused AG-Dispatch 同理：将 intra-node AG 与 inter-node Dispatch 重叠。首轮 pairwise 和末轮 AG 不可重叠，其余 n_node-2 轮通信完全重叠。时间复杂度 O(n_node)，RS-Combine 空间复杂度 O(bsh·n_proc)，AG-Dispatch 空间 O(1)。

术语一般如何实现？如何使用？
- MixServe 基于 vLLM（Ascend 910B）和 Tutel（H20）实现，通过向 MoE model 的 forward method 注入 RS/AG/A2A 通信算子。
- 异步机制：使用多个 CUDA/HCCL stream 并行执行 intra-node RS/AG 和 inter-node A2A isend/irecv，通过 await 同步点确保数据一致性。
- 性能收益：在 Ascend 910B 上，DeepSeek-R1 TTFT 加速 2.67× vs vLLM TP+PP，1.70× vs vLLM DP+EP；H20 上 Throughput +50.3% vs vLLM TP+PP。消融实验（Fig. 12）显示异步重叠的收益约等于 inter-node 通信开销。
- 适用场景：多节点 MoE 推理服务，intra-node 带宽显著高于 inter-node 带宽（如 NVLink 900 GB/s vs InfiniBand 50 GB/s，或 HCCS 60 GB/s vs RoCE 25 GB/s）。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

---

## All-to-All Communication (Pairwise Algorithm)

术语是什么？
Pairwise All-to-All 是 All-to-All collective communication 的一种经典实现算法。在 MoE 的 Expert Parallelism 中，A2A 用于将 tokens 从各 GPU dispatch 到拥有对应 expert 的 GPU（Dispatch），并在 expert FFN 计算后将结果 collect 回原 GPU（Combine）。Pairwise 算法需要 N-1 轮通信（N 为参与设备数），每轮中每个 rank 向特定目标 rank 发送数据并从特定源 rank 接收数据，形成成对交换。MixServe 利用 Pairwise 算法的多轮特性，在每轮中将 intra-node RS/AG 与 inter-node send/recv 重叠执行。

从kernel调度角度拆解术语：
以 N=4 设备为例的 Pairwise A2A 通信流程：
```
Round 1: rank0↔rank1, rank2↔rank3  (step=1)
Round 2: rank0↔rank2, rank1↔rank3  (step=2)
Round 3: rank0↔rank3, rank1↔rank2  (step=3)
```
每轮通信量 O(bs/d · hk)（b=batch, s=seq_len, h=hidden_dim, k=top-k, d=degree）。总通信量 ∝ (size/degree) × (degree-1)。

与 Ring All-to-All 对比：
- Ring：数据沿环形链路传递，N-1 步，每步传输 size/N 数据，总传输量 = size × (N-1)/N
- Pairwise：每步直接 send/recv，N-1 步，总传输量 = size × (N-1)/N，但每步的通信对可以并行（利用 full-duplex 链路）

术语一般如何实现？如何使用？
- NCCL/HCCL 实现：NCCL 的 All-to-All 通过 P2P send/recv 组合实现（因无原生 A2A 原语），支持 Pairwise 和 Ring 两种算法。
- MixServe 中的使用：Pairwise 算法的多轮特性使每轮中的 intra-node RS/AG 可与本轮 inter-node send/recv 重叠。Fused AR-A2A 算法要求使用 Pairwise（而非 Ring）以确保每轮的 send/recv targets 可预测。
- 关键参数：d（参与设备数）越大则轮数越多、通信开销越大。MixServe 通过 hybrid TP-EP 将 d_EP 降至 n_node（而非纯 EP 的 n_node × n_proc），减少轮数。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

---

## All-to-All Communication in MoE Training

术语是什么？
All-to-All 是 MoE 训练中 Expert Parallelism 的核心通信原语。在 MoE 的每一层，Router（门控网络）为每个 token 分配目标 expert 后，需要通过 All-to-All 将 tokens 从当前设备 dispatch 到拥有对应 expert 的设备，并在 expert 计算完成后通过第二次 All-to-All 将处理后的 tokens collect 回原设备。这一过程涉及两个 All-to-All 阶段：(1) Dispatch：将 tokens 按路由决策分散到各 GPU（T_I → T_DI）；(2) Combine：将 expert 输出收集回原设备（T_DO → T_O）。NCCL 没有原生 All-to-All 原语，PyTorch 通过 `torch.distributed.all_to_all_single` 配合 `input_split_sizes`/`output_split_sizes` 参数实现，底层由 NCCL 的 P2P send/recv 组合实现。

从kernel调度角度拆解术语：
MPMoE 中的 All-to-All 沿 batch 维度切分为多个 micro All-to-All（每个 micro-batch 一次独立 All-to-All），而非 FasterMoE 的沿 node 维度切分（降级为多组 P2P 通信）。这两种方式的区别：

```
// FasterMoE方式: 按 device 维度切分 (Figure 5a)
将 N 个 devices 分为 m 个 groups
for each group g (size G):
    for each partition p in {1..m}:
        在 group g 内部执行 P2P dispatch/recover
// 问题: (m-1) 次 NCCL group calls，退化为 P2P

// MPMoE方式: 按 batch 维度切分 (Figure 5b)
将 B 个 tokens 切分为 n 个 micro-batch，每个大小 B/n
for each micro-batch i in {1..n}:
    全局 All-to-All dispatch(T_I[i])  // 保留 NCCL 优化
    全局 All-to-All collect(T_DO[i])
// pipeline 交替调度 S 和 R stage 以增强内存局部性
```

MPMoE 方式的优势：(1) 保留 NCCL 对 All-to-All 的 ring/tree topology 聚合优化；(2) pipeline granularity n 不受 device 数 N 限制（batch size B >> N）；(3) 异构带宽下不会因同步等待浪费资源。

术语一般如何实现？如何使用？
- PyTorch 实现：`torch.distributed.all_to_all_single(output, input, output_split_sizes, input_split_sizes)`
- 底层 NCCL 实现路径：基于 `ncclSend`/`ncclRecv` 的非对称 P2P 通信，通过预设的 send/recv counts 协调。对于 MoE 的不均匀 token 分布，需先交换 metadata（各 rank 的 send/recv counts），再执行实际数据传输。
- 性能关键点：(a) 消息大小：token 数 × hidden_dim × sizeof(fp16)，小消息时 latency-bound，大消息时 bandwidth-bound；(b) 网络拓扑：NVLink 节点内延迟 ~10μs，InfiniBand 跨节点延迟 ~1-2μs + 带宽共享；(c) 不均匀 token 分布导致某些 link 成为 bottleneck。
- MPMoE 的通信效率验证：micro-benchmark（Figure 13）显示 MPMoE 的 dispatch/recovery 时间明显低于 FasterMoE（因避免 P2P 拆解的 kernel launch 开销和组同步等待）。

MixNet 从通信拓扑角度揭示了 EP all-to-all 的三个关键动态特性（基于生产环境测量）：
- **时间非确定性**：每个 training iteration 中 token routing 的结果不同，导致 all-to-all 通信矩阵在 iterations 间变化。即使 load balancing loss 使 overall volume 收敛，traffic matrix 的 sparsity 仍然持续。
- **空间非均匀性（sparse all-to-all）**：每个 traffic matrix 中仅有少数 GPU 对之间有大量通信，大部分 pairs 之间流量很小或为零。这种稀疏性源自 MoE 的 sparse activation（每个 token 仅激活 top-k expert）。
- **强局部性**：仅同一 MoE block 内的 expert 层需要 all-to-all，不同 PP stage 的 expert 层不直接通信。

基于这些特性，MixNet 设计了 topology-aware EP routing：优先将通信密集型 GPU 对通过 OCS 直连电路传输（专用高带宽、无排队），其余 pairs 走 EPS fallback。路由依赖 5 步流程：(1) topology lookup 确定 delegation GPU → (2) intra-host gather via NVSwitch → (3) inter-host all-to-all via OCS（优先）或 EPS → (4) intra-host all-to-all via NVSwitch → (5) intra-host scatter。步骤 (3) 和 (4) 通过 CUDA stream overlap 并行执行以减少总完成时间。OCS 重配置利用 all-to-all 通信之间的 computation phase（>100ms）隐藏延迟（~25ms）。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

---

## CUDA Stream Overlapping for Computation-Communication Overlap

术语是什么？
CUDA Stream Overlapping 是利用 CUDA 的异步执行模型，将计算（computation kernel）、通信（NCCL collective）、内存拷贝（cudaMemcpyAsync）分配到不同的 CUDA stream 中并行执行，使 GPU 的 SM（计算单元）、copy engine（内存拷贝引擎）和 NIC（网络接口卡）同时工作。在 MoE 训练中具体体现为：All-to-All dispatch/collect（通信）与 Expert FFN 计算（计算）的 overlap，以及 CPU offload 场景下 D2H/H2D 内存拷贝（mem）与前两者的 overlap。

从kernel调度角度拆解术语：
MPMoE 定义了 α(y, x) slowdown 因子量化并行操作间的干扰（Section 2.3），将操作分为三类流：comm（通信）、comp（计算）、mem（内存拷贝）：

```
// 基准：单独执行各操作
W_comp(B) // 处理 B 个 tokens 的计算时间
W_comm(B) // All-to-All B 个 tokens 的通信时间
W_mem(B)  // D2H/H2D B 个 tokens 的拷贝时间

// 并行执行时的 slowdown（α > 1 表示变慢）
实际 comp 时间 = W_comp / α(comp, comm)  // 通信对计算的干扰
实际 comm 时间 = W_comm / α(comm, comp)  // 计算对通信的干扰
实际 mem 时间  = W_mem / α(mem, comm)    // 通信对内存拷贝的干扰

// MPMoE micro-benchmark 观测（Figure 3）:
// α(comp, x) ≈ 1.0（计算几乎不受影响，因为SM和copy engine/NIC独立）
// α(comm, mem) < 1.0（通信和内存拷贝共享 memory bandwidth，互相干扰严重）
// α(comm, comp) 需 > 0.5 才能获得正向 overlap 收益
```

MPMoE 的性能模型（Section 4.2）利用这些 α 因子，在 3 种 pipeline paradigm（图 8）中估算各阶段（P0-P4）的真实执行时间。每个 paradigm 的瓶颈由最大瓶颈 CUDA stream 决定：
- Paradigm 1（仅 comp+comm，适用 S4）：`T_P2 = max((t_S+t_R)/α(comm,comp), t_C/α(comp,comm))`
- Paradigm 2（前向+mem copy，适用 S1/S2/S3 前向）：M 依赖 S 和 C 的输出
- Paradigm 3（后向+mem copy，适用 S1/S2/S3 后向）：C 依赖 M 的输入

术语一般如何实现？如何使用？
- 实现要点：(1) 使用非默认 CUDA stream（`cudaStreamCreate`），避免 default stream 的隐式同步；(2) `cudaMemcpyAsync` 需 pinned memory（`cudaMallocHost`）；(3) NCCL 通信通过 `ncclGroupStart/End` 在指定 stream 上执行；(4) 深度优先 issue order（先提交一个 stream 的所有操作，再提交另一个 stream）通常比广度优先更能实现良好 overlap。
- MPMoE 中的应用：S1/S2 策略需要 3 个 CUDA stream（comp、comm、mem）；S4 仅需 2 个（comp、comm）。S2 在 N 大时（如 64 GPU）性能恶化，因为 mem copy 和 comm 共享 memory bandwidth（α(comm,mem) 显著小于 1）。
- 局限性：(a) kernel 资源饱和时无法 overlap（如 SM 全占满时计算 kernel 已用完 GPU）；(b) 小数据量时 kernel launch overhead 抵消 overlap 收益；(c) 不同 GPU 架构的 copy engine 数量不同（A100 有 1 个 copy engine 但支持双向并发）。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---

## Pipeline Paradigm and Performance Model for Multi-Stream GPU Execution

术语是什么？
Pipeline Paradigm 是 MPMoE 为不同类型的内存复用策略抽象出的 3 种流水线执行模式，每种模式由不同的 CUDA stream 依赖关系定义。Performance Model 基于这些 paradigm 和 piecewise 速度函数，在运行时估算不同配置 (n, S) 的端到端执行时间，实现无需 profiling 的自适应配置选择。

从kernel调度角度拆解术语：
MPMoE 的 Performance Model 构建分三步：

```
// Step 1: Piecewise 速度函数（Figure 9）
// 小数据量时硬件利用不足，速度低于峰值；大数据量时速度饱和
W_comm(volume) = {
    k1_comm * volume,  if volume < V_threshold_comm  // 线性增长段
    k2_comm * volume,  otherwise                      // 饱和段 (k2<k1)
}
// 通过一次微基准 profiling 获得分段参数

// Step 2: Pipeline Paradigm 时间估算（Figure 8）
// Paradigm 1（仅 S+C+R，适用 S4 全阶段）:
//   P0: 仅 1 个 stream 工作，初始阶段
//   P1: 逐步启动所有 stream，饱和过渡阶段
//   P2: 所有 stream 饱和运行（可能有多个 P2）
//   P3: 逐步关闭部分 stream，熔化阶段
//   P4: 仅 1 个 stream 收尾
// 每个阶段执行时间 = 该阶段瓶颈 stream 的执行时间 / α(干扰因子)

// 以 Paradigm 1 的 P2 阶段为例:
t_S = W_comm(B/n)  // 单个 micro-batch 的 dispatch 时间
t_C = W_comp(B/n)  // 单个 micro-batch 的 expert 计算时间
t_R = W_comm(B/n)  // 单个 micro-batch 的 collect 时间
T_P2 = max(
    (t_S + t_R) / α(comm, comp),  // 通信流的瓶颈
    t_C / α(comp, comm)            // 计算流的瓶颈
)

// Step 3: 总时间汇总
T_total(n, S) = T_P0 + T_P1 + (n-3) * T_P2 + T_P3 + T_P4  // n 够大时
```

MPMoE-pm 通过此模型在无 profiling 开销下（<1% overhead）估算所有 (n, S) 组合的耗时，选择 T_total 最小的配置。

术语一般如何实现？如何使用？
- 适用场景：(a) 网络环境稳定时（如 Valor 集群），performance model 可替代 profile-based search；(b) 生产环境中避免每次配置变更都 profiling；(c) 作为 profile-based 方法的 warm-start（先用 model 估算，再用 profiling 微调）。
- 局限性：(a) 网络波动大时（如 Adira 集群）模型精度下降，MPMoE-pb 更优；(b) 依赖 α 因子的准确性，不同 GPU 架构和 NCCL 版本可能需要重新 calibrate；(c) 对极细粒度 pipeline（n>8）的 kernel launch overhead 建模不够精确。
- MPMoE-pm 的效果：比 MPMoE-pb 平均速度损失约 6-7%（1.66× vs 1.55× vs FasterMoE），但 profiling overhead 从 ~3% 降至 <1%。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---

## Block-Sparse Matrix Multiplication (SDD/DSD/DDS) for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-sparse matrix multiplication 是 MegaBlocks 用于替代 MoE 中 batched GEMM 的核心计算原语。使用三字母记法（源自 Triton Blocksparse, Tillet et al. 2019）描述稀疏-密集矩阵乘法：每个字符表示输出/左输入/右输入，S=Sparse, D=Dense, T=Transpose。SDD（Sparse = Dense × Dense）即采样密集-密集矩阵乘法（SDDMM），输出为稀疏矩阵；DSD（Dense = Sparse × Dense）和 DDS 是两种不同的稀疏-密集矩阵乘法（SpMM）。在 MoE FFN 前向传播中：第一层 expert 用 SDD（稀疏输出 = 密集 tokens × 密集权重 w1），第二层用 DSD（密集输出 = 稀疏中间结果 × 密集权重 w2）。向后传播需 SDD^T、DS^T D、DSD^T、DD^T S 四种操作。

从kernel调度角度拆解术语：
MegaBlocks SDD kernel 的 CUDA 伪代码（对应图 11）：
```
__global__ void sdd(Matrix a, Matrix b, SparseMatrix c) {
    // (1) 加载 non-zero block 坐标
    int row    = c.row_idxs[blockIdx.x];    // BCOO 行索引
    int column = c.column_idxs[blockIdx.x]; // BCSR 列索引
    // 每个 threadblock 处理一个 128×128 non-zero block

    // (2) 零初始化 accumulator (128×128 tile)
    Tile<128, 128> tile_c(0);

    // (3) Main loop: n_k 维度以 128 步进
    for (int i = 0; i < n_k; i += 128) {
        Tile<128, 128> tile_a = LoadTile(a, row, i);
        Tile<128, 128> tile_b = LoadTile(b, i, column);
        tile_c += tile_a * tile_b;  // Tensor Core MMA (m=128,n=128,k=128)
    }

    // (4) 写结果到 sparse output 的对应 non-zero block
    StoreTile(tile_c, c);
}
```

DSD kernel（对应图 12）：每个 dense output tile 启动 1 个 threadblock，按 BCSR row offsets 迭代对应 row 的 non-zero blocks，从 each non-zero block 的 column_idx 确定加载 b 的哪一行。128×128 block size 基于 A100 的 CUTLASS tile dimension benchmark 选择（图 5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 基于 NVIDIA CUTLASS 2.5 扩展实现，利用其 tile-based Tensor Core GEMM 基础设施。Kernel 启动策略：SDD → 每 non-zero block 1 个 threadblock；DSD → 每 dense output tile 1 个 threadblock。
- cuSPARSE blocked-ELL 格式要求所有 row 等量 non-zeros（与 MoE 负载不均衡冲突）。Triton Blocksparse 假定稀疏拓扑在迭代间不变（与 MoE 每 iteration 变化的动态路由冲突）。MegaBlocks 的自定义 kernel 专为动态拓扑设计。
- 在 MoE workload 上平均达到 cuBLAS 密集 GEMM 98.6% 吞吐量（标准差 4%，范围 91%-104%，图 9）。
- Hopper GPU (H100) 上推荐使用 Grouped MLP（grouped GEMM）替代 Sparse MLP（block-sparse），因 Hopper 的 grouped GEMM 性能更优。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

---

## Blocked-CSR-COO Encoding (Hybrid Sparse Matrix Format)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Blocked-CSR-COO 是 MegaBlocks 设计的混合块稀疏矩阵编码（图 6）。以 Blocked Compressed Sparse Row (BCSR) 为主格式，额外物化每个 non-zero block 的行索引（row_idxs），使 BCSR 同时具备 Blocked Coordinate (BCOO) 格式的随机访问能力。BCSR 天然高效支持按行迭代（DSD/DDS 操作），但 SDD 并行化需要知道每个 output block 的行坐标——纯 BCSR 需要搜索 row_offsets。通过物化 row_idxs（每 128×128 block = 16384 非零值仅需 1 个索引），SDD kernel 中 threadblock 可 O(1) 直接定位其 non-zero block 坐标。

从kernel调度角度拆解术语：
```
// Blocked-CSR-COO 数据结构（图 6）：
// BCSR 部分（主格式，按行压缩）
row_offsets:   [0, 3, 5, 8, ...]  // row i 的 non-zero blocks 起始偏移
column_idxs:   [0, 2, 4, 1, 3, ...]  // 每个 non-zero block 的列索引

// BCOO 部分（额外物化，SDD 并行化需要）
row_idxs:      [0, 0, 0, 1, 1, ...]  // 每个 non-zero block 的行索引
// row_idxs[i] 指示第 i 个 non-zero block 位于 matrix 的哪一行

// SDD kernel 利用 BCOO 行索引:
// 直接 O(1) 获取坐标，无需搜索
row = c.row_idxs[blockIdx.x];
col = c.column_idxs[blockIdx.x];

// 对比：纯 BCSR 的 SDD 需要搜索
// for row in 0..n_rows:
//     if blockIdx.x < row_offsets[row+1]: break
// 在高稀疏度（MoE 中 >90% zeros）下此搜索开销显著
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Metadata 存储开销 <0.1%（因 large block size，128×128 = 16384 个 FP16 值 = 32KB 仅需 4 字节 row index）。
- 同一数据结构的 BCSR 部分用于 DSD（按行迭代），BCOO 部分用于 SDD（坐标访问），无需维护两套格式。
- MoE 场景中稀疏拓扑随每 iteration 变化，make_topology CUDA kernel 每层每 iteration 重建完整元数据（O(num_experts + num_tokens/128)）。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

---

## Transpose Indices (Secondary Index for Sparse Matrix Transposition)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transpose Indices 是 MegaBlocks 用于高效支持 block-sparse 矩阵转置访问的技术（§5.1.4）。在模型训练的前向+后向传播中，需要混合转置和非转置的 block-sparse 操作（SDD^T, DS^T D, DSD^T, DD^T S）。纯 BCSR 按行存储，在转置顺序下迭代（按列访问）需要搜索所有行来查找目标列中的 non-zero blocks（Buluç et al. 2009）。MegaBlocks 避免显式转置稀疏矩阵（需 O(nnz) 数据复制），而是仅构造转置元数据：等效 BCSC（Blocked Compressed Sparse Column）编码的 column_offsets + 转置顺序的 non-zero block 偏移索引数组（transpose_indices）。kernel 通过 transpose_indices 的间接索引在转置顺序下迭代矩阵，类似数据库的 secondary index。

从kernel调度角度拆解术语：
```
// Transpose Indices 数据结构:
// column_offsets: [0, 2, 5, 7, ...]  // BCSC: 每列 non-zero blocks 的起始偏移
// transpose_indices: [3, 7, 1, 5, 0, ...]  // 转置顺序下的 block 偏移
//   transpose_indices[k] = 原 BCSR 中的第 k 个（转置顺序）non-zero block 的存储偏移

// 在 DSD^T kernel 中使用:
for (int i = 0; i < nnz_in_transposed_row; i++) {
    int blk_offset = a.transpose_indices[col_offset + i];  // 间接索引
    Tile<128,128> tile_a = LoadTile(a, blk_offset);
    // ... 其后与标准 DSD 相同
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 存储开销可忽略：仅 column_offsets（~num_experts 个整数）和 transpose_indices（~nnz 个整数），总 metadata <0.1%。
- 与 BCSR-COO 元数据在 make_topology kernel 中同时构造，摊销到 forward+backward 共 6 次 block-sparse 操作。
- 间接访问降低了 DS^T D/DD^T S 操作的空间局部性（<10% 吞吐量损失），但这些 weight gradient 操作仅占端到端训练时间的小部分。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

---

## M2N Communication Library for MoE Token Dispatch

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
M2N Communication Library 是 MegaScale-Infer 中为 MoE 推理场景的 M（attention GPU 数量）×N（expert GPU 数量）非对称通信模式专门设计的高性能通信库（~4900 行 C/C++ + ~5000 行 Python PyTorch extension）。在 Disaggregated Expert Parallelism 中，每个 MoE layer 需将 token embeddings 从 M 个 attention GPU 发送到 N 个 expert GPU，再从 N 个 expert GPU 返回 M 个 attention GPU——这是 many-to-many 通信模式，不同于 NCCL all-to-all（等量对称）。使用 GPUDirect + RDMA write with immediate + CUDA stream blocking（cuStreamWaitValue32）+ GDRCopy flush 消除 NCCL 的三大开销：GPU-to-CPU 中间拷贝、group initialization/closing、GPU synchronization instability。256KB data size 下 vs NCCL：68.2% median latency 降低、92.9% P99 latency 降低、4.2× throughput 提升。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
M2N Sender（Attention GPU → Expert GPU）：
```
// 1. cudaEventSynchronize 等待前序 kernel 完成
// 2. cuStreamWaitValue32(stream, &flag, EQ, 0) 阻塞 stream
// 3. CPU Core Sender: for each receiver in N:
//      ibv_post_send(qp[i], RDMA_WRITE_WITH_IMM, gpu_buffer+offset, len)
//    GPUDirect: 数据从 GPU 显存直接经 NIC 发出，无 CPU buffer 拷贝
// 4. while (ibv_poll_cq(cq, &wc) == 0) spin;  // 确认远端写入完成
// 5. flag = 1;  // 唤醒 CUDA stream
```

M2N Receiver（Expert GPU 侧）：
```
// 1-2. 同上 event wait + stream block
// 3. poll CQ 确认数据到达
// 4. gdr_copy_to_mapping(...)  // GDRCopy flush: 清除 GPU L2 stale cache
//    （RDMA 直接写 GPU 显存绕过 L2 → 需 flush 保证后续 kernel 读最新数据）
// 5. recv_flag = 1;  // 唤醒 stream, Expert FFN kernel 执行
```

NCCL 额外开销（M2N 消除）：(a) GPU→CPU proxy buffer copy；(b) batch-of-8 group operation 限制；(c) general collective setup/verification；(d) GPU sync 引发 P99 instability（NCCL P99 >1000μs, M2N <100μs at 32 receivers）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 依赖：CUDA driver API、libibverbs、GPUDirect RDMA、GDRCopy、NVIDIA RDMA completion queue。
- CPU vs GPU 通信（vs DeepEP）：M2N 用 CPU 控制 inter-node 通信，单线程在 ~256KB/pair 下饱和带宽，不占 GPU SM；DeepEP 用 GPU SM 并行管理 QP，需 PTX 优化避免 L2 cache 争用。MegaScale-Infer 场景（每 pair 几百 KB）CPU 方案更优。
- Traffic optimizations：(a) ACK 高优先级队列隔离（避免 ACK 被 data 阻塞）；(b) 拥塞控制微调（适应不均衡流量）。
- M/N scaling：8×8→32×32 均保持 3.3-5.8× throughput 优势 vs NCCL。

涉及论文标题：
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---

## Fused Kernel for Communication-Computation Overlap in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Kernel 是 MegaScale-Infer 中两类 kernel fusion 优化：(1) TP Communication-Computation Fusion：使用 Flux（ByteDance kernel fusion 库）将 tensor parallelism 的 all-gather/reduce-scatter 与相邻 GEMM 融合为单 CUDA kernel，利用 NVLink P2P load 实现 zero-copy；(2) Sequential Memory-Intensive Operator Fusion：将 gating + top-k selection + per-expert count + token scatter 等多个 memory-bound 操作融合为单 kernel，消除多次 kernel launch 和中间 global memory access。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TP All-Gather + GEMM Fusion（Flux 风格）：
```
__global__ void fused_allgather_gemm(input_chunk, weight, output) {
    // Phase 1: NVLink P2P load 从 peer GPU 读取远程 chunk 到 shared memory
    // Phase 2: 直接在 shared memory 上执行 GEMM Tensor Core MMA
    // 无需等待 All-Gather 完成——通信与计算在寄存器/shared mem 级别融合
}
```

Sequential Operator Fusion（Gating + Top-K + Scatter）：
```
__global__ void fused_gating_pipeline(h, W_gate, tokens, expert_inputs) {
    // 1. Per-token gating: scores = dot(h[tid], W_gate)  // registers
    // 2. Top-K partial sort in registers
    // 3. Atomic scatter to expert buffer (shared mem + global atomics)
    // 4. Shared memory count reduce to global
    // 4+ kernel launches → 1 kernel launch, 无中间 global memory roundtrip
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Flux：ByteDance 开源的 CUDA kernel fusion 库，在 kernel 内使用 NVLink P2P ld_volatile 从 peer GPU 加载数据。
- 适用：intra-node TP（NVLink 高带宽低延迟），跨节点不适合（InfiniBand 延迟高、无 P2P load）。
- 效果：未单独评估，综合在 MegaScale-Infer 整体性能中。

涉及论文标题：
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---

## FineEP All-to-All Communication Group Expansion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FineEP All-to-All Communication Group Expansion 是 FineMoE 中为支持 token scheduling 而扩展 all-to-all 通信组大小的设计。传统 EP 中 all-to-all 在 EP group 内执行（大小为 EP_degree），token 只能在 EP group 内的 GPU 间 dispatch。FineEP 将通信组扩展为 FineEP group（大小为 d × EP_degree，其中 d ≤ DP_degree/EP_degree），允许 token 在更大的 GPU 集合间进行跨 EDP group 调度。通信组扩大 d 倍意味着：调度空间从 1 个 replica/expert 扩展到 d 个 replica/expert（EDP）；但同时通信量可能增加（intra-node → inter-node 转换）。

从kernel调度角度拆解术语：
通信组扩展对 all-to-all 的影响：
- 通信量：每 token 仍需发送到 1 个 replica（top-2 时 2 tokens × 1 dispatch），通信量不变（每个 token 只发给 1 个 replica）。
- 通信模式：d× group size 的 all-to-all（NCCL 或 DeepEP），大 group 的 collective 可能有更多并行度但也有更多同步开销。
- Inter-node 转换：当 d×EP_degree > GPUs per node 时，部分 intra-node 通信变为 inter-node（额外延迟）。但两种情况影响小：(1) d×EP_degree ≤ GPUs/node → 全部 intra-node；(2) EP_degree 极大（如 DeepSeek-V3 的 64）→ 几乎全是 inter-node。

术语一般如何实现？如何使用？
- 在 Megatron-LM 中通过修改 NCCL communicator 大小（从 EP group → FineEP group）实现。
- Locality-Aware Routing（Algorithm 1）配合减少实际跨 GPU 通信量。
- 支持两种 backend：NCCL（默认）+ DeepEP（高性能 all-to-all）。
- 注意事项：(a) FineEP + DeepEP 的数据格式不兼容会引入额外 pre-processing 开销；(b) Pipelining FineEP（Appendix A.2）可进一步隐藏通信延迟。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

## Expert Tensor Parallelism (ETP)（专家张量并行）

术语是什么？
Expert Tensor Parallelism (ETP) 是 Tensor Parallelism (TP) 在 MoE 层的特定形式。与标准 TP 切分 Attention 层的 hidden dimension 不同，ETP 在 MoE 的 expert FFN 内部进行张量切分——每个 expert 的权重矩阵沿 hidden/intermediate dimension 被切分到多个 GPU 上。ETP 与 EP 构成二维网格 [EP, ETP]，即 experts 先按 EP 分布式放置，每个 expert 内部再按 ETP 切分。

从 kernel 调度角度拆解术语：
在 MoE Parallel Folding 框架中，MoE 层的 ETP 通信流程（以 ETP=2, EP=2, 4 GPU 为例）：

```
Forward Pass:
1. Router: 本地 token → expert assignment + permutation
2. All-to-All-V (跨 EP 组): token 发送到对应 expert 所在 rank
3. AllGather-V (跨 ETP 组): ETP 组内广播，确保所有 rank 持有完整 activation
4. Expert GEMM: 各 rank 计算其分配的 weight partition
5. ReduceScatter-V (跨 ETP 组): 聚合分发输出 hidden states
6. All-to-All-V (跨 EP 组): token 返回原始 rank
7. Unpermutation: 恢复 token 顺序

Backward Pass: AG/RS 互换为 RS/AG
```

通信量对比：
- **ETP 通信**：AllGather + ReduceScatter = 2 × bsh (n-1)/n，通信量与 TP 相同
- **EP 通信**：2 × All-to-All = 2 × (k/n) × bsh (n-1)/n，其中 k 为 top-k
- 当 k < n 时，EP 通信量小于 ETP；但 fine-grained MoE 中 k 大且 expert hidden size 小，ETP 通信占比可达 70%+

术语一般如何实现？如何使用？
- ETP 在 Megatron-Core 中通过 moe_groups["TP"] 实现，其 degree = etp
- 当需要将大 expert 切分到多 GPU 以减少单 GPU 内存压力时使用 ETP
- MoE Parallel Folding 允许将 ETP 替换为 EP（设置 etp=1），将通信从 AG/RS 转为 A2A，对 fine-grained MoE 特别有效
- 实现使用 NCCL AllGather-V 和 ReduceScatter-V 集合通信

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

## Context Parallelism (CP)（上下文并行）

术语是什么？
Context Parallelism (CP) 是一种将长序列的 token 沿序列维度切分到多个 GPU 上的并行策略，使每个 GPU 仅处理序列的一个子段。CP 的核心通信原语是 All-to-All 风格的序列重分布（如 DeepSpeed-Ulysses 的 all-to-all 或 Ring Attention 的 P2P 环形传递）。在 MoE 模型训练中，CP 与 EP 的组合尤为关键——当 CP×EP group 超过 NVLink 域时，通信开销会急剧增加。

从 kernel 调度角度拆解术语：
以 DeepSpeed-Ulysses 风格的 Sequence Parallelism 为例（在 Attention 层替代 TP）：

1. **输入切分**：将长度为 S 的序列沿序列维度切分为 cp_size 份，每 GPU 处理 S/cp_size 个 token
2. **All-to-All**：将 (S/cp_size, d) 的输入从序列分片转为 head 分片 → (S, d/cp_size)
3. **Attention 计算**：各 GPU 在 head 子集上独立执行 QKV projection + attention
4. **All-to-All**：将 head 分片转回序列分片 → (S/cp_size, d)
5. **输出 MLP**：各 GPU 计算其子序列的 output projection

CP 的通信量 = 2 × bsh (n-1)/n × (2 + 2/m)/n（m 为 GQA group 数），当 m=4 时约为 TP 的 1/4。

在 MoE Parallel Folding 中的关键作用：
- CP 切分的序列在进入 MoE 层前通过 reshape 展平为 token batch（零通信开销）
- MoE Parallel Folding 允许 CP 和 EP 组折叠在一起，使 EP 的 All-to-All 优先使用 NVLink 而非 InfiniBand
- 当 CP×EP > 8 时，无 Folding 的 EP A2A 走跨节点 InfiniBand，延迟显著上升；Folding 后保持稳定

术语一般如何实现？如何使用？
- 适用于序列长度 > 8192 tokens 的场景（论文中测试至 128K）
- Megatron-Core 中通过 context_parallel_size 参数配置
- 与 TP/EP 组合使用时需注意通信域重叠：理想情况下 CP×EP ≤ node_size（保持在 NVLink 域内）

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

## MoE Token Dispatcher（MoE Token 分发器）

术语是什么？
MoE Token Dispatcher 是负责在 EP/ETP 并行组之间路由 token 的运行时组件。在 MoE 层中，Router 为每个 token 分配 expert 后，Dispatcher 负责：(1) 将 token 从当前 rank 发送到持有对应 expert 的 rank，(2) 处理 ETP 组内的 activation 同步，(3) 计算完成后将输出 token 送回原 rank。MoE Parallel Folding 中的 Dispatcher 统一处理 ETP 和 EP 的任意组合，支持 token-dropping 和 token-dropless 两种训练范式。

从 kernel 调度角度拆解术语：
MoE Token Dispatcher 的前向计算流程（以 ETP=2, EP=2, 4 GPU 为例）：

```
Forward Pass 伪代码:
1. router_probs, router_indices = Router(local_input)  # 本地计算 gating
2. if token_dropping:
3.     # Sub-sequence dropping: 仅基于本地 logits 决策（零额外通信）
4.     capacity = CF * total_tokens / num_experts
5.     expert_counts = count_tokens_per_expert(router_indices)
6.     exceeded = expert_counts > capacity  # 标记超容量 expert
7.     # 丢弃超出容量的 token
8. 
9. permuted_tokens, permuted_indices = Permute(local_input, router_indices)
10. # EP 组内 All-to-All: 每个 rank 发送/接收 token
11. dispatched_tokens = AlltoAllV(permuted_tokens, EP_group)
12. # ETP 组内 AllGather: 确保 ETP rank 间 activation 一致
13. gathered_tokens = AllGatherV(dispatched_tokens, ETP_group)
14. # Expert GEMM 计算
15. expert_output = ExpertFFN(gathered_tokens)
16. # ETP 组内 ReduceScatter: 聚合切分的输出
17. scattered_output = ReduceScatterV(expert_output, ETP_group)
18. # EP 组内反向 All-to-All: token 返回原 rank
19. returned_tokens = AlltoAllV(scattered_output, EP_group)
20. output = Unpermute(returned_tokens, permuted_indices)
```

关键设计：
- **Sub-sequence dropping**：仅基于本地 sub-sequence 的 logits 做 token dropping，无需跨 rank AllGather 收集全局 logits（论文验证不影响模型收敛）
- **统一接口**：无论 Attention 层使用 TP/CP/DP 何种组合，Dispatcher 的输入始终是 token batch（通过 reshape 统一）
- **动态 tensor shape**：支持 EP 和 ETP 任意组合下的可变 token 数量

术语一般如何实现？如何使用？
- 在 Megatron-Core 中作为 MoE layer 的内部组件实现
- 使用 NCCL All-to-All-V（可变长度 all-to-all）、AllGather-V、ReduceScatter-V
- 配置方式：通过 capacity_factor 控制 token dropping（CF=1 用于 benchmark，dropless 模式用于训练）
- Backward pass 中 AG/RS 操作与 Forward 互换

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 采用完全不同的 token dispatch 范式——**基于索引的轻量级路由**，不依赖 All-to-All 通信，也不需要 per-expert materialized token buffer。核心思路：Gate → TopK 选择后，构建四组轻量级 int32 索引数据结构替代 materialized buffer：(1) expert_token_indices[L×K]——按 expert 拼接的 token ID 列表；(2) expert_token_offsets[E+1]——每个 expert 的 token 起止位置；(3) token_expert_indices[L×K]——按 token ID 排列的 expert ID；(4) token_index_map[L×K]——每个 token 在 expert_token_indices 中的位置。Forward 中通过 on-the-fly gather（expert_token_indices 索引原输入张量）和 on-the-fly reduction（token_index_map 索引中间结果）完成专家计算。反向传播通过相同的逆向索引 scatter 操作将梯度直接映射，无需传统的 (L,d)→(L×k,d) 中间展开步骤。总内存开销仅 4×L×K×4 bytes（int32 索引）vs 传统方法的 L×K×d×2 bytes（bf16 materialized buffer），在 DeepSeek 规模下从约 94GB 降至约 16MB。此方法适用于单 GPU 训练场景（论文实验聚焦 H100 单卡），扩展至多 GPU 需处理跨设备索引映射。

## Intra-operator Communication-Computation Overlap via Tile-Level Fused Kernels（瓦片级通信-计算融合内核的算子内重叠）

术语是什么？
Intra-operator Communication-Computation Overlap 是 MegaScale-MoE 提出的一种细粒度 kernel 级优化技术，将有直接数据依赖关系的通信算子与计算算子以 tile（瓦片）粒度融合到单个 GPU kernel 中执行。与 inter-operator overlap（在不同 CUDA stream 上异步执行独立算子）不同，intra-operator overlap 针对的是通信和计算之间存在直接依赖关系的场景（如 token dispatch 必须在 expert 计算之前完成）。核心思路是将通信和计算的工作负载切分为 tile，使用 device memory barrier（而非 host CPU）实现 tile 级别的细粒度同步——当 remote data tile 到达本地内存时，signal 通知 GEMM kernel 继续计算该 tile。这消除了 host CPU 干预引起的非确定性延迟，也避免了多 stream pipelining 的复杂 stream 控制和尾端计算浪费。

从 kernel 调度角度拆解术语：
MegaScale-MoE 实现了四类 fused kernel（论文 §4.2, Figure 10）：

**类型 1: A2A+GEMM（SP Attention Output Projection）**
```
// All-to-All 通信与 GEMM 计算以 tile 粒度融合
GEMM_Kernel():
  // Step 1: 启动本地数据 tile 的计算 + 远程数据 tile 的通信
  launch_local_GEMM_tiles_on_all_SMs()
  launch_A2A_communication_on_copy_engines()  // 使用 GPU copy engine, 不占用 SM

  // Step 2: 等待远程 tile 到达
  for each remote_tile:
    device_memory_barrier_poll(remote_tile_ready_flag)
    // Flag 由通信完成信号设置
    compute_GEMM_on_tile(remote_tile)

// SM 分配: 少量 SM 处理 A2A 通信管理（数量 tuned 使 comm≈comp latency），
// 其余 SM 全部用于 GEMM 计算
```

**类型 2: GEMM+A2A（SP Attention QKV Projection）**
```
// GEMM 计算完成后，每个 tile 立即发起 remote data transfer
Fused_GEMM_A2A_Kernel():
  for each tile:
    output_tile = GEMM_compute(input_tile, weight)
    // 直接发起 all-to-all remote write（嵌入 kernel 内部）
    A2A_send_async(output_tile, target_rank)
    // 无需额外 kernel launch
```

**类型 3: AG+Scatter+GroupedGEMM（FFN Token Dispatch）**
```
Fused_AG_Scatter_GroupedGEMM_Kernel(input_tokens, routing_map):
  // Step 1: Token 排序以最小化每 tile 的依赖 rank 数
  sorted_tokens = sort_by_expert_then_source_rank(input_tokens, routing_map)
  tiles = slice_into_computation_tiles(sorted_tokens)

  // Step 2: 每个 tile 检查依赖后执行
  for each tile:
    // 等待该 tile 所需的所有 source rank 数据到达
    for rank in tile.dependent_ranks:
      device_memory_barrier_poll(rank_data_ready[rank])
    // Local scatter 内联为按 index mapping 选择输入行（无额外 kernel launch）
    expert_input = select_rows(input_buffer, tile.row_indices)
    // GroupedGEMM 计算
    tile_output = GroupedGEMM(expert_input, expert_weights[tile.expert_id])

// 关键优化:
// - 排序使每个 computation tile 依赖尽可能少的 source rank（理想情况仅 1 个）
// - 减少等待时间，避免重复加载 expert 参数
```

**类型 4: GroupedGEMM+Gather+RS（FFN Token Combine）**
```
// 类型 3 的逆过程：GroupedGEMM 输出 → Gather → Reduce-Scatter
Fused_GroupedGEMM_Gather_RS_Kernel():
  for each tile:
    tile_output = GroupedGEMM(tile_input, expert_weights)
    gathered = local_gather(tile_output, routing_map)
    RS_send_async(gathered, target_rank)
```

**SM 分配与 Swizzling 策略**：
- 对于 A2A+GEMM 类 kernel：分配少量 SM 专门处理通信管理（A2A 比 AG/RS 更复杂），数量通过 profiling tuned 使通信和计算时延匹配
- Swizzling（重排）：重新编排 tile 的通信和计算顺序，使各 rank 的 remote data 到达节奏与 GEMM tile 消费节奏对齐，避免多 rank 同时读写同一 GPU 导致的 NVLink 带宽争用
- 所有同步通过 device memory barrier 实现（类似 FLUX [5]、Comet [53]、TileLink [56] 的方法），无需 host CPU 介入

术语一般如何实现？
- 基于 CUDA 编程模型：使用 device memory（global memory）中的 flag 变量作为 barrier，GPU copy engine 处理数据传输（不占用 SM）
- 依赖项：NCCL 通信库 + CUDA Toolkit + GPU copy engine（NVIDIA GPU 的专用 DMA 引擎）
- 与 inter-operator overlap 的互补关系：
  - Inter-operator overlap：处理无依赖的通信和计算（如 backward 中 activation recomputation 与 gradient communication 并发）
  - Intra-operator overlap：处理有依赖的通信和计算（如 forward 中的 token dispatch 与 expert computation）
- 实测效果：六种 MoE 模型下 fused vs non-fused 通信+计算总时间减少 1.2-4.7x，端到端训练 iteration time 减少 7.1-12.9%
- 论文未开源（ByteDance 内部系统），但技术原理与公开工作 FLUX、Comet、TileLink 一致

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

## GroupedGEMM in MoE Expert Computation（MoE 专家计算中的分组通用矩阵乘法）

术语是什么？
GroupedGEMM（Grouped General Matrix Multiplication）是 MoE 模型训练和推理中用于加速多个 expert FFN 层并行计算的核心 kernel。在 MoE 架构中，每个 token 被路由到 top-k 个 expert，不同 expert 接收不同数量的 token，因此需要对多个不同形状的小矩阵乘法进行批量计算。GroupedGEMM 将多个 expert 的矩阵乘法（每个 expert 执行 input_tokens × expert_weight）合并到一个 CUDA kernel 中执行，通过 cuFuncSetAttribute 精细控制每个 expert 的资源使用（shared memory、L1 cache、线程数），避免逐个 expert 串行调用的 kernel launch 开销。在 MegaScale-MoE 中，GroupedGEMM 是 SwiGLU FFN 的三次矩阵乘法（fc1、fc3 gate、fc2）的核心计算原语。

从 kernel 调度角度拆解术语：
MegaScale-MoE 中 GroupedGEMM 的完整计算流程（以 SwiGLU FFN 为例）：

```
输入: ffn_in [b*s*k/n, h]  // n 个 GPU 上的 token hidden states
      expert_weights = {fc1_weight_i, fc3_weight_i, fc2_weight_i for i in 1..E}

// Step 1: Token 路由信息预处理
token_to_expert = router(ffn_in)  // 每个 token 的 top-k expert index
expert_token_counts = count_tokens_per_expert(token_to_expert)

// Step 2: Scatter - 按 expert 分组 token
for expert_i in 1..E:
    expert_input_i = gather_tokens_for_expert(ffn_in, token_to_expert, expert_i)
    // expert_input_i shape: [num_tokens_i, h]

// Step 3: SwiGLU 三次 GroupedGEMM
// FC1: input → gate hidden
fc1_outputs = GroupedGEMM({
    expert_1: (expert_input_1, fc1_weight_1),  // [n1, h] × [h, fh] → [n1, fh]
    expert_2: (expert_input_2, fc1_weight_2),  // [n2, h] × [h, fh] → [n2, fh]
    ...
    expert_E: (expert_input_E, fc1_weight_E),  // [nE, h] × [h, fh] → [nE, fh]
})

// FC3 (gate): input → gate values
fc3_outputs = GroupedGEMM({
    expert_i: (expert_input_i, fc3_weight_i) for i in 1..E
})  // 每个 expert: [ni, h] × [h, fh] → [ni, fh]

// SwiGLU activation
fc2_inputs = {SiLU(fc1_outputs[i]) * fc3_outputs[i] for i in 1..E}

// FC2 (down projection): gate output → hidden
fc2_outputs = GroupedGEMM({
    expert_i: (fc2_inputs_i, fc2_weight_i) for i in 1..E
})  // 每个 expert: [ni, fh] × [fh, h] → [ni, h]
```

**GroupedGEMM 的 GPU 执行模型**：
- 单个 CUDA kernel 内通过动态形状处理多个不同 [m_i, k] × [k, n] 的矩阵乘法
- 使用 cuFuncSetAttribute 精细控制每个 expert 的 shared memory、L1 cache 和线程配置
- 输入/输出为动态形状 tensor（因 token 路由不均衡），频繁的动态内存分配可能引起 GPU 内存碎片化

术语一般如何实现？
- PyTorch 生态中的实现方式：
  - Megatron-LM 使用 Python for-loop 逐个 expert 调用 GEMM（简单但 kernel launch 开销大）
  - MegaScale-MoE 使用自定义 CUDA GroupedGEMM kernel，一次 kernel launch 处理所有 experts
  - PyTorch 官方 Triton Persistent Grouped GEMM（2025.08）：persistent kernel + TMA + grouped launch ordering，2.62x vs naive for-loop
  - SonicMoE（2025.12, Dao-AI Lab）：CuTe-DSL 实现，IO-computation overlap + token rounding，1.86x vs ScatterMoE
  - grouped_gemm 库（fanshiqing/grouped_gemm）：支持任意 expert 数量的批量 GEMM
- MegaScale-MoE 的观察：GroupedGEMM 中 expert intermediate dimension 远小于 dense FFN，导致 GPU 利用率低于 dense GEMM；GroupedGEMM 的细粒度资源控制可能引入同步延迟（straggler 来源之一）
- 从 MegaScale-MoE 的计算-通信比公式（R ≈ 3/2 × h_ffn × bandwidth/peak），expert intermediate dimension h_ffn 是决定 training efficiency 的关键参数——h_ffn 越大，GroupedGEMM 的 compute time 相对于 EP communication time 越充裕，越容易实现通信完全隐藏

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

## Fused MoE（融合 MoE Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused MoE 是一种针对 Mixture of Experts（MoE）推理优化的 GPU kernel 融合技术，核心思想是将 expert 选择（routing）、token 分发（dispatch）和 expert FFN 计算（GEMM + activation + GEMM）融合为单个 GPU kernel，消除中间结果的 HBM 往返和 kernel launch 开销。在未融合的 naive MoE 实现中，推理流程包含多个串行 kernel launch：(1) router 计算 top-k expert 分配 → (2) token-to-expert dispatch（scatter/gather 操作）→ (3) 逐 expert GEMM（w1 @ x → silu → w2 @ x）→ (4) weighted sum reduction。每个步骤之间的中间 tensor 需要写入 HBM 再读出，产生显著的显存带宽开销。Fused MoE 将上述步骤合并为单次 kernel launch：Triton/CUDA kernel 直接使用 sorted_token_ids（按 expert 索引排序后的 token 索引表）间接寻址 token，每个 thread block 根据 expert_ids 加载对应 expert 权重矩阵并直接计算，top-k routing weight 在 kernel 内直接乘回输出。vLLM 中 Fused MoE 的核心实现位于 `vllm/model_executor/layers/fused_moe/fused_moe.py`，使用 Triton JIT kernel 实现，支持 FP16/BF16/FP8 精度，并可与 GPTQ/AWQ 等量化方法结合。H100 GPU 上 Fused MoE 可提供 15-20% 吞吐量提升（MoE-Inference-Bench Section 7.2），大 batch 时优势更明显。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused MoE kernel 的简化伪代码（以 vLLM Triton fused_moe_kernel 为蓝本，Mixtral-8x7B 配置：8 experts, top-k=2, hidden_dim=4096, ffn_dim=14336）：

```
// Fused MoE Kernel (Triton pseudocode)
// Grid: num_experts × ceil(num_tokens / BLOCK_M) blocks

@triton.jit
def fused_moe_kernel(
    A,                  // input tokens [total_tokens, hidden_dim]
    B,                  // stacked expert weights [E, hidden_dim, ffn_dim*2 + ffn_dim]
    C,                  // output [total_tokens, hidden_dim]
    sorted_token_ids,   // token indices sorted by expert assignment
    expert_ids,         // expert index per block
    topk_weights,       // routing weights (FP32)
    num_tokens_post_padded,
):
    pid = tl.program_id(0)
    expert_id = tl.load(expert_ids + pid)
    block_start = pid * BLOCK_M

    // Step 1: Indirect token gather — only load tokens for this expert
    token_indices = sorted_token_ids[block_start : block_start+BLOCK_M]
    a_block = tl.load(A + token_indices[:, None] * H + range(H))  // [BLOCK_M, H]

    // Step 2: Load expert weights once (w1_gate, w1_up, w2_down packed)
    w1 = tl.load(B + expert_id * stride_E)  // [H, ffn_dim * 2]

    // Step 3: FC1 — fused gate+up projection (single GEMM, no HBM write)
    gate = silu(tl.dot(a_block, w1_gate))   // [BLOCK_M, ffn_dim]
    up   = tl.dot(a_block, w1_up)            // [BLOCK_M, ffn_dim]
    hidden = gate * up                       // element-wise fused gating

    // Step 4: FC2 — down projection
    w2 = tl.load(B + expert_id * stride_E + offset_w2)  // [ffn_dim, H]
    expert_out = tl.dot(hidden, w2)          // [BLOCK_M, H]

    // Step 5: Fused routing weight application
    routing_w = tl.load(topk_weights + block_start + range(BLOCK_M))
    expert_out = expert_out * routing_w[:, None]

    // Step 6: Atomic scatter-add to output (combines top-k experts)
    tl.atomic_add(C + token_indices[:, None] * H + range(H), expert_out)
```

MoE-Inference-Bench 的关键性能发现（Section 7.2，以 Mixtral-8x7B on 4×H100 + vLLM）：Fused MoE 在大 batch 时提供 15-20% 吞吐量提升，不同 sequence length 下保持 12-18% 优势。收益机制：(a) 消除中间 tensor 的 HBM 往返（每层节省约 3-4 次 HBM read/write）；(b) 减少 kernel launch 开销（从约 6-8 次 kernel launch 降至 1-2 次）；(c) 利用 sorted token 的连续性提升 L2 cache 命中率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Fused MoE 的跨框架实现：
- **vLLM**：`vllm/model_executor/layers/fused_moe/fused_moe.py`，Triton JIT kernel。通过配置文件或环境变量启用。MoE-Inference-Bench 在 vLLM 中直接使用其内置 Fused MoE kernel。
- **Megatron-Core (NVIDIA)**：`core.inference.moe.vllm_fused_moe`，CUDA graph 兼容版本，token alignment 和 indirection table 构建均在 GPU 端完成。
- **DeepGemm**：高度优化的 FP8 MoE kernel，专门针对 Hopper (SM90+)。
- **FlashInfer**：CUTLASS-based MoE for SM90/SM100。
- 前置步骤 **moe_align_block_size**：将 topk_ids 展平、按 expert 排序、padding 到 BLOCK_SIZE 对齐，是 Fused MoE 的必要预处理。
- 限制：(a) 小 batch 下 padding overhead 可能抵消融合收益；(b) 需要 Triton 或 CUDA 支持；(c) 对 expert 数量多但每 expert token 极少的场景（如大量 expert 中仅少数激活），indirect addressing 的 cache 局部性退化。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

**MoE-SpeQ 的 fuseMoE Kernel 扩展**：针对量化 MoE draft 阶段的细粒度场景（如 Qwen2-MoE: K=1408, N=2048），标准 Marlin 后端慢于 PyTorch FP16（图 11），因每个 expert 矩阵太小无法占满 GPU SM。fuseMoE 将 per-layer 所有 expert 的 gate_proj + up_proj + SiLU + down_proj 融合为单次 CUDA kernel launch，batch 多 expert 的 token hidden states 增大有效矩阵维度，提升 GPU occupancy。消融显示 fused kernel 贡献 31.8% 速度提升（从 8.88 tok/s 到 13.02 tok/s）。与 vLLM Fused MoE 不同：fuseMoE 专为量化（INT4, Marlin）和 draft 阶段设计，且与 async prefetching 独立叠加以达到 additive speedup。

**MoEBlaze 的 Epilogue Fusion for Training**：MoEBlaze 将 fused kernel 概念从推理扩展到训练，针对 SwiGLU MoE 训练设计 "epilogue fusion" kernel——将两个第一层投影 (W1, W2) 的 GEMM 与 SwiGLU epilogue（SiLU + element-wise multiply）融合为单 kernel。关键差异于推理融合：(1) 融合两个独立 GEMM（W1 和 W2 投影），输入 x 仅加载一次（vs 分别两次），两个 GEMM 流式并行执行；(2) SiLU(a) 计算在 register/shared memory 中完成，不写回 HBM——forward 仅保存 a, b, y_swi；(3) backward 中 recompute SiLU(a)（activation checkpoint），利用 SiLU 的 element-wise 特性（memory bandwidth bound，recompute 开销 ≈ 直接从 HBM 读取的成本）；(4) backward 中两个分支的 activation derivatives (∇a, ∇b) 通过 tiled reduction 做 in-place 聚合。在 SwiGLU 下实现最高 4× 激活内存减少和 2×–6.2× 训练加速。

## CPU GQA Attention Kernel for MoE Decode Offloading (MoE Decode卸载的CPU GQA Attention Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU GQA (Grouped Query Attention) Kernel 是 MoE-Lightning 在 CPU 端实现的 decode 阶段 attention 计算 kernel，基于 Intel MKL 库。其核心设计动机来自 HRM 分析：在 GPU 内存受限场景下，decode attention 的 operational intensity 极低（< 1 FLOP/Byte，属于 GEMV 模式），低于 HRM 的 P1 turning point 对应的 critical intensity——因此将 KV cache 从 CPU H2D 传输到 GPU 再做 attention 是不划算的（KV cache transfer 时间 > attention 计算时间）。替代方案：直接在 CPU 上执行 attention（利用 CPU DRAM 高带宽和 MKL 加速），仅将 attention 结果的 hidden states（远小于 KV cache）H2D 传输到 GPU 用于后续 O projection 和 MoE FFN。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU GQA Kernel 计算流程（伪代码）：
```
Input: Q [μ, n_q, d] in CPU pinned memory
       K_cache [s, n_kv, d] in CPU memory
       V_cache [s, n_kv, d] in CPU memory
       GQA_group_size = n_q / n_kv
Output: AttnOut [μ, n_q, d] in CPU pinned memory

for each batch in range(μ):
    for each kv_head in range(n_kv):
        // Step 1: QK dot product (MKL SGEMM)
        // Q_group [GQA_group_size, d] × K[kv_head]^T [d, s]
        scores = MKL_SGEMM(Q[batch, kv_head*gs:(kv_head+1)*gs, :],
                           K_cache[:, kv_head, :].T)  // [gs, s]
        
        // Step 2: Softmax over sequence dimension
        scores = softmax(scores / sqrt(d), dim=-1)  // vectorized
        
        // Step 3: Attention-weighted value sum (MKL SGEMM)
        // scores [gs, s] × V_cache[:, kv_head, :]
        AttnOut[batch, kv_head*gs:(kv_head+1)*gs, :] = 
            MKL_SGEMM(scores, V_cache[:, kv_head, :])  // [gs, d]
```
关键优化：(1) 利用 MKL batch GEMM 批量处理多个 attention heads；(2) GQA 共享 KV heads 减少 MKL 调用次数（n_kv < n_q）；(3) 使用 CPU pinned memory 存放 Q 和 output（与 GPU 共享地址空间，便于 H2D/D2H 直接传输）；(4) AttnOut 作为 PostAttn 的 input 通过 cudaMemcpyAsync H2D 回 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MoE-Lightning 中为 PyTorch C++ extension，调用 Intel MKL SGEMM（单精度通用矩阵乘法）。编译时链接 libmkl_rt.so。
- 性能：CPU GQA kernel 比 KV cache H2D 到 GPU 快 3-4×（Fig. 9），接近 CPU BW (~200 GB/s) / PCIe BW (~50 GB/s) ≈ 4× 的理论比值。
- 瓶颈：当微批次 μ 和 context length s 增大时，CPU attention 可能成为 bottleneck（需要更多 CPU DRAM BW 和 compute），此时需要更高 CPU scaling ratio 或考虑 GPU attention（A_g=1）。
- 与 MoE-Lens 的 AVX512 intrinsics kernel 对比：MoE-Lightning 使用 MKL（高层库），MoE-Lens 使用手工 AVX512 SIMD + loop unrolling + prefetching（底层优化）。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

## Paged Weight Transfer with Dual Buffering (双缓冲分页权重传输)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Paged Weight Transfer with Dual Buffering 是 MoE-Lightning 提出的 GPU memory 受限场景下的权重传输机制。将 MoE 每层所有 experts FFN weights 从 CPU memory 分页传输到 GPU，利用双缓冲（2 × per-layer-weight-buffer-size）重叠当前层计算与下一层权重预取。传输采用两阶段流水线：CPU DRAM → CPU pinned memory (memcpy) → GPU HBM (cudaMemcpyAsync)，连续 pages 的 Stage 1 和 Stage 2 重叠执行。GPU expert FFN kernel 通过 page table（映射 expert_id × page_id → GPU buffer offset）访问正确的 weight pages。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Paged Weight Transfer 与 GPU kernel 的交互流程：
```
// 初始化：分配两个 GPU weight buffers
buf_A = cudaMalloc(sizeof_per_layer_weights)
buf_B = cudaMalloc(sizeof_per_layer_weights)

// 主循环 (CGOPipe 中每一层)
for i = 1 to num_layers:
    for j = 1 to num_micro_batches:
        // Page j of layer i weights: pinned→GPU (cudaMemcpyAsync on stream_w)
        cudaMemcpyAsync(buf_A + page_offset[j], pinned_weights[i][j], 
                        page_size, HtoD, stream_w)
        
        // PostAttn(i, j): GPU kernel accesses weights via page table
        // page_table[expert_id][j] → GPU address in buf_A
        post_attn_kernel<<<..., stream_c>>>(
            hidden_states, page_table, buf_A)
        
        // Concurrently: page j+1 of layer i+1 weights: CPU→pinned
        memcpy(pinned_weights[i+1][j+1], cpu_weights[i+1][j+1], page_size)
        
        // Swap buffers for next layer
        swap(buf_A, buf_B)

// GPU Expert FFN kernel 内部:
__global__ void moe_ffn_kernel(hidden, page_table, weight_buffer):
    expert_id = gate_routing(token)
    page_id = micro_batch_id
    weight_ptr = page_table[expert_id][page_id]  // lookup
    // GEMM using weight_ptr...
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MoE-Lightning Memory Manager (Appendix A.1)。(1) 使用 CUDA streams 分离计算和传输；(2) cudaMemcpyAsync 用于异步 H2D；(3) CUDA events 用于 stream synchronization；(4) Page table 实现为 GPU 端的 simple lookup array。
- 优势：消除整层 weights 一次性传输导致的后续微批次 H2D 阻塞（FlexGen 的主要问题）。在 GPU memory 极受限时（如 T4 16GB running Mixtral 8x7B），paged transfer 是维持 GPU utilization 的关键。
- 参数：分页数 n_pages = num_micro_batches = N/μ，由 HRM policy optimizer 确定。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

## Computation Reordering for MoE Verification（MoE验证阶段计算重排序）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Computation Reordering for MoE Verification 是 MoE-SpeQ Execution Engine 在 verify 阶段的内存局部性优化：在 target model 并行验证 k 个草稿 tokens 前，分析 ELB 按 expert_id 将 tokens 重新分组排序，使分配给同一 expert 的 tokens 连续计算。Expert weights 加载到 GPU cache 后由所有需要该 expert 的 tokens 连续消费，最大化 L1/L2 cache 复用率，减少 global memory traffic。

从kernel调度角度拆解术语：
```
# Naive: token-by-token, expert weights cache thrashing
for t in 1..k:
    for e in router(h[t]).topk():
        load(W_e) → compute FFN(h[t], W_e)
# expert weights 在 cache 中被频繁置换

# Reordered: expert-by-expert, maximal cache reuse
token_groups = group_by_expert(ELB, all_tokens)
for e, tokens in token_groups.items():
    load(W_e) → 一次 cache fill
    for t in tokens: compute FFN(h[t], W_e)
# L1/L2 cache hit rate 显著提升
```
Reordering 在 CPU 端完成（基于 ELB），按 expert 分组序列发射 GPU kernels。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 Fused MoE kernel 正交：fused kernel 解决 draft 阶段 kernel launch overhead；computation reordering 解决 verify 阶段 memory locality。论文消融未单独测量 reordering 收益。

涉及论文标题：
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

## Atomic-Free GPU Parallel Dispatch Data Structure Construction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Atomic-Free GPU Parallel Dispatch Data Structure Construction 是 MoEBlaze 提出的用于替代传统 sort-based MoE token dispatch 的 GPU kernel 设计方法。传统方法（如 MegaBlocks）通过 multi-pass radix sort 将 (expert_id, token_id) 对按键排序来分组 token，但 sort 需要多次 global memory pass（pass 数与 key width 成正比）和复杂的 multi-kernel pipeline（radix sort → segmented scan → index recovery），导致 kernel launch latency 高且 GPU 资源利用率低。MoEBlaze 用 3 步 atomic-free 并行构建流程替代 sort：每一步都设计为无原子操作的 GPU 并行执行，利用 shared memory prefix sum 和 warp-level reduction 避免全局内存冲突。

从kernel调度角度拆解术语：
```
// Step 1: Build Dense Token-Expert Map
// Grid: L×E threads or warp-mapped CTA grid
// 每个 warp 分配不相交的 token rows, 写入 dense_token_map
Kernel BuildDenseMap(topk_experts[L, K], dense_token_map[L, E]):
    token_id = blockIdx.x * blockDim.x + threadIdx.x
    if token_id >= L: return
    for slot in 0..K-1:
        expert_id = topk_experts[token_id, slot]
        dense_token_map[token_id, expert_id] = token_id  // 无 collision
    // 每个 (token, expert) pair 最多 write 一次

// Step 2: Compute Expert Lengths via Warp-Level Reduction
// Grid: E CTAs, 每个 CTA 处理一个 expert 列
Kernel ComputeExpertLengths(dense_token_map[L, E], expert_lengths[E]):
    expert_id = blockIdx.x
    local_count = 0
    for row in thread_range(0, L, blockDim.x):
        if dense_token_map[row, expert_id] != EMPTY:
            local_count += 1
    // Warp-level reduction within CTA
    expert_lengths[expert_id] = warpReduceSum(local_count)

// Step 3: Route Indices to Gates (Atomic-Free via Location Map)
// 两阶段: (a) tile-level scan in shared memory
//         (b) global offset addition
Kernel RouteIndices(dense_token_map[L, E], expert_offsets[E+1],
                    expert_token_indices[L*K]):
    expert_id = blockIdx.x
    tid = threadIdx.x
    // 3a. Tile-level exclusive scan in shared memory
    shared tile_counts[BLOCK_SIZE]
    // Count non-empty entries per tile
    count = (dense_token_map[row, expert_id] != EMPTY) ? 1 : 0
    tile_counts[tid] = count; __syncthreads()
    // Exclusive scan within tile (prefix sum)
    pos = exclusiveScan(tile_counts, tid)
    // 3b. Write to final position = expert_offset + tile_offset + pos
    if count > 0:
        expert_token_indices[expert_offsets[expert_id] + pos] =
            dense_token_map[row, expert_id]
```
关键设计：三步均无原子操作。Step 1 利用 expert ID per token 唯一性保证无 intra-warp collision。Step 2 的 warp-level reduction 在 shared memory 内完成。Step 3 的 tile-level scan 在 shared memory 内做 prefix sum，然后加全局 expert_offsets 得到 final position ID——location map 构建是确定性的，写入位置无冲突。

术语一般如何实现？如何使用？
- 输入：topk_experts [L×K]（int32 expert IDs，由 Gating Network 产生）
- 输出：expert_token_indices [L×K], expert_token_offsets [E+1], token_expert_indices [L×K], token_index_map [L×K]
- 与 sort-based dispatch 对比：sort ≈ 4+ global memory passes（radix sort）× O(LK)；MoEBlaze ≈ 3 passes（dense map fill + expert count + location write），每次 pass 的 global memory traffic 更少
- 在 H100 上衡量：dispatch kernel 成为训练的关键路径优化——与传统 sort dispatch 相比，kernel launch chain 从 6-8 降至 3，GPU utilization 更高
- 论文未明确说明开源链接

涉及论文标题：
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

## Distributed Muon (分布式Muon优化器 / ZeRO-1 Style Distributed Matrix Orthogonalization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distributed Muon 是 Liu et al. (2025) 提出的基于 ZeRO-1 范式的分布式 Muon 优化器实现。核心挑战：标准 ZeRO-1 对 AdamW 高效是因为 AdamW 计算更新是逐元素的（element-wise），各 DP rank 可独立在本地分片上计算。但 Muon 需要全梯度矩阵才能执行 Newton-Schulz 正交化——若直接按 ZeRO-1 分片，每个 rank 只有 1/DP 的梯度矩阵，无法完成正交化。解决方案：在 DP 组内引入 bf16 DP Gather 操作，将分片梯度恢复为全矩阵执行 Newton-Schulz，计算完成后仅保留本地分片的更新结果。额外通信开销为 Distributed AdamW 的 1~1.25 倍，在实践的多 DP 组场景下接近下限（约 1.0x）。内存方面，Muon 仅需 1 个动量 buffer（vs AdamW 的 2 个），内存消耗减半。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Distributed Muon 的通信-计算流水线（Algorithm 1）：

```
Algorithm: Distributed Muon (per optimizer step, per DP rank)

Input: Full gradients G, DP-partitioned momentum m, DP-partitioned params p, momentum μ

# Phase 1: Gradient synchronization (标准 ZeRO-1)
1. g = reduce_scatter(G, dp_group)          # fp32, 通信量 = 4×|G| (每个 rank 得 1/DP)

# Phase 2: Local momentum update
2. g' = update_with_momentum(g, m, μ)        # 本地计算: g' = μ*m + g

# Phase 3: DP Gather — Muon 特有的额外操作
3. G_full = gather(g', dp_group)             # bf16 DP Gather, 通信量 = 2×|G| (bf16 vs fp32)
                                              # 注意: 仅在 DP 组内 gather，非全局 gather

# Phase 4: Newton-Schulz on full matrix
4. U = Newton-Schulz(G_full)                 # bf16, N=5 步, 本地计算
                                              # G_full ∈ R^{A×B}, U ≈ (G_full G_full^T)^{-1/2} G_full

# Phase 5: Discard non-local partitions
5. u = U[local_partition]                    # 仅保留对应本 rank 参数分片的更新

# Phase 6: Apply update locally
6. p' = apply_update(p, u)                   # p' = p - lr*(0.2*u*sqrt(max(A,B)) + λ*p)

# Phase 7: All-gather updated params
7. P = all_gather(p', dp_group)              # fp32, 通信量 = 4×|P|

# Phase 8: Return update RMS for logging
8. return sqrt(u².mean())
```

通信分析：
- Distributed AdamW 通信量：4 (fp32 reduce-scatter G) + 4 (fp32 all-gather P) = 8 单位
- Distributed Muon 通信量：4 (fp32 reduce-scatter G) + 2 (bf16 DP gather) + 4 (fp32 all-gather P) = 10 单位
- 比率：10/8 = 1.25x (上界)。若有 TP 启用，需额外 bf16 TP gather
- 多 DP 组下 DP gather 通信进一步分摊，实际接近 1.0x
- Newton-Schulz 在 bf16 下计算使 DP gather 通信量减半至 fp32 的 50%

端到端延迟：优化器延迟（含 Newton-Schulz 5 步迭代 + DP gather）通常为模型 forward-backward 时间的 1%~3%，可忽略。可通过 overlap gather 与 Newton-Schulz 计算、overlap reduce-scatter 与参数 gather 等工程优化进一步降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 框架集成：基于 Megatron-LM 的 ZeRO-1 分布式优化器框架实现，充分利用其 TP/PP/EP/DP 并行策略
- 开源状态：Moonshot AI 承诺将 Distributed Muon 以 PR 形式贡献给 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)；社区已有 CPU 友好复现 [bird-of-paradise/muon-distributed](https://huggingface.co/datasets/bird-of-paradise/muon-distributed)
- Megatron-LM 集成细节：
  - 使用 `dist_group` (DP) 和 `tp_group` (TP) process group handles
  - 通过 `param_groups (buffer_idx)` 和 communication `buckets (bucket_idx)` 组织参数
  - `dist_meta` 和 `global_buffer_size` 管理 ZeRO-1 分片的 "虚拟缓冲区"
  - DP all_gather 需要 bucketing 以摊销延迟；TP all_gather (on-node, NVLink 高速) 不需要 bucketing
- 关键工程注意事项：
  - DP gather 仅需在 DP 组内操作（非全局），因每个 rank 只在 DP 组内分片
  - Newton-Schulz 输入需保持 2D shape（从 flat buffer unpack 后恢复原始矩阵维度）
  - NCCL backend 用于 GPU 间通信；CPU 复现可用 gloo backend
  - 需处理通信与计算的 overlap：DP gather 可与后续计算流水线化
- 内存优势：Muon 仅需 1 个 fp32 momentum buffer per parameter（vs AdamW 的 m + v 两个），在大型 MoE 模型中节省可观内存

涉及论文标题：
- Muon is Scalable for LLM Training

## Grouped GEMM for MoE Inference (面向MoE推理的分组矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Grouped GEMM 是 NVIDIA cuBLAS 12.5 引入的 API（`cublasGemmGroupedBatchedEx`），允许在单次 kernel launch 中执行多个不同形状、不同转置、不同缩放因子的矩阵乘法。与传统的 batched GEMM（要求所有矩阵具有相同的尺寸和转置参数）不同，Grouped GEMM 专为 MoE 推理场景设计：每个 expert 接收不同数量的 token（不同 M 维度），但所有 expert 共享相同的权重矩阵形状（K×N）。使用 Grouped GEMM 可将所有 expert 的计算批量化为一个 kernel launch，避免逐 expert 循环带来的 kernel launch overhead 和 GPU 利用率低下。在 MoE decode 阶段（batch size 8-64, FP16），Grouped GEMM 相比 naive batched GEMM 循环可达到约 1.2× 加速。

从kernel调度角度拆解术语：
Grouped GEMM 在 MoE layer decode 中的执行流程：
```
输入: B个token, N个expert, 每expert权重W_e[K,N]
Router输出: token i → experts S_i, gate weights w_i

// Step 1: Token-to-Expert Dispatch
for each token i:
    for each expert e in S_i:
        dispatch token i → expert e (记录M_e++)

// Step 2: 构造 Grouped GEMM 参数
group_count = |{e : M_e > 0}|  // 有token的expert数
for each active expert e:
    A_desc[e] = {tokens_e, K}   // M_e × K (变长)
    B_desc[e] = W_e             // K × N (固定)
    C_desc[e] = output buffer   // M_e × N

// Step 3: 单次kernel launch执行所有expert的GEMM
cublasGemmGroupedBatchedEx(
    handle, &A_desc, &B_desc, &C_desc,
    group_count, ...)

// 等价于并行执行:
// for e in active_experts:  (并行, 单kernel)
//     C_e = tokens_e @ W_e   // [M_e, K] × [K, N]
```
Grouped GEMM 当前仅使用 warp-level MMA 指令（未使用 wgmma），但因减少了 kernel launch overhead 和提高了 GPU SM 占用率，实际性能优于逐 expert 调用 batched GEMM。在 memory-bound decode 下，T（唯一激活 expert 数）= group_count，每个 expert 的 GEMM 仍需其权重从 HBM→SRAM 加载。因此即使 Grouped GEMM 优化了计算调度，延迟仍与 T 成正比。

术语一般如何实现？如何使用？
- 在 cuBLAS 12.5+ 中通过 `cublasGemmGroupedBatchedEx` 使用，支持 FP16/BF16/FP32/FP64。
- 在 PyTorch 中可通过 `torch._C._cuda_grouped_gemm` 或自定义 CUDA 扩展调用。
- SGLang/vLLM 等 serving 框架在 MoE layer 实现中集成 Grouped GEMM 或 DeepGEMM（DeepSeek 的专用 fused MoE kernel）。
- 限制：Grouped GEMM 不改变每个 expert 的权重仍需从 HBM 加载的事实。在 memory-bound decode 下，优化的重点仍是减少 T（如 OEA），而非 Grouped GEMM 本身。

涉及论文标题：
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

## Sparse Cell Communication (Brainstorm 稀疏 Cell 通信原语)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Cell Communication 是 Brainstorm 框架实现的多 GPU 间 Cell 路由通信原语，用一组点对点（point-to-point）send/recv 操作替代传统 all-to-all collective。其动机是动态网络中 Router 的 Cell 分发不均匀——某些 (src_gpu, dst_gpu) pair 间传输的 Cell 数远小于其他 pair。传统 all-to-all 需要将所有 GPU pair 的传输量 padding 到 equal size，导致大量冗余通信。而 sparse communication 按实际 Cell 数量逐对传输，避免 padding。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 输入: routes[N_cells] — 每个 Cell 的目标 (gpu_id, branch_id)
//       cells[N_cells] — Cell 数据
// 输出: 每个 dst GPU 收到其负责的 Cells，按 branch 组织

// Step 1: 统计每个 (src, dst) pair 的 Cell 数量
for cell_id in 0..N_cells-1:
    dst_gpu = routes[cell_id].gpu_id
    send_counts[src_gpu][dst_gpu]++

// Step 2: All-to-all 交换 send_counts 得到 recv_counts
// (仅交换元数据，数据量极小)

// Step 3: 生成 point-to-point send/recv 计划
for dst_gpu in 0..num_gpus-1:
    if send_counts[src_gpu][dst_gpu] > 0:
        pack_cells_for_dst(dst_gpu)  // 将发往同一 GPU 的 Cell 打包
        schedule_nccl_send(buf, send_counts * cell_size, dst_gpu)
    if recv_counts[dst_gpu][src_gpu] > 0:
        schedule_nccl_recv(buf, recv_counts * cell_size, dst_gpu)

// Step 4: 执行所有 point-to-point 传输（可并行）
execute_all_scheduled_transfers()

// vs. 传统 All-to-All: 
// 每个 src→dst pair 都传输 max_count 个 Cell，总传输量为 N_gpus^2 * max_count
// Sparse: 总传输量为 sum(actual_counts)，节省 sum(padding_counts)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 用 ~3,000 LOC C++/CUDA 实现。在 CUDA 层面使用 NCCL 的 point-to-point send/recv API（ncclSend/ncclRecv），而非 ncclAllToAll。Pack 操作用 custom GPU kernel 完成 Cell 的 gather-scatter 重排列。Micro-benchmark 显示：1024 Cells（512 float32 each），4 branch/GPU，2 GPU 加速 2.13×，8 GPU 加速 2.66× vs NCCL all-to-all。加速随着 branch 数和 Cell 大小增加而放大（更多 padding 被省去）。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm

## Cross-token Expert Correlation Heatmap (跨Token专家关联热力图)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-token Expert Correlation Heatmap 是记录 MoE 模型中相邻 token 之间 expert 选择条件概率的二维矩阵。其数学定义：对同一层 l，heatmap 元素 $H_l[i][j] = P(e_j^{t+1} | e_i^t)$，即在 token t 中选择了 expert i 的条件下，token t+1 中选择 expert j 的概率。矩阵维度为 n×n（n=该层 expert 数量）。Heatmap 通过离线 profiling 构建：对大量请求的每层每 token 记录 expert selection → 统计所有相邻 token pairs → 计算条件概率。论文对 4 个模型的 >24,000 requests 构建了完整的 cross-token 和 cross-layer heatmap（>150 GB JSON traces）。

从kernel调度角度拆解术语：
Heatmap 在 kernel 调度中的核心作用是作为 **Data-Driven Predictor** 的 lookup table。在 MoE kernel launch 时：

```
Predictor 算法（基于 cross-token heatmap）:
Input: 当前 token 的 expert selection E_curr = {e1, e2, ..., ek}
       cross-token heatmap H (预计算并缓存在 Global CP)
Output: 预测的下一 token 热门 experts 列表 E_pred

1. 对 E_curr 中的每个 expert e_id:
   从 H 中定位第 e_id 行 → row = H[e_id][:]
   取 row 中 top-n 个最大概率对应的 expert IDs

2. E_pred = 所有 e_id 的 top-n 结果的并集

3. 为每个 die 生成 cp_en bits:
   for each die d:
       该 die 当前计算涉及哪些 experts → E_die
       E_pred 中与 E_die 相交且不在本地的 experts → 标记为应缓存
```

论文使用此 predictor 指导 hardware-managed HBM 的本地缓存决策。Cross-token heatmap 区别于 cross-layer heatmap：token-level correlation 的 reuse distance 较长（遍历所有层后才生成下一 token），适合映射到大容量存储（DRAM）；layer-level correlation 的 reuse distance 短（相邻层连续执行），适合映射到快速小容量存储（LLC）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 Global CP DRAM 中的 50 MB full heatmap + 0.5 MB on-chip SRAM cache（一次缓存一层）。对 100 layers × 512 experts 的支持远超当前 SOTA（Kimi K2: 61 layers, 384 experts）。
- Heatmap 构建需要 <2000 GPU hours 的离线 profiling（论文使用 8×H100 DGX + 8×H200 AWS instances 收集 traces）。
- 在 kernel 执行过程中：(1) Global CP 在 kernel launch 时查 heatmap 生成 prediction → (2) 将 cp_en bits (prediction table) 配置到各 die 的 PDU → (3) 在后续 remote data access 时 PDU 自动决定是否缓存 → (4) 已缓存数据通过 ATU 地址翻译从本地读取。
- 开源 traces 和 heatmap：https://huggingface.co/datasets/core12345/MoE_expert_selection_trace

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---

## Data-Driven Expert Predictor (数据驱动的专家预测器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Data-Driven Expert Predictor 是运行在 wafer-scale GPU 的 Global Command Processor 上的轻量级预测算法，利用 cross-token heatmap（历史 expert selection 条件概率）在 MoE kernel launch 时预测下一 token 的热门 experts，生成 duplication guidance（cp_en bits）指导各 die 的 hardware-managed HBM 自动缓存远程热门 expert。Predictor 是 Insight 2（cross-hierarchy memory management）在 token-level 的具体实现。其设计理念是：与其在 kernel 执行后被动响应 cache miss，不如在 kernel launch 时利用 temporal correlation 主动预测并 prefetch/cache。

从kernel调度角度拆解术语：
Predictor 在 kernel 调度流程中的嵌入位置：

```
Wafer-scale GPU MoE kernel 执行流程（含 Predictor）:
==================================================
Phase 1: Global CP (kernel launch 时)
------------------------------------------
1. 读取当前 batch 的 expert selection E_curr
   （来自上一层 MoE kernel 的输出或 prefill traces）

2. 运行 Predictor:
   for each expert e_id in E_curr:
       row = cross_token_heatmap[e_id]  // 从 Heatmap Cache 读取
       top_n = argsort(row)[-n:]        // top-n most likely next experts
   E_pred = union of all top_n results  // 合并去重

3. 对每个 die d:
   E_die = 该 die 当前计算的 experts
   for e in E_pred ∩ E_die:
       if e not already in die d's local HBM:
           set cp_en[e] = 1  // 标记为应缓存

4. 将 cp_en bits 打包发送到各 die 的 Local CP
   Local CP 配置到 D2D controller 的 PDU Prediction Table

Phase 2: Per-die execution (kernel 执行期间)
------------------------------------------
5. SM 请求 expert 数据:
   if is_local[expert]:
       ATU 翻译远程地址到本地地址 → 从本地 HBM/LLC 读取
   else:
       D2D XY routing → 从远程 die 读取
       返回时 PDU 检查 cp_en[expert]:
           if cp_en[expert] == 1:
               写入本地 HBM + LLC
               更新 ATU entry
               设置 is_local[expert] = 1
```

Predictor 的输出不是硬性约束（即不等同于 "必须用这些 expert"），而是"如果这些 expert 被远程访问，值得本地缓存"的 soft guidance。这种方式天然容忍预测错误——预测错误仅浪费少量本地 HBM 空间，不影响计算正确性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 Global CP 上的软件算法（运行于 A76-class ARM core），不增加专用硬件。输入 heatmap 存储在 Global CP DRAM（50 MB full）中，运行时仅需要 0.5 MB on-chip cache 缓存一层 heatmap。
- 每 token prediction 的额外开销极低（仅查表 + 取 top-n + 组合 cp_en），完全隐藏在 kernel launch 的常规 overhead 中。
- 在 Dojo 5×5 配置上的效果：Pred Only（仅 predictor, 无 task allocation）在 DeepSeek V3 上实现 3.0× throughput 提升，hop count 降低 4.5×。Allo+Pred（task allocation + predictor）进一步实现 7.0× throughput 和 >213× hop count 降低。
- 局限：Predictor 效果取决于 cross-token correlation 的强度——Llama 4 受益最多（top 20% candidates cover 80% mass），DeepSeek V3 受益最少（47%）。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---

## Multi-Die Task Allocation for MoE (多Die MoE任务分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Die Task Allocation 是运行在 wafer-scale GPU Global CP 上的启发式算法，将 MoE kernel 计算按 expert 拆分为 per-die 子任务，基于 expert 在各 die 上的分布信息和 cost model 将子任务分配到最优 die。核心创新是引入 **Candidate Mechanism**（扩展候选 die 到邻居 die，而不限于 expert 所在 die）和 **Block-Granularity Distribution**（以 block size=50 为单位分配请求，在效率和精度间折中）。该算法是 Insight 3（expert-placement-aware workload distribution）的实现——在 single-GPU-like programming model 下，算法对软件完全透明，运行在 Global CP 硬件上。

从kernel调度角度拆解术语：
算法（Algorithm 1 in paper）的完整流程：

```
Input:  expert_reqs_dict = {expert_id: num_requests}
        expert_die_map = {expert_id: [die_ids where expert resides]}
Output: allocation_plan = [(expert_id, target_die, num_requests)]

1. 初始化 load_per_die[d] = 0 对所有 die d

2. Sort experts by req_num ascending  // 先处理冷门 expert

3. for each (expert_id, req_num) in sorted experts:
     
     // Candidate Mechanism: 候选 die = 存有该 expert 的 die + 邻居 die
     candi_list = GenCandidateList(expert_id, dis=1)
     // 按当前负载排序候选 die
     candi_list = Sort(candi_list, key=lambda i: load_per_die[i])
     // 限制候选数 max_split_num ∝ req_num
     candi_list = candi_list[:max_split_num]
     
     // Block-Granularity Distribution
     while req_num > 0:
         req_blk = min(50, req_num)  // block size = 50
         costs = CostModel(candi_list, req_blk)
         // CostModel = f(DRAM_access, compute, D2D_comm)
         target_die = Argmin(costs)
         allocation_plan.append((expert_id, target_die, req_blk))
         load_per_die[target_die] += req_blk
         req_num -= req_blk

4. allocation_plan = MergeTasks(allocation_plan)
   // 合并在同一 die 上的相同 expert 的任务

5. return allocation_plan

Function GenCandidateList(expert_id, dis):
    local_dies = expert_die_map[expert_id]
    remote_dies = FindNearDies(local_dies, dis)  // Manhattan distance ≤ dis
    return local_dies + remote_dies
```

**Cost Model** 考虑三个维度：
- $C_{\text{DRAM}}$: 读取 expert 权重的 HBM access time（local=300ns, remote=300ns + hops×200ns）
- $C_{\text{compute}}$: 基于 die 的 FP16 TFLOPS 和请求数估算的 GEMM 执行时间
- $C_{\text{D2D}}$: 从远程 die 读取 expert 权重跨 D2D links 的通信时间（bandwidth contention 通过 central resource manager 建模）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 运行时特征：(1) 每 MoE layer 的 kernel launch 时执行一次（非 per-token）；(2) 输入 expert_reqs_dict 来自上一层的 Gate 网络输出（已确定每个 token 的 expert 选择）；(3) expert_die_map 由 Global CP 的 Expert Distribution Table 动态维护（expert migration/replication 后更新）。
- 两个关键 heuristic 的设计动机：(1) Candidate Mechanism——允许将请求分配到邻居 die（而非仅本地 die），在 workload balance 和 D2D traffic 之间 trade-off（传统 EP 将所有请求分配到本地 die 避免 D2D 但负载严重不均）；(2) Block-Granularity——split_num 和 block_size 的 trade-off（split 越多越平衡但 overhead 越大）。
- 效果：Allo Only (仅 task allocation) 降低 hop count 142×（vs Base），实现 6.3× throughput。大部分 performance gain 来自 allocation 使得绝大多数请求分配到本地 die。Host CPU 实现 overhead：Dojo 上 5.2-14.2%，Dojo-Enhanced 上 19.3-51.6%。
- 开源：https://github.com/zhongkaiyu/waferscale_gpu_moe_sim（Python 实现为 simulator 的一部分；论文讨论若 future programming model 变为 multi-GPU-like，此算法可在 host CPU 软件层实现而无需硬件修改）。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting
