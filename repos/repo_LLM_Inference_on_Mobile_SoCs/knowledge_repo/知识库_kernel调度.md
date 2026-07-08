## GPU-NPU Tensor-level 异构并行执行

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU-NPU Tensor-level 异构并行执行是 HeteroInfer 提出的核心机制，将单个 Matmul 算子的计算负载在 GPU 和 NPU 之间按张量维度拆分并并行执行。与传统的 layer-level 分配（整层分配给 GPU 或 NPU）不同，tensor-level 并行在算子内部实现 GPU 和 NPU 的并发计算，通过三种互补的分区策略解决不同场景下的 NPU 性能退化问题。

三种分区策略：
1. **Weight-centric Partition（权重中心分区）**：沿权重张量的行维度拆分，GPU 和 NPU 各计算一部分权重行的矩阵乘法，最终拼接输出。由 offline solver 确定最优 partition ratio。prefill 阶段用于解决 NPU 的 shape-sensitive 性能退化，decoding 阶段用于最大化 SoC 内存带宽。
2. **Activation-centric Partition（激活中心分区）**：沿激活张量的序列长度维度拆分，将激活拆分为多个标准形状子张量（NPU 预生成图执行）+ 一个动态形状子张量（GPU 执行）。解决 NPU 静态图不支持动态序列长度的问题。
3. **Hybrid Partition（混合分区）**：结合 activation-centric（处理动态形状）和 weight-centric（优化 NPU 子任务的形状适配）。当只用 activation-centric 导致 GPU 负载过小或 NPU 形状不佳时采用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Tensor-level GPU-NPU Parallel 的 Kernel 调度伪代码:

function dispatch_matmul_operator(W[M,K], A[K,N], solver_output):
    strategy, ratio = solver_output.lookup(shape(W), shape(A))
    
    if strategy == NO_PARTITION:
        if GPU_latency < NPU_latency:
            return gpu_kernel(W, A)       // GPU-only
        else:
            return npu_kernel(W, A)       // NPU-only
    
    elif strategy == WEIGHT_CENTRIC:
        // 沿权重行维度 M 拆分
        split_point = M * ratio.GPU       // e.g., 28672 * 0.75 = 21504
        W_gpu = W[0:split_point, :]       // [21504, 4096]
        W_npu = W[split_point:M, :]       // [7168, 4096]
        
        // GPU 和 NPU 并行执行
        parallel:
            O_gpu = gpu_kernel_matmul(W_gpu, A)   // [21504, N]
            O_npu = npu_kernel_matmul(W_npu, A)   // [7168, N]
        
        // Fast sync + merge
        fast_sync(wait_for=GPU, mode=NPU_DOMINANT)  // prefill
        O = concat(O_gpu, O_npu, axis=0)            // [M, N]
        return O
    
    elif strategy == ACTIVATION_CENTRIC:
        // 沿激活序列长度维度 N 拆分
        // 例如: seq_len=300 → 256 (standard) + 44 (dynamic)
        A_std = A[:, 0:256]                 // 标准形状
        A_dyn = A[:, 256:300]               // 动态形状
        
        parallel:
            O_npu = npu_kernel_prebuilt(W, A_std)   // 预编译图
            O_gpu = gpu_kernel_matmul(W, A_dyn)     // GPU 处理动态部分
        
        fast_sync(wait_for=GPU, mode=NPU_DOMINANT)
        O = concat(O_npu, O_gpu, axis=1)            // [M, N]
        return O
    
    elif strategy == HYBRID:
        // Step 1: activation-centric 处理动态形状
        A_std = A[:, 0:256]
        A_dyn = A[:, 256:300]
        
        // Step 2: weight-centric 优化 NPU 子任务
        split_point = M * ratio.GPU       // e.g., 4096 * 0.4 = 1638
        W_gpu = W[0:split_point, :]       // [1638, 14336]
        W_npu = W[split_point:M, :]       // [2458, 14336]
        
        parallel:
            O_gpu_sub = gpu_kernel_matmul(W_gpu, A_dyn)  // [1638, 44]
            O_npu_sub = npu_kernel_matmul(W_npu, A_std)  // [2458, 256]
        
        fast_sync(wait_for=GPU, mode=NPU_DOMINANT)
        // 拼接: 行方向(weight split) + 列方向(activation split)
        O = concat_complex(O_gpu_sub, O_npu_sub)
        return O

// Solver 的决策逻辑 (offline)
function solver_decision(W_shape, A_shape, profiler_data):
    candidates = []
    
    // GPU-only
    candidates.append(("gpu_only", profiler_data.gpu_latency(W_shape, A_shape)))
    // NPU-only
    candidates.append(("npu_only", profiler_data.npu_latency(W_shape, A_shape)))
    // NPU-only with padding (if A_shape not standard)
    if A_shape.seq_len not in STANDARD_LENGTHS:
        padded_len = next_standard(A_shape.seq_len)
        candidates.append(("npu_padded", profiler_data.npu_latency(W_shape, padded_len)))
    
    // Enumerate weight-centric ratios
    for ratio in [0.1, 0.2, ..., 0.9]:
        T_gpu = profiler_data.gpu_latency(W_shape*ratio, A_shape)
        T_npu = profiler_data.npu_latency(W_shape*(1-ratio), A_shape)
        T_total = max(T_gpu, T_npu) + T_sync
        candidates.append((f"weight_centric_{ratio}", T_total))
    
    // Enumerate activation-centric splits
    for std_len in STANDARD_LENGTHS:
        if std_len < A_shape.seq_len:
            dyn_len = A_shape.seq_len - std_len
            T_gpu = profiler_data.gpu_latency(W_shape, dyn_len)
            T_npu = profiler_data.npu_latency(W_shape, std_len)
            T_total = max(T_gpu, T_npu) + T_sync
            candidates.append((f"activation_centric_{std_len}", T_total))
    
    return argmin(candidates, key=latency)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**HeteroInfer 中的实现细节**：

- **Solver 输入**：Offline profiler 对 LLM 中所有 Matmul 算子的权重形状（[4096,4096], [28672,4096], [4096,14336] 等）在 GPU (OpenCL) 和 NPU (QNN) 上测量所有标准激活形状（seq_len=64/128/256/512/1024）的执行延迟。搜索空间受限于：仅 LLM 权重形状、NPU stage performance 下界 32、预定义标准序列长度。
- **Solver 输出示例**（Table 3）：`[4096,4096] × [4096,1]` → Weight-centric, GPU:NPU=1:1；`[28672,4096] × [4096,1]` → Weight-centric, GPU:NPU=3:1；`[4096,14336] × [14336,257-384]` → Hybrid, 2:3 (Weight)。
- **Result Merge**：GPU 和 NPU 输出直接写入共享 LPDDR 中的不同区域，merge 操作仅涉及指针/偏移计算，无数据拷贝。
- **Memory Pool**：专用 memory pool 管理 GPU/NPU 的输入输出 buffer（几个 slot 跨层复用），buffer 不会被 driver 回收，确保地址映射在推理全程有效。

涉及论文标题：
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

## Fast GPU-NPU Synchronization（快速异构同步）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fast GPU-NPU Synchronization 是 HeteroInfer 提出的微秒级 GPU-NPU 同步机制，替代传统的 clFinish (~400 μs) 同步命令。其核心原理是利用三个硬件/系统特性：(1) UMA 统一内存架构——GPU 和 NPU 共享 LPDDR 物理内存，CPU 可直接轮询 GPU 写入的内存地址；(2) LLM 推理的逐层重复性——同一 decoder block 中 GPU kernel 的执行时间在层间高度一致，可预测；(3) 移动 SoC 的小/中 CPU 核可用于低功耗轮询。

具体实现：CPU sync thread 先 `usleep(predicted_wait_time - margin)` 睡眠到接近 GPU kernel 预计完成的时间点（usleep 最小粒度约 80-100 μs），然后切换到轮询模式——读取 GPU output tensor 旁边的 flag bit（GPUkernel 完成时置位），轮询仅需数微秒。一旦 flag bit 置位，CPU 立即通知 NPU 提交下一层的 kernel。这替代了传统的 clFinish 阻塞等待（~400 μs 固定开销，在 decoding 阶段与 kernel 执行时间（数百微秒）可比甚至更大）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Fast Synchronization 在 Prefill vs Decoding 阶段的差异:

=== Prefill Phase (NPU-Dominant) ===
// NPU 执行时间长于 GPU → GPU 执行时间被 NPU 完全覆盖
// 但需延迟提交下一个 GPU kernel 直到 NPU 完成

Timeline (seq_len=256, FFN layer):
  NPU: |══════════ FFN_Up (1.9ms) ══════════|        |══ FFN_Down_NPU ══|
  GPU:          |═ FFN_Down_GPU (0.7ms) ═|    [IDLE]  |════ FFN_Up_GPU ══|
  CPU:          |submit|               |poll|         |submit|
  Sync:                                  ↑ flag=1 → notify NPU
  Time: ──────────────────────────────────────────────────────────────────→

伪代码:
  def prefill_sync_strategy():
      # NPU 先提交 (计算量大)
      qnn_submit(npu_graph_ffn_up)
      
      # GPU 后提交 (计算被 NPU 覆盖)
      gpu_submit(gpu_kernel_ffn_down)
      
      # CPU sync thread:
      predicted_npu_time = 1900  # μs, from profiler
      predicted_gpu_time = 700   # μs
      wait_time = predicted_npu_time - predicted_gpu_time - margin
      usleep(wait_time)  # ~1200 μs
      
      # 轮询 GPU output flag
      while (*gpu_output_flag == 0):
          continue  # 数微秒
      
      # GPU 完成 → 通知 NPU 继续下一层
      # NPU 大概率已完成 (NPU time > GPU time)
      qnn_submit(npu_graph_next_layer)
      
      # 同步开销: ~数微秒 ~ 数十微秒 (vs. clFinish ~400μs)


=== Decoding Phase (GPU-Dominant) ===
// GPU 执行时间长于 NPU → NPU 执行时间被 GPU 完全覆盖
// GPU queue ordering 自动保证 kernel 顺序, 无需显式同步

Timeline (seq_len=1, FFN layer, weight-centric 3:1):
  NPU: |═══ FFN_NPU (25% weight) ═══|  [IDLE]
  GPU: |══════════ FFN_GPU (75% weight) ══════════|
  CPU: |submit NPU|    |submit GPU|               |GPU queue auto-order|
  Sync:                  NPU done → enqueue next GPU kernel
                         GPU queue ensures kernel_1 before kernel_2
  Time: ───────────────────────────────────────────────────────────────→

伪代码:
  def decoding_sync_strategy():
      # NPU 先提交 (计算量小, 执行快)
      qnn_submit(npu_kernel_25pct)
      
      # GPU 提交 kernel_1
      gpu_submit(gpu_kernel_75pct)  # GPU queue pos=1
      
      # NPU 完成后挂载 callback:
      # → enqueue gpu_kernel_next_layer  # GPU queue pos=2
      # GPU hardware queue ordering 保证 kernel_1 先于 kernel_next 执行
      
      # 同步开销: 0 (GPU queue ordering 无需 CPU 介入)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**HeteroInfer 中的实现细节**：

- **Memory Pool 与 Flag Bit**：为每个 GPU/NPU kernel 的输出 tensor 维护一个专用 memory pool slot。每个 slot 旁附加一个 flag bit（UMA 中的特定地址）。GPU kernel 完成时写入 flag bit = 1。CPU sync thread 仅需轮询该 flag bit。
- **可预测等待时间的来源**：LLM 的每层 decoder block 执行完全相同的操作序列，因此 GPU kernel 在各层的执行时间高度一致。Profiler 离线测量各 kernel 的平均执行时间和方差，solver 在线使用这些值计算 `predicted_wait_time`。
- **CPU 核选择**：使用小/中 CPU 核（而非大核）进行 flag bit 轮询——轮询仅涉及读内存操作，功耗极低。大核留给 OS 和其他应用。

**效果**（来自论文实验）：
- Prefill: fast sync 对 Hetero-layer 平均 15.8% speedup，对 Hetero-tensor 平均 24.3% speedup（Llama-8B, seq_len=256 时从 196.44 提升至 236.92 tokens/s）
- Decoding: fast sync 对 Hetero-tensor 实现 4.01× speedup（Llama-8B），其他模型 2.2× speedup。Decoding 阶段收益远大于 prefill，因为 decoding kernel 执行时间（数百 μs）与 clFinish 开销（~400 μs）量级相当。

涉及论文标题：
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

## NPU Stage / Order-Sensitive / Shape-Sensitive 性能特征

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

这是 HeteroInfer 论文通过对 Snapdragon 8 Gen 3 Hexagon NPU 的深入 profiling 揭示的三种 NPU 计算性能特征，它们源于 NPU 的 systolic array 硬件架构和 weight-stall 计算范式，是实现高效 GPU-NPU 异构调度的关键输入。

**NPU-1: Stage Performance（阶梯性能）**
由于 NPU 内部 systolic array 的固定尺寸（32×32），张量维度若不整除 32 则需要内部 padding 补齐，导致在特定维度范围内（如 1-32, 33-64）执行延迟相同（阶梯状）。例如：任何 M < 32 或 K < 32 的 Matmul 操作延迟完全相同。
影响：小激活张量（decoding 的 [4096,1]）无法充分利用 systolic array 的计算资源，NPU 性能退化至与 GPU 相当甚至更低。决定了 decoding 阶段 GPU 而非 NPU 成为主要计算单元。

**NPU-2: Order-Sensitive Performance（顺序敏感性能）**
Weight-stall 范式要求 weight tensor 适配 systolic array 以最大化复用。当 weight tensor 远大于 activation tensor 时，weight 无法完全驻留在 PE 中，需频繁从内存重新加载——NPU 效率急剧下降。`[14336,4096] × [4096,K]` 比 `[K,4096] × [4096,14336]` 快最多 6×（同等计算量下）。
影响：HeteroInfer 利用计算不变量 `[M,N] × [N,K] → [[K,N] × [N,M]]^T` 交换张量顺序，将较小维度作为 weight。

**NPU-3: Shape-Sensitive Performance（形状敏感性能）**
即使 input tensor 大于 weight tensor，NPU 效率仍受输入张量的行列比影响。行尺寸大于列尺寸时（较大的 M/K 比）NPU 性能更优，因为列尺寸与 weight tensor 共享——列尺寸越大则 weight tensor 越大，越不利于 weight-stall 范式。
影响：FFN-down 层的 [4096, 14336] weight（列 > 行，经转置后）仅实现 0.5-1.5× GPU 性能，需要 weight-centric partition 将部分计算卸载到 GPU。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
NPU 三种性能特征对 Kernel 调度的具体影响:

=== NPU-1: Stage Performance ===
Kernel: Matmul [4096, K] × [K, 1]  (Decoding, K = d_model)

实测延迟 (Snapdragon 8 Gen 3):
  K=1:   T_npu ≈ 693 μs  (32×32 systolic array 严重未充分利用)
  K=32:  T_npu ≈ 693 μs  (同一阶梯)
  K=33:  T_npu ≈ 1386 μs (需要 2 个 tile, 阶梯跳变)
  K=64:  T_npu ≈ 1386 μs

GPU 对比:
  K=1:   T_gpu ≈ 511 μs  (GPU 线性性能, 无阶梯效应)
  K=32:  T_gpu ≈ 540 μs
  K=64:  T_gpu ≈ 580 μs

→ 调度决策: decoding 阶段 GPU 为 primary (更优的小 K 性能)
              weight-centric partition ratio 偏向 GPU (3:1)


=== NPU-2: Order-Sensitive Performance ===
Kernel A: [14336, 4096] × [4096, K]    // weight=14336×4096, act=4096×K
Kernel B: [K, 4096] × [4096, 14336]    // weight=K×4096, act=4096×14336
// 两者均执行 2 × 14336 × 4096 × K 次操作

实测对比:
  Kernel A: T_npu ≈ 1.2 ms  (weight 较大, 但能驻留在 systolic array)
  Kernel B: T_npu ≈ 7.2 ms  (weight 极大, 频繁从内存加载 → 6× slowdown)

HeteroInfer 的变换:
  [M, N] × [N, K] → [[K, N] × [N, M]]^T
  使 K (通常 < N) 成为 weight 维度 → 优化 NPU-2

  例: [4096, 14336] × [14336, 256]
      → [[256, 14336] × [14336, 4096]]^T
      → weight=[256, 14336] (不是 [4096, 14336])
      → 但 256 < 14336, weight 仍然很大, 收益有限
      → 需要 weight-centric partition 进一步优化


=== NPU-3: Shape-Sensitive Performance ===
Kernel: Matmul [M, N] × [N, K]  (Prefill)

  M >> N (行 >> 列): NPU 效率高 (weight 小, 复用高)
  M << N (行 << 列): NPU 效率低 (weight 大, 复用低)

  例: [4096, 14336] × [14336, 256]
     M=4096, K=14336 (行 < 列)
     → NPU 仅 0.5-1.5× GPU 性能
     → weight-centric partition: 40% GPU, 60% NPU (2:3 ratio)

  例: [28672, 4096] × [4096, 256]
     M=28672, K=4096 (行 >> 列)
     → NPU 性能显著优于 GPU
     → no partition: NPU-only
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**通用性**：这三种性能特征并非 Snapdragon 8 Gen 3 独有，而是任何使用 weight-stationary systolic array 的 NPU（Google TPU、Huawei Da Vinci、Apple ANE 的矩阵单元）都会不同程度表现出的固有特征。HeteroInfer 论文明确声称其设计围绕这些"广泛采用的硬件特征"，因此可移植到其他移动 SoC。

**HeteroInfer 中的 profiling 流程**：
- Offline profiler 对每种 (weight_shape, activation_shape) 组合在 GPU 和 NPU 上实测延迟
- 搜索空间约束：仅 LLM 权重形状（非全搜索）、子张量 ≥ 32（stage performance 下界）、激活仅标准序列长度
- 非标准序列长度使用特性插值：GPU-1 linear performance 线性插值、NPU-1 stage performance 阶梯插值
- Profiling 耗时：< 20 分钟（因搜索空间受限）

涉及论文标题：
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

## Swapping-Recompute Pipeline（交换-重计算流水线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Swapping-Recompute Pipeline 是 LLMS 论文提出的用于加速 LLM context switching 的 I/O-计算重叠技术。当 LLM context 的某些 KV cache chunk 被换出到磁盘时，恢复它们有两种方式：(1) 从磁盘读入（I/O）或 (2) 从原始 prompt text 重计算（computation）。Swapping-Recompute Pipeline 将这两种方式以流水线形式并发执行——I/O 线程加载 chunk 到内存的同时，计算线程重计算其他 chunk，利用在传统 context switching 中闲置的 CPU/GPU。

核心技术挑战是：chunk 可以被不连续地换出（chunk_3 和 chunk_7 在磁盘而 chunk_4-6 在内存），标准 LLM 的 position encoding（连续递增）和 causal mask（连续因果）无法处理不连续 chunk 的重计算。LLMS 通过修改 position encoding（使用全局位置而非相对位置）和 causal mask（每条对角线仅保留到当前 token 为止的 mask）解决此问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Chunk Recompute Procedure（处理不连续 chunk）**：

```
# ========== 输入 ==========
# 原始 prompt text: T = ["a", "b", "c", "d", "e", "f"]
# 在内存中的 chunk: {chunk_1="ab", chunk_3="d", chunk_4="f"}  
# 在磁盘中的 chunk (需恢复): {chunk_2="c", chunk_4="e"}
#   注: chunk_size=2 (示例简化, 实际为 16)
#
# 目标: 通过重计算恢复 "c" 和 "e" 的 KV cache

def recompute_missing_chunks(in_memory_kv, missing_positions, prompt_texts):
    """
    in_memory_kv: {(layer, head): [K/V tensors for positions in memory]}
    missing_positions: [2, 4]  (0-indexed: "c"=pos2, "e"=pos4)
    prompt_texts: {2: "c", 4: "e"}
    """
    
    for layer l in 0..L-1:
        # === Step 1: Embed missing tokens ===
        K_missing = []  # 待插入的 K tensors
        V_missing = []  # 待插入的 V tensors
        
        for pos in missing_positions:
            token_emb = embed(prompt_texts[pos])  # [1, d_model]
            
            # 使用全局位置编码 (非相对位置)
            pos_enc = global_position_encoding(pos)  # pos 2 → encoding for position 2
            # 不是相对位置 "第一个缺失 token=0, 第二个=1"
            
            # 计算 Q, K, V (仅对缺失 token)
            for head h in 0..H-1:
                Q_lh = (token_emb + pos_enc) @ W_Q_lh
                K_lh = (token_emb + pos_enc) @ W_K_lh
                V_lh = (token_emb + pos_enc) @ W_V_lh
                K_missing.append(K_lh)
                V_missing.append(V_lh)
        
        # === Step 2: 构建完整 K, V (内存 + 重计算) ===
        for head h in 0..H-1:
            # 合并: 按原始 token 顺序插入
            K_full = []  # 长度为 seq_len 的完整 K 序列
            V_full = []
            
            full_pos = 0  # 原始序列位置
            for i in range(seq_len):
                if i in in_memory_kv:
                    # 从内存中获取
                    K_full.append(in_memory_kv[l][h][i])
                    V_full.append(in_memory_kv[l][h][i])
                else:
                    # 从重计算结果中获取
                    idx = missing_positions.index(i)
                    K_full.append(K_missing[idx * H + h])
                    V_full.append(V_missing[idx * H + h])
            
            K_full = stack(K_full)  # [seq_len, d_head]
            V_full = stack(V_full)  # [seq_len, d_head]
        
        # === Step 3: Attention with modified causal mask ===
        for head h in 0..H-1:
            # 标准的因果 mask (下三角):
            #   pos:  a  b  c  d  e  f
            #   a:   1  0  0  0  0  0
            #   b:   1  1  0  0  0  0
            #   c:   1  1  1  0  0  0
            #   d:   1  1  1  1  0  0
            #   e:   1  1  1  1  1  0
            #   f:   1  1  1  1  1  1
            #
            # 这里的 mask 和标准无差别——因为使用全局位置编码后，
            # 重计算的 token 处于其应有的全局位置，因果 mask 自然正确。
            # 关键区别在于 position encoding 使用了全局位置而非局部连续位置。
            
            causal_mask = tril(ones(seq_len, seq_len))  # 下三角
            
            Q_lh = K_full @ W_Q_lh  # 简化表示
            attn_scores = softmax(Q_lh @ K_full^T / sqrt(d_head) + causal_mask)
            attn_out = attn_scores @ V_full
        
        # Step 4: O projection + FFN → 传递到下一层
        # 下一层的 input 由完整的 attn_out 确定
```

**Swapping-Recompute Pipeline 架构**：

```
# ========== Pipeline 线程结构 ==========

Thread 1 (I/O Thread):
    for layer l in 0..L-1:
        for chunk c in chunks_to_load_from_disk[l]:
            read_from_disk(chunk_c_K)  # 读 K cache for layer l
            read_from_disk(chunk_c_V)  # 读 V cache for layer l
        signal_completion(l)  # 通知计算线程: layer l I/O 完成

Thread 2 (Compute Thread):
    for layer l in 0..L-1:
        wait_for_completion(l)  # 等待 layer l 的 I/O 完成
        if has_chunks_to_recompute(l):
            recompute_missing_chunks(layer_l_KV, missing_l, texts_l)
        
        # 此时 layer l 的完整 K/V 就绪 (I/O loaded + recomputed)
        # 继续下一层的 pipeline

# ========== Pipeline Timeline ==========
#              time →
# I/O:        |==L0 load==|==L1 load==|==L2 load==| ...
# Compute:    |==L0 recomp==|==L1 recomp==| ...
# 
# Wall-clock time ≈ max(ΣT_IO, ΣT_recomp) 而非 T_IO + T_recomp
# 在 I/O 和 recompute 负载均衡时接近最优重叠
```

**Elastic Pipeline Planning (Equation 4)**：

```
# ========== Offline Profiling (安装时一次性) ==========
# 测量当前设备的:
#   T_re(x, f, e): 重计算 x 个 chunk 的延迟
#     x = chunk 数
#     f = CPU/GPU 频率
#     e = 能耗模式
#   T_IO(m): 从磁盘加载 m MB 的延迟
#
# 实践中用线性函数近似:
#   T_re(x) ≈ a_re * x + b_re
#   T_IO(m) ≈ a_IO * m + b_IO
# 通过若干离散测试点拟合参数 a, b

# ========== Online Planning (每次 callLLM 触发) ==========
# 给定:
#   m_onload: 需加载的总内存量 (各压缩级别 chunk 的加权和)
#   {x_{ratio_w}}: 各压缩级别下缺失的 chunk 数
#   {ratio_w}: 各压缩级别的压缩率
#
# 优化问题:
#   minimize: pipelineDelay = max(T_re(Σ x_{re}^{w}), T_IO(m - Σ ratio_w * x_{re}^{w}))
#   s.t.:     0 ≤ x_{re}^{w} ≤ x_{ratio_w}  for each w
#
# 其中 x_{re}^{w} 是决定通过重计算（而非 I/O）恢复的压缩级别 w 的 chunk 数
#
# 这是一个简单的线性规划问题（对于少量的压缩级别）:
#   目标: 让重计算时间和 I/O 时间尽可能接近（完全重叠）
#   如果 T_re 全部 < T_IO 全部, 则全部重计算
#   如果 T_IO 全部 < T_re 全部, 则全部 I/O
#   否则在两者之间找到平衡点使 max 最小化

def plan_pipeline(m_onload, x_by_ratio, ratios):
    best_x_re = {w: 0 for w in ratios}
    best_delay = float('inf')
    
    # 遍历可能的重计算比例
    for x_re_INT8 in range(x_by_ratio['INT8'] + 1):
        for x_re_INT4 in range(x_by_ratio['INT4'] + 1):
            for x_re_INT2 in range(x_by_ratio['INT2'] + 1):
                x_total = x_re_INT8 + x_re_INT4 + x_re_INT2
                io_size = (m_onload
                          - 1.0 * x_re_INT8 * chunk_size_INT8
                          - 0.5 * x_re_INT4 * chunk_size_INT8  # INT4 = half of INT8
                          - 0.25 * x_re_INT2 * chunk_size_INT8) # INT2 = quarter
                
                delay = max(T_re(x_total), T_IO(io_size))
                if delay < best_delay:
                    best_delay = delay
                    best_x_re = {'INT8': x_re_INT8, 'INT4': x_re_INT4, 'INT2': x_re_INT2}
    
    return best_x_re, best_delay
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LLMS 中的具体实现（§4 Implementation）：
- **多线程架构**: 使用 Python threading——一个独立的 I/O 线程和一个计算线程（主线程）。
- **同步机制**: 计算线程在进入下一层前等待当前层的 I/O 完成。使用 `threading.Event` 或简单的条件变量实现。
- **I/O 实现**: 使用 Pickle (Python) 和 Pickle-in-Cpp (C++) 进行内存-磁盘序列化。chunk 数据以序列化格式写入磁盘文件（每个 chunk 一个文件或使用共享文件 + offset）。
- **重计算实现**: 利用 HuggingFace Transformers 的 forward pass，逐层重计算缺失 token——调用 `model.model.layers[l].self_attn()` 的 Q/K/V 投影 + attention 部分。

Pipeline 的弹性特性：
- **自适应负载均衡**: Planning 阶段根据离线 profiled 的硬件参数（T_re, T_IO）自动决定最优的 I/O-recompute 分配，无需手动调参。
- **设备自适应**: 在不同设备上 pipeline 自动调整——例如在 TX2（SATA HDD, 低 I/O 带宽）上倾向于更多重计算、更少 I/O；在 Orin（NVMe SSD, 高 I/O 带宽）上倾向于更多 I/O、更少重计算。
- **压缩感知**: 重计算 chunk 的决策考虑压缩率——INT8 chunk 的重计算与 I/O（8MB）等价于 INT2 chunk 的 4 倍 I/O（2MB），因此更倾向于重计算轻量 chunk。

消融实验中的贡献（Jetson Orin NX, Llama2-7B, 8 active contexts, Markov pattern）：
- LLMS 全部技术: 0.27s switching latency
- 去掉 Swapping-Recompute Pipeline: 1.62s switching latency
- Pipeline 独立贡献: 1.62s → 0.42s 的延迟降低（约 3.86× 加速）

涉及论文标题：
- LLM as a System Service on Mobile Devices

## OpenCL Command Queue on Mobile GPU（移动 GPU 的 OpenCL 命令队列）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

OpenCL Command Queue 是 OpenCL 编程模型中 CPU host 向 GPU device 提交 kernel 执行任务的异步队列机制。CPU 端调用 `clEnqueueNDRangeKernel()` 将 kernel 入队（非阻塞调用），GPU 端按队列顺序（in-order queue）或乱序执行。FUSE 论文揭示的关键移动端限制：ARM Mali GPU 的命令队列硬件深度仅 2 个 outstanding entries（vs. 桌面 GPU 的数十至数百个），这意味着 CPU 上的 OpenCL runtime 必须频繁地与 GPU 同步并补充新 kernel——一旦 CPU 频率过低导致 runtime 响应慢，GPU 就会出现 idle bubble，降低 GPU 利用率。

从kernel调度角度拆解：在 llama.cpp（OpenCL + CLBlast）的 LLM 推理中，每个 Transformer layer 的 MatMul (Q/K/V/O/FFN) 和 element-wise 算子被编译为一系列 OpenCL kernel，按序入队到 Mali GPU。由于仅 2 个 outstanding entries，CPU 必须在 GPU 完成前 2 个 kernel 之前提交第 3 个 kernel——这要求 CPU 的 OpenCL runtime 响应延迟 < GPU 2 个 kernel 的执行时间。若 CPU 频率过低（如 EAS 降至 500 MHz），runtime 响应延迟超过此窗口，GPU 将出现 idle bubble → GPU utilization 降低 → GPU governor 降频 → 形成拮抗效应。

涉及论文标题：
- Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

## LUT-Based Computation on NPU（基于查找表的 NPU 计算加速，vlut16 / vgather）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LUT-Based Computation 是一种用查表指令替代复杂数值计算的技术。在 Hexagon NPU 上，HVX 向量单元提供了两条关键的 LUT 指令：**vlut16**（16-entry 查表，每个 8-bit index → 16-bit output）和 **vgather**（从 TCM 收集 64 个 2-byte 元素到连续 128-byte 区域）。论文利用这两条指令将 Softmax 中的指数计算和 dequantization 中的 INT4→FP16 转换替换为查表操作，避免了向量单元执行复杂算术的低效问题（HVX ~33 GFLOPS vs. HMX ~12 TFLOPS）。

**vlut16**：输入一个 128-byte 向量寄存器（含 128 个 8-bit index），查一个 16-entry 的 LUT 表（每个 entry 为 16-bit），输出一对 128-byte 向量寄存器（每个 index 被替换为对应的 16-bit LUT 值）。用途：(1) INT4→FP16 转换——16-entry LUT = [FP16(-7), ..., FP16(7)]，每个 8-bit index 含 2 个 INT4 值，分别作为 index 查表；(2) Scales 广播——将 4 组 FP16 scales 作为 LUT，使用预定义 constant indices 在一次指令中广播到整个向量寄存器。

**vgather**：从 TCM 中收集 64 个 2-byte 元素（由 64 个 byte offsets 指定），存入连续 128-byte 区域。延迟：24-48 instruction packets (V75)。用途：实现 32768-entry FP16 exp LUT（64 KiB, 存储于 TCM），输入 Safe Softmax 的非正 FP16 值 → 忽略符号位 + 左移一位 → byte offset → vgather 查表。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== LUT-Based INT4→FP16 Dequantization (vlut16) =====
# 传统方法 (mask-unpack-convert, ~5-8 instructions per group):
#   v_int4 = vload(128B)
#   v_lo = vand(v_int4, 0x0F)         # mask low nibble
#   v_hi = vand(v_int4 >> 4, 0x0F)    # mask high nibble
#   v_fp16_lo = vconvert_i16_to_f16(v_lo)  # + qfloat conversion (V75-)
#   v_fp16_hi = vconvert_i16_to_f16(v_hi)
#   # Total: many sequential instructions, VLIW parallelism limited

# LUT方法 (vlut16, ~2 instructions per group):
LUT_INT4_TO_FP16 = [FP16(-7), FP16(-6), ..., FP16(0), ..., FP16(7)]  # 16 entries
v_int4 = vload_128byte(addr)                  # 128 bytes = 256 INT4
v_fp16_lo = vlut16(v_int4, LUT_INT4_TO_FP16)  # low nibble → FP16, 1 instruction
v_fp16_hi = vlut16(v_int4 >> 4, LUT_INT4_TO_FP16)  # high nibble → FP16
# V79+: direct IEEE-754 output (no qfloat overhead)
# V75-: vlut16 outputs IEEE-754 directly (avoids qfloat conversion!)

# ===== LUT-Based Scales Broadcast (vlut16) =====
# 传统: vscale = vsplat(scalar) + vcombine → 2 指令
# LUT方法:
scales_4group = vload_16byte(addr)            # 4 FP16 scales
# Extend to 16-entry LUT (repeat or pad)
scale_lut = [s0, s0, s0, s0, s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3]
CONST_INDICES = [0,0,0,0,0,0,0,0,  4,4,4,4,4,4,4,4,  # 32 lanes for lo
                 8,8,8,8,8,8,8,8, 12,12,12,12,12,12,12,12]  # 32 lanes for hi
v_scales_lo = vlut16(CONST_INDICES_LO, scale_lut)  # 1 instruction broadcast!
v_scales_hi = vlut16(CONST_INDICES_HI, scale_lut)

# ===== LUT-Based exp in Softmax (vgather) =====
# LUT预计算 (系统初始化, 64 KiB TCM):
lut_exp = [FP16(exp(-i/256.0)) for i in range(32768)]  # 32768 entries

# 在线exp (每个FlashAttention tile):
for each 64-element chunk in S_safe:  # S_safe ≤ 0 (safe softmax)
    v_abs = vand(v_chunk, 0x7FFF)       # 忽略符号位
    v_offsets = vasl(v_abs, 1)           # ×2 (FP16 = 2 bytes)
    # v_offsets: [0, 65534] → 覆盖 32768 entries × 2 bytes
    v_exp = vgather(lut_exp_base, v_offsets)  # 24-48 packets latency
    # 输出: 64 个 FP16 exp 值, 连续 128 bytes

# 加速比: 1.26-2.19× vs FP32 polynomial exp
#         1.60× vs FP16 polynomial exp (query=16, KV=16384)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **vlut16 的使用模式**：(a) 数值转换（INT4→FP16, NF4→FP16 等）——调整 LUT 内容即可支持不同编码；(b) 数据广播（scales, biases）——比 vsplat+vcombine 更高效；(c) 小规模非线性函数近似（如 GELU, SiLU 的分段线性近似）
- **vgather 的使用模式**：(a) 大规模 LUT（任意非线性函数, 可达 64 KiB）；(b) 间接寻址数据重组；(c) 稀疏数据收集
- **限制**：vlut16 的 LUT 仅 16 entries（适用于小范围映射如 INT4→FP16）；vgather 延迟较高（24-48 packets）——需权衡 LUT size vs. 指令延迟
- **通用性**：LUT 思想可推广至其他 NPU/加速器——任何有查表指令的 SIMD 架构均可受益

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Mixed-Precision Dequantization GEMM on NPU（NPU 上的混合精度反量化 GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixed-Precision Dequantization GEMM 是论文提出的在 Hexagon NPU 上高效执行 W4A16（INT4 权重 + FP16 激活）GEMM 的 kernel 设计。核心流程：HVX 向量单元执行 INT4 权重→FP16 的运行时反量化（使用 vlut16 LUT 指令），反量化结果按 HMX tile layout 连续写入 TCM，然后 HMX 矩阵单元执行 FP16 tile MatMul。多精度混合体现在：(1) 权重 INT4 存储 + FP16 计算（W4A16 方案）；(2) HMX 内部 FP32 累加 + FP16 I/O；(3) DMA（≥60 GB/s）负责 DDR↔TCM 批量数据传输，HVX 仅处理 TCM 上的反量化计算。

与 baseline（column-major 权重 + scatter dequantization + HMX MatMul）的关键区别：offline 权重布局变换使反量化结果连续写入 TCM（消除 scatter memory access），super-group coalesce 使每次 HVX 操作处理 256 个 INT4 值（填满 128-byte 寄存器）。加速 9.65–19.04×，仅比"no dequantization"（直接拷贝，理论上限）慢 27%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== Mixed-Precision Dequantization GEMM Kernel (NPU 端) =====
# 输入: FP16 activation A [batch, K] (DDR, 已 DMA→TCM)
#       Q4_0 super-group weights (DDR, DMA→TCM per tile column)
# 输出: FP16 result C [batch, N] (TCM→DDR)

# 权重格式 (offline prepared):
#   每 tile column [K=2048, 32] = 8 super-groups
#   每 super-group: [128 bytes INT4 (256 values)] + [16 bytes FP16 (8 scales)]

for tile_j in range(N // 32):  # 遍历权重 tile 列
    # DMA: 预取下一 tile 列的 8 个 super-groups 到 TCM
    dma_prefetch_2d(TCM_W, DDR_W + tile_j * 8 * 144, [8, 144])

    # === HVX Dequantization (向量单元) ===
    for sg in range(8):
        # 1. Load super-group
        v_int4 = vload_128(TCM_W + sg*144)           # 256 INT4 → 1 HVX register

        # 2. vlut16: INT4 → FP16 (低/高 4-bit 分别查表)
        v_fp16_lo = vlut16(v_int4, LUT_INT4_TO_FP16)  # low nibble
        v_fp16_hi = vlut16(v_int4 >> 4, LUT_INT4_TO_FP16)  # high nibble

        # 3. vlut16: scales broadcast (4 groups × 2 registers each)
        v_scales_4g = vload_16(TCM_W + sg*144 + 128)
        v_scales_lo = vlut16(CONST_IDX_LO, scales_lut)  # based on v_scales_4g
        v_scales_hi = vlut16(CONST_IDX_HI, scales_lut)

        # 4. FP16 multiply: dequantized = int_val × scale
        v_deq_lo = vmpy_f16(v_fp16_lo, v_scales_lo)
        v_deq_hi = vmpy_f16(v_fp16_hi, v_scales_hi)

        # 5. Continuous write to TCM (HMX tile layout, NO scatter!)
        vstore_128(TCM_WEIGHT_TILE + sg*64,      v_deq_lo)  # first 16 FP16
        vstore_128(TCM_WEIGHT_TILE + sg*64 + 128, v_deq_hi)  # next 16 FP16

    # === HMX FP16 MatMul (矩阵单元) ===
    hmx_load_activation(TCM_ACT_TILE)               # [32, 32] from TCM
    hmx_load_weight(TCM_WEIGHT_TILE)                 # [32, 32] from TCM
    hmx_matmul_accumulate()                           # FP32 internal accumulate
    # ... loop over tile rows ...
    hmx_store(TCM_OUT_TILE)                           # FP16 [32, 32] to TCM

# DMA writeback: TCM → DDR
dma_writeback_2d(DDR_C, TCM_OUT, [batch, N])

# ===== 性能 =====
# Baseline (column-major + scatter): 1× (slow)
# + HMX layout (continuous write, no scatter): 1.82-3.45×
# + Super-group coalesce (full HVX register utilization): 9.65-19.04× total
# vs. "no dequantization" upper bound: only 27% slower (near-optimal)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **量化方案支持**：论文使用 Q4_0（4.5 BPW, symmetric, group_size=32）用于大多数矩阵，Q8_0（8.5 BPW）用于 FFN down 矩阵。LUT 内容可灵活更换以支持 FP4, NF4, IQ4_NL 等编码方案
- **Super-group coalesce 原理**：8 个 group (每 group 32 INT4 = 16 bytes packed) → 128 bytes INT4（恰好 1 HVX 寄存器）+ 16 bytes FP16 scales。Coalesce 不是为了减少 memory access，而是为了让每次 HVX 操作填满 128-byte 寄存器宽度
- **与 QNN 的对比**：QNN per-tensor INT8 GEMM 仅使用 DMA+HMX（无 HVX 参与）→ 速度快但精度不可接受。本论文在 HVX dequantization 上付出 ~27% overhead 换取 per-group 量化精度
- **未来方向**：论文指出类似 T-MAC 的方法可在 NPU 上实现更高效的 GEMV dequantization（通过 table lookup 直接生成 dequantized 结果，进一步减少 HVX 指令数）

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CLBlast 是面向 OpenCL 设备的调优 BLAS（Basic Linear Algebra Subprograms）库，提供高度优化的矩阵乘法 (GEMM)、向量运算等 kernel 实现。在 FUSE 论文的 llama.cpp + OpenCL 配置中，CLBlast 作为 GPU 后端的数学库——llama.cpp 将 LLM 推理中的 MatMul 算子委托给 CLBlast 的 GEMM kernel 在 Mali GPU 上执行。CLBlast 为不同矩阵形状和硬件平台提供 auto-tuned kernel 参数（tile size, work-group size, vector width 等），通过 offline tuning 找到最优配置。

从kernel调度角度拆解：CLBlast 的 GEMM kernel 在 Mali-G710 GPU 上以 OpenCL work-group 形式并行执行。每个 work-group 处理输出矩阵的一个 tile（如 8×8 或 16×16），work-items 协作加载输入 tile 到 local memory 后执行乘加。CLBlast kernel 的执行时间由 GPU 频率决定（Mali-G710 上 151-848 MHz, 执行时间差异可达 5.6×），kernel 提交延迟由 CPU 频率决定（OpenCL runtime 运行在 CPU 上）。两者共同决定了 LLM 推理的端到端延迟——FUSE 的贡献正是协同优化这两个频率。

涉及论文标题：
- Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE
