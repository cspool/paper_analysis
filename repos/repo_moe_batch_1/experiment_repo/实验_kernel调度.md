# 实验_kernel调度

## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是三个运行时代码/调度层面的基础设施：(1) **XPUTimer**：轻量级分布式训练性能分析工具，集成在 DLRover 中。包含 lightweight selective tracing（Python/C++/CUDA 级截获，异步事件管理）和 diagnostic engine（O(1) 快速定位、细粒度诊断）。通过 event pool 管理、异步后台线程数据收集、数据压缩实现减少 90% 内存开销（~1.5MB/加速器/step）。(2) **EDiT (Elastic Distributed Training)**：基于 Local SGD 的高效异步分布式训练方法，包含 layer-wise synchronization（逐层同步+prefetch 重叠通信计算）、pseudo gradient penalty（异常 worker 排除 + 加权平均 + 梯度裁剪）、time-based synchronization（按时间而非固定步数触发同步，解决 straggler 问题）。(3) **PCache** + **Babel**：PCache 是全闪存分布式文件缓存系统，利用 FUSE（用户空间文件系统）+ shared memory 减少用户/内核态切换开销，metadata cache 加速随机读，AI co-design（分散 DP group checkpoint 写入）降低 checkpoint 写延迟 50%+ 峰值内存 60%。Babel 是跨集群数据同步中间件，支持 PB 级数据并行 metadata prefetch（190M 文件从 >6h 降至 ~10min，36× 加速）和 content-sampling CRC 校验（100GB 文件校验 ~3s）。

  实验比较：(a) XPUTimer 内存使用 vs 其他 profiling 方法（减少 ~90%）；(b) EDiT vs 传统同步分布式方法在速度上的对比（最大加速 66.1%）；(c) PCache vs GPFS checkpoint 写延迟（70s vs 160s @ 128 accelerators, 90s vs 240s @ 512 accelerators）；(d) Babel 并行 metadata prefetch vs 串行（>6h vs ~10min）；(e) Babel CRC 校验 vs MD5（~3s vs tens-to-hundreds of seconds for 100GB files）。

- 后端平台是什么，配置是什么。
  异构 AI 加速器集群（Device A~E，见算法 pipeline 条目）。集群规模从 128 到 10,000+ 加速器。PCache 在 1,000 加速器集群上聚合吞吐 1 TB/s，10,000 加速器线性扩至 8 TB/s。EDiT 性能评估在理想环境和异构环境中分别测试。

- 评估性能的软件/脚本是什么。修改了什么。
  - **XPUTimer** 集成在 **DLRover**（[https://github.com/intelligent-machine-learning/dlrover](https://github.com/intelligent-machine-learning/dlrover)）中，独立开源。修改：(a) 在 Python 层通过环境变量 TRACED_PYTHON_API 动态配置需要监控的 API；(b) C++/CUDA 层通过框架无关的 kernel 截获机制监控 cuBLAS、Flash Attention、NCCL 操作及自定义算子；(c) CUDA event 注入 NCCL kernel launch 后并通过后台线程异步监控完成状态。
  - **EDiT** 修改分布式训练同步逻辑：在每个 worker 上逐层独立 forward→backward→sync（而非等所有 worker 完成整个 step 后 All-Reduce），pseudo gradient 使用 EMA 跟踪检测异常 worker 并排除。
  - **PCache** 修改 checkpoint 写入逻辑：将 Megatron 默认的 DP group rank_0 集中写入改为分散写入不同物理节点，通过 FUSE + shm 消除多次用户/内核态切换和数据拷贝。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  DLRover: [https://github.com/intelligent-machine-learning/dlrover](https://github.com/intelligent-machine-learning/dlrover)。XPUTimer 诊断流程：

  ```
  === XPUTimer Selective Tracing 原理 ===

  Step 1: Python 层截获（错误拦截）
    export TRACED_PYTHON_API = "gc.collect,torch.cuda.synchronize,DataLoader.__iter__"
    # 运行时通过 monkey-patch 注入 hook
    @trace_hook
    def traced_synchronize():
        t_start = record_timestamp()         # 同步 API 记录时间戳
        original_synchronize()               # 执行原始操作
        t_end = record_timestamp()

  Step 2: C++/CUDA 层截获（框架无关 kernel 监控）
    # 以 NCCL AllReduce 为例
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, compute_stream);
    ncclAllReduce(...);                       # 原始通信 kernel
    cudaEventRecord(ev_stop, compute_stream);

    # 后台线程异步检查
    background_thread:
      while training:
        if cudaEventQuery(ev_stop) == cudaSuccess:
          elapsed = cudaEventElapsedTime(ev_start, ev_stop)
          log({kernel: "ncclAllReduce", time: elapsed, layout: input_dims})

  Step 3: 低开销设计
    # Event Pool: 预分配并复用 CUDA event，避免动态分配
    ev_pool = [cudaEventCreate() for _ in range(MAX_PENDING_EVENTS)]

    # 数据压缩: 仅记录时间戳 + kernel 输入布局
    log_entry = {ts: t_start, kernel: "gemm", m: 4096, n: 1536, k: 5120}
    # 不保存完整 tensor 内容，压缩后 ~1.5MB/accelerator/step

  === EDiT 异步训练流程 ===

  Workers W_0, W_1, W_2, W_3 (4 workers example):

  for step in training:
    # 各 worker 独立 forward + backward
    for layer in model:
      # Forward (并行)
      worker_i: hidden = layer.forward(hidden)
      # Layer-wise sync: 完成一层后立即同步该层
      if layer % sync_interval == 0:
        broadcast_layer_weights()  # 非阻塞 prefetch

      # Backward
      grad = layer.backward(loss)

    # Pseudo Gradient Penalty
    pseudo_grad_i = (current_params - prev_params) / lr
    # 1. 异常检测: EMA 跟踪 pseudo_grad 的 norm
    if |norm(pseudo_grad_i) - EMA(norm)| > threshold:
        exclude_worker(i)  # 异常 worker 被排除

    # 2. 加权平均: 按 pseudo_grad norm 加权
    weights = softmax([1/norm(pg_j) for j in valid_workers])
    fused_grad = sum(w_j * pseudo_grad_j)

    # 3. 梯度裁剪
    fused_grad = clip_by_norm(fused_grad, threshold)

  # Time-based sync（非固定步数）
  if elapsed_time > sync_deadline:   # 达到时间阈值
      synchronize_all_workers()

  === PCache Checkpoint 写入优化 ===

  # Megatron 默认: rank_0 集中写入
  Default: all DP groups write through rank_0 of each group
            → 集中到少数物理节点 → CPU+网络拥塞

  # PCache AI Co-design: 分散写入
  Optimized: round-robin assign write target per DP group
    dp_group_0 → physical_node_0
    dp_group_1 → physical_node_7
    dp_group_2 → physical_node_3
    ...

  # FUSE + shm 加速
  Write path:
    App → FUSE (userspace) → shm (shared memory, zero-copy) → NVMe SSD
    # 避免传统路径: App → kernel VFS → kernel FS → block layer → SSD
  ```

## EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  EPS-MoE 在 kernel 级别的核心实现包括：
  1. **Load-aware GEMM 动态切换**：根据输入 token 数 m 动态选择 GroupGemm（cutlass，小 m 时更优）或 DenseGemm（cublas，m≥4096 时更优）。通过 profiling 数据发现：m<2048 时 GroupGemm 效率高于 DenseGemm；m≥4096 时 DenseGemm 反超。
  2. **水平切分（Horizontal Split）输入张量**：对 MoE 输入按行切分，权重按专家切分。相比传统垂直切分（列切分），水平切分避免了参数矩阵的重复内存 I/O。当 pipeline 数 N=E（专家数）时，GroupGemm 退化为 DenseGemm，获得更高计算效率。
  3. **SM 数量控制优化**：控制 GEMM kernel 占用的 SM 数量（如 H800 上从 132 降至 116），留出 SM 资源给 all2all 通信 kernel（16 SM），实现计算与通信在 SM 级别的并行调度。通过 SM 控制，避免了 GEMM 独占所有 SM 导致通信 kernel 无法调度的问题。
  4. **Pipeline 重叠调度**：将 MoE FFN（Gate/Up/Down GEMM）与 all2all 通信以 pipeline 方式重叠，通过分 N 个 pipeline stage 顺序提交专家组计算，每轮计算时下一组通信已在并行执行。

  实验比较：
  - GroupGemm vs DenseGemm 在不同输入规模下的吞吐量对比（图5a-c）
  - 不同 SM 数量下的 GEMM + all2all 重叠延迟（表6，SM 从 92 到 132）
  - 不同 Pipeline Number（PN=1,5,20）的影响（表5、图11）
  - 不同输出维度（1536 vs 15360）下的性能差异
  - FP8 通信 + 重叠策略 vs 纯 FP8 vs 纯重叠（消融表5）
  - 计算 ε0（通信时间变化系数）和 ε1（计算时间变化系数）确定 EPS-MoE 有效工作区间（图10、表7）

- 后端平台是什么，配置是什么。
  - NVIDIA H800-80GB SXM GPU（132 SM，NVLink 互联），4x/8x 配置
  - NVIDIA A100-80GB SXM GPU（用于 profiling 和通信测试）
  - 矩阵计算库：cutlass（GroupGemm）、cublas（DenseGemm）
  - 通信原语：ncclAllReduce、ncclReduceScatter、ncclAllGather、all2all
  - 通信加速：Flux（kernel fusion 优化通信与 GEMM 的融合）

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 real model settings 的 GEMM profiling：
  1. **GEMM 效率 Profiling**（图5）：
     - 测试矩阵维度：Gate/Up 矩阵 [1536, 5120]（与 DeepSeekV2 一致），16 Experts
     - 变输入 m（batch token 数），测试 GroupGemm 和 DenseGemm 的吞吐量
     - 变 GroupGemm 的 group 数和 SM 数，测试相对吞吐量
  2. **Kernel 延迟 Profiling**（图6、7）：
     - 测试 silu_activation（memory-bound kernel）在不同 load 和 SM 数下的延迟
     - 测试 all2all 通信 kernel 在不同 load 和 SM 数下的延迟
  3. **消融实验脚本**（表5、6、7）：
     - 单层 Gate/Up/Down GEMM 的 MoE 层测试（输入 dim=5120, 输出 dim=1536 或 15360）
     - 变 m (256/1024/3072)、PN (1/5/20)、SM (92-132)、FP8 on/off、Overlapping on/off
     - 测试 DeepSeekV2 全模型 prefill throughput（表2）
     - 测试 Mixtral 8x7B / DBRX / Snowflake Arctic 的 TTFT（表3、4）
  4. **修改内容**：
     - 在 vLLM 框架中替换 MoE FFN 的 GEMM 调用路径，根据 load 动态选择 cutlass GroupGemm 或 cublas DenseGemm
     - 实现水平切分输入张量的逻辑，修改 weight partition 方式为按专家切分
     - 添加 SM 控制接口，限制 GEMM kernel 的 CUDA stream 优先级或 MPS 配置
     - 实现 all2all + GEMM pipeline 调度器，管理 token dispatch 和专家计算的流水线

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源代码。评估方法基于 cutlass/cublas 的 GEMM profiling 和 vLLM 集成。
  
  **评估原理和 Kernel 输入到性能输出全过程**：
  
  **阶段 1: GEMM Profiling（独立 kernel 评估）**
  - **输入**：MoE FFN 矩阵维度（如 Gate 矩阵 [1536, 5120]，16 experts），输入 token 数 m∈{256, 1024, 3072, ...}
  - **原理**：对每个 (m, dim, experts, groups) 配置，分别调用 cutlass GroupGemm 和 cublas DenseGemm（cublasGemmEx），测量 wall-clock 延迟。用 Nsight Systems/CUPTI 采集 SM 利用率、内存带宽、计算吞吐。
  - **输出**：各配置下的吞吐量（TFLOPS）、延迟（ms），绘制图5 的效率曲线，确定 GroupGemm vs DenseGemm 的交叉点（约 m=2048~4096）。
  
  **阶段 2: 通信 Profiling**
  - **输入**：all2all 通信数据量（取决于 batch token 数和每个 token 的 activation size）
  - **原理**：调用 ncclAllReduce/ncclReduceScatter/ncclAllGather/all2all，测量不同数据量和 SM 数下的延迟。测试 NVLink 带宽利用率。
  - **输出**：图7 的通信延迟曲线，确定通信 kernel 的 SM 需求（10-20 SM 足够，超过无显著改善）。
  
  **阶段 3: 单层 MoE Pipeline 评估**
  - **输入**：单层 MoE（Gate+Up+Down GEMM，dim [5120, 1536] 或 [5120, 15360]），配置 (PN, SM, FP8, overlapping 开关)
  - **流程**：
    1. 输入 tokens 按行切分为 N 组（N=PN）
    2. 每组 tokens 先进行 all2all dispatch 通信（如使用 FP8 量化则先量化）
    3. 同时启动下一组 all2all 和前一组 GEMM 计算（overlap）
    4. GEMM 根据 m/N 的大小选择 GroupGemm 或 DenseGemm
    5. GEMM SM 限制为 116（留 16 SM 给通信）
    6. 完成 all2all combine 聚合结果
  - **输出**：各配置下的单层延迟（ms），表5/6/7 的消融数据。
  
  **阶段 4: 全模型 Serving 评估**
  - **输入**：完整 MoE 模型（DeepSeekV2/Mixtral/DBRX），请求 batch（变 seqlen 和 batchsize）
  - **流程**：EPS-MoE 集成到 vLLM，端到端推理。对每层 MoE block 应用 expert pipeline scheduler。
  - **输出**：Prefill throughput（tokens/s）和 TTFT（s），表2/3/4。

## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是针对 DualSparse-MoE 推理系统的 **Triton kernel 优化**，主要解决 token-expert grouped-GEMM 在计算粒度变化和额外控制操作引入后的效率问题：(1) **Token-Expert Grouped-GEMM Triton Kernel 优化**：由于 2T-Drop 的 dual-threshold 机制导致 token-expert 计算粒度从均匀变为非均匀（有些 expert 被完全跳过、有些仅计算 major half、有些计算完整 expert），标准的 grouped-GEMM kernel 无法高效处理这种变长计算模式。论文使用优化的 Triton kernel 来适配变粒度 grouped-GEMM，在保持 2T-Drop 细粒度计算丢弃的同时，实现与 1T-Drop（粗粒度丢弃）相当的计算效率。(2) **Gating 函数控制逻辑的 Kernel 集成**：将 dual-threshold comparison、expert selection filtering、major/minor expert dispatch 等控制逻辑集成到 Triton kernel 中，减少 host-device 数据传输和 kernel launch 开销。实验比较：在 8×H20 GPU 节点上，1T-Drop vs 2T-Drop 在不同模型（Mixtral TP=8、OLMoE single GPU、DeepSeek EP=8）下的实际 speedup，验证 22%-27% 的 MoE computation drop rate 能有效翻译为 1.17-1.23× MoE module speedup 和 1.07-1.12× end-to-end speedup。

- 后端平台是什么，配置是什么。
  8×NVIDIA H20 GPU 服务器。软件栈：PyTorch + Triton kernel language + SGLang framework + NCCL backend。部署配置：(a) Mixtral-8×7B：TP=8（8×H20 单节点）；(b) OLMoE-Instruct：单 H20 GPU；(c) DeepSeek-V2-Lite-Chat：EP=8（8×H20 单节点）。Speedup 评估：2,000 条随机 prompts（input length=500, output length=100）。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 SGLang inference framework 的 Python/Triton 实现。主要修改：(a) 实现 token-expert computation dropping 的定制 Triton grouped-GEMM kernel，支持变粒度（skip / major-only / full）expert 计算；(b) 将 gating 函数中的 dual-threshold comparison + expert dispatch 逻辑融合到 inference kernel pipeline 中；(c) 确保 2T-Drop 的细粒度计算丢弃不引入额外 kernel launch 开销（与 1T-Drop 保持相同 speedup 水平）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供独立开源代码仓库，基于 SGLang 框架的 Triton kernel 实现。Kernel 原理如下：

  ```
  === Triton Token-Expert Grouped-GEMM Kernel Flow ===

  Input (after gating + dual-threshold decision):
    token_hidden_states: [num_tokens, d_model]
    expert_assignments: [[(expert_id, compute_mode), ...] for each token]
      compute_mode ∈ {SKIP, MAJOR_ONLY, FULL}
    expert_weights: {major_W1, major_W2, major_W3, full_W1, full_W2, full_W3}

  Step 1: Token-Expert Dispatch (Triton kernel)
    Group tokens by compute_mode and target expert:
      For each token t:
        For each activated expert e_j of token t:
          mode = decide(thresholds, s_norm_j)
          if mode == SKIP: continue
          elif mode == MAJOR_ONLY:
            dispatch to group: (expert_e_j, MAJOR)
          else:  # FULL
            dispatch to group: (expert_e_j, FULL)

    Output: per-group token indices + per-group weight pointers

  Step 2: Variable-Length Grouped-GEMM (Triton kernel)
    For each (expert, mode) group:
      tokens_in_group = gathered hidden states
      if mode == MAJOR_ONLY:
        gate_out = Swish(tokens · W_1_major)
        up_out = tokens · W_3_major
        hidden = gate_out ⊙ up_out
        output = hidden · W_2_major
      elif mode == FULL:
        gate_out = Swish(tokens · W_1_full)
        up_out = tokens · W_3_full
        hidden = gate_out ⊙ up_out
        output = hidden · W_2_full

  Step 3: Scatter + Weighted Sum (Triton kernel)
    For each token t:
      y_t = 0
      For each computed expert e_j of token t:
        y_t += s_{e_j} · expert_output[e_j]  // s is original gating score
      token_output[t] = y_t

  === Performance Translation ===
  22%-27% computation drop rate → 1.17-1.23× MoE module speedup
  Key insight: dropping at tensor-level (expert/token granularity)
  enables effective GPU utilization vs fine-grained neuron-level sparsity
  that struggles to convert to real speedup on current hardware
  ```

## DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是针对 MoE 推理的专用 CUDA kernel 优化，共三类：**(1) Gating Function Kernel Fusion**：将 top-k 选择、cumsum（Token ID per expert）、scatter 操作融合为单一 CUDA kernel，使用 dense token-to-expert mapping table 替代 sparse one-hot 掩码。cumsum 使用 Blelloch scan 算法并行化。**(2) Data Layout Transformation 替代 Sparse Einsum**：将 sparse einsum 的 token sorting（按 expert ID 排序输入 token）和 re-sorting（恢复原始 token 顺序）实现为基于 mapping table 的数据布局变换，复杂度从 S×E×M×c^e 降至 S×M×c^e，并融合 gating logits 概率缩放。**(3) Expert 参数内的并行运算优化**：利用 DeepSpeed inference 的高带宽利用率 Transformer kernel 处理 non-expert 层。实验比较：MoE kernel 延迟降低 6x+；per-GPU 吞吐随 GPU 数量增加而超线性增长（super-linear throughput scaling）；不同 MoE 模型规模（107B→2T params）的端到端推理延迟和吞吐量。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（Azure ND A100 instances），最多 256 GPUs。节点内 8 GPUs 通过 NVLink 互联，节点间 Mellanox InfiniBand 互联。软件栈：DeepSpeed-MoE，PyTorch distributed，NCCL / Microsoft SCCL 通信后端，自定义 CUDA kernels。

- 评估性能的软件/脚本是什么。修改了什么。
  DeepSpeed-MoE inference framework（开源）。主要修改：(a) 实现 MoE Gating 融合 kernel —— 单 kernel 内完成 top-k + cumsum (Blelloch scan) + scatter，使用 dense mapping table；(b) 实现 data-layout transformation kernel 替代 sparse-dense einsum，将 token 排序/反排序作为显式内存布局操作；(c) 在 token 反排序时融合 gating logits 的概率域缩放；(d) 优化所有 MoE 相关 kernel 为 dense representation 消除稀疏张量运算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源，代码位于 https://github.com/microsoft/DeepSpeed。

  **MoE Gating + Token Routing Kernel 原理**：
  ```
  Input: S tokens, hidden dimension M, E experts, k=1 (top-1 gating)

  === Kernel 1: Fused Gating Kernel (单个 CUDA kernel) ===
  Input:  gate_logits[S][E]   // raw logits from gating linear layer

  // Step A: Top-k selection (k=1)
  For each token t in parallel:
    expert_id[t] = argmax(gate_logits[t])    // 选 logit 最高的 expert
    // 存入 dense mapping table, 不使用 sparse one-hot

  // Step B: Cumsum (Blelloch scan) - 计算每个 expert 处理多少 token
  // Parallel prefix sum on GPU:
  expert_counts[E] = {0}
  For each token t in parallel:
    atomicAdd(expert_counts[expert_id[t]], 1)
  // Blelloch scan to compute exclusive prefix sum:
  token_offset[0] = 0
  BlellochScan(expert_counts) → expert_offset[E]
    // expert_offset[i] = 起始位置 for expert i's tokens

  // Step C: Scatter - 计算每个 token 在其对应 expert 中的局部 ID
  For each token t in parallel:
    local_id[t] = atomicAdd(expert_offset[expert_id[t]], 1)

  Output: expert_id[S], local_id[S], expert_offset[E+1]
  // Complexity: O(S) parallel, S×E reduced to dense mapping table

  === Kernel 2: Data Layout Transformation (替代 Sparse Einsum) ===
  Input:  activations[S][M], expert_id[S], local_id[S], expert_offset[E]

  // Forward pass: 按 expert ID 重排 token 顺序 (sort)
  Output: sorted_acts[E][ce][M] = {0}   // ce = expert capacity
  For each token t in parallel:
    e = expert_id[t]
    pos = local_id[t]
    sorted_acts[e][pos] = activations[t]      // 直接 memcpy, 无 sparse 乘法

  // Expert FFN computation (per-expert, standard linear layers)
  For each expert e in parallel:
    expert_output[e] = W2_e @ GeLU(W1_e @ sorted_acts[e])

  // Backward pass: 恢复原始 token 顺序 (unsort) + probability scaling
  Input: expert_output[E][ce][M], expert_id[S], local_id[S], gate_probs[S]
  Output: final_output[S][M]
  For each token t in parallel:
    e = expert_id[t]
    pos = local_id[t]
    final_output[t] = gate_probs[t] * expert_output[e][pos]  // 融合概率缩放

  // 复杂度分析:
  // Sparse Einsum: S × E × M × ce = O(S·E·M·ce) → 含大量零乘法
  // 优化后: S × M × ce = O(S·M·ce) → 仅移动非零元素
  ```
  
  关键优化点：
  - Sparse einsum 中 (E-1)/E 的运算为与零相乘 → 完全消除
  - 从立方复杂度 S×E×M×ce 降至二次 S×M×ce
  - 多个 kernel launch 融合为单一 kernel → 减少 launch overhead
  - Dense mapping table 替代 sparse mask → 减少内存和计算开销
  - 组合优化实现 MoE kernel 延迟降低 6x+

## DeepSeek-V3 Technical Report

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现包括三部分：**(1) DualPipe Pipeline Parallelism Algorithm**：双向流水线调度（bidirectional pipeline），将每个 chunk 拆分为 attention/all-to-all dispatch/MLP/all-to-all combine 四个组件，后向 chunk 的 attention 和 MLP 进一步拆分为 backward for input 和 backward for weights（类似 ZeroBubble）。手动调整 SM 比例实现通信与计算完全重叠，pipeline bubble 比 1F1B 和 ZB1P 更少。**(2) Custom Cross-Node All-to-All Communication Kernels**：使用 warp specialization 技术，20 SMs 分为 10 个通信通道。Dispatching：IB send → IB-to-NVLink forwarding → NVLink receive，各由专用 warp 处理。Combining：NVLink send → NVLink-to-IB forwarding+accumulation → IB receive+accumulation。动态调整每 task 的 warp 数，使用定制 PTX 指令和 auto-tuned chunk size 减少 L2 cache 干扰。**(3) FP8 GEMM with Promotion to CUDA Cores**：Tensor Core 执行 WGMMA，每 N_c=128 elements 将中间结果拷贝到 CUDA Core 的 FP32 寄存器进行高精度累积，解决 H800 Tensor Core 仅 14-bit 累积精度限制；同时 dequantization scaling factor 乘在 CUDA Core 上。实验比较：DualPipe vs 1F1B vs ZB1P 的 pipeline bubble 和峰值内存；FP8 vs BF16 训练的 loss error 对比（<0.25%）；all-to-all kernel SM 占用效率（仅 20/132 SMs）。

- 后端平台是什么，配置是什么。
  NVIDIA H800 GPU 集群（2048 GPUs）。节点内 NVLink 160 GB/s + NVSwitch，节点间 InfiniBand 50 GB/s。H800 每 GPU 132 SMs，Tensor Core FP8 GEMM 默认仅 ~14-bit 累积精度。软件栈：自研 HAI-LLM 框架，PyTorch distributed (NCCL backend)，定制 CUDA kernel 使用 PTX (Parallel Thread Execution) 指令。

- 评估性能的软件/脚本是什么。修改了什么。
  自研 HAI-LLM 训练框架。主要修改：(a) 实现 DualPipe 调度器替代标准 1F1B pipeline schedule；(b) 手写 cross-node all-to-all CUDA kernels 替代 NCCL all-to-all，含 warp specialization 和 PTX 优化；(c) 修改 FP8 GEMM 实现，将 Tensor Core WGMMA 与 CUDA Core FP32 promotion 交错调度，使两个 warpgroup 交替执行；(d) 实现 RMSNorm 和 MLA up-projection recomputation 策略；(e) EMA 参数异步更新在 CPU 内存中。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  训练框架 HAI-LLM 未开源。模型 checkpoint 开源在 https://github.com/deepseek-ai/DeepSeek-V3。

  **DualPipe 调度原理**：
  ```
  正向: [Attention] [Dispatch] [MLP] [Combine] [PP_Comm]
  反向: [Attn_BW_In] [Attn_BW_W] [Disp_BW] [MLP_BW_In] [MLP_BW_W] [Comb_BW] [PP_Comm]
  
  重叠策略（一对 forward+backward chunk）:
  Time ──────────────────────────────────────────────────────►
  Fwd_Chunk_A:  [Attn][Dispatch][ MLP  ][Combine]
  Fwd_Chunk_B:                      [Attn][Dispatch][MLP][Combine]
  Bwd_Chunk_A:  [Attn_BW][Disp_BW][MLP_BW][Comb_BW]
  Bwd_Chunk_B:                               [Attn_BW][Disp_BW][MLP_BW]
  
  双向调度: micro-batches 从 pipeline 两端同时注入
  Bubble = (PP-1)/(PP) * (F&B-3W)/(F+B-W)  ← 比 1F1B 和 ZB1P 更小
  ```

  **Cross-Node All-to-All Kernel 流程**：
  ```
  // Dispatching (20 SMs, 10 channels, warp specialization)
  Input: token activations [N_tokens, d_model] in FP8
  
  // Channel allocation is dynamic per workload
  For each token:
    1. IB Send Warp: memcpy(HBM → RDMA buffer), post IB send to target node
    2. IB-to-NVLink Forward Warp (on target node):
       IB recv → shared memory → NVLink send to target GPU within node
    3. NVLink Receive Warp (on target GPU):
       NVLink recv → HBM (expert input buffer)
  
  // Combining (reverse direction)
  For each token (after expert computation):
    1. NVLink Send Warp: HBM → NVLink send to aggregation GPU
    2. NVLink-to-IB Forward + Accumulate Warp:
       NVLink recv → FP32 accumulation in shared mem → IB send
    3. IB Receive + Accumulate Warp (on source node):
       IB recv → FP32 accumulate → HBM (final output)
  
  PTX optimization: custom ld.global/st.global with cache bypass hints
  Auto-tuned chunk size: minimize L2 cache eviction interference with compute SMs
  ```

  **FP8 GEMM with CUDA Core Promotion**：
  ```
  Input: A [M, K] in FP8 (1×128 tile quantized), B [K, N] in FP8 (128×128 block quantized)
  
  Warpgroup-0:                    Warpgroup-1:
  WGMMA(0..127, :, :)            (idle)
  │                              │
  ├─ partial sums → FP32 regs    ├─ WGMMA(128..255, :, :)
  │  (CUDA Cores promotion)      │  (executes while WG-0 promotes)
  │  × scale_A[group] × scale_B  │
  │  + accumulate to FP32        │
  └─ finish → (idle)             ├─ partial sums → FP32 regs
                                 │  (CUDA Cores promotion)
  // Continue alternating every N_c=128 elements
  Output: C [M, N] in FP32/BF16
  ```

## Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 LUFFY 的 **Sequence Migration Controller** 和 **Token Condensation Scheduler** 两个运行时调度模块：
  
  1. **Sequence Migration Controller**：在 combine phase 中，一个指定的 controller 机器收集所有 GPU 的 token 分布信息（token_to_sequence、token_to_gpu、sequence_to_gpu 三张哈希表），运行迁移算法决定每个 sequence 应在哪个 GPU 重构以最小化跨 GPU token 拉取流量，并优化后续 attention 计算的效率（通过将相似长度 sequence 聚集到同一 GPU 减少 padding zeros）。迁移决策通过 `torch.distributed.rpc` API 指导 GPU 间的 token 交换。
  
  2. **Token Condensation Scheduler**：每个 GPU 维护一个独立 CUDA stream 运行 token condensation scheduler。使用 DGL (Deep Graph Library) API 构建 token 图，通过 edge-wise 函数 `edge_sim_calculation` 并行计算 token 间相似度，维护 `token_to_token` 哈希表记录凝聚映射。

  实验比较（与其他模块混合评估）：
  - **Ablation Study (Fig. 9)**：Token Condensation Only vs Sequence Migration Only vs Both，在三种 MoE 模型上分析各调度模块的独立贡献
  - **Sensitivity Analysis (Fig. 10a, 10b)**：候选 GPU 数 q 对 traffic 和 computation time 的影响；cost model 估计精度（平均误差 ~5%）
  - **Performance Breakdown (Table III)**：Computation time vs Communication time 分解，验证调度优化的效果

- 后端平台是什么，配置是什么。
  16× NVIDIA V100 GPU (16GB HBM)，PCIe 互联，无 NVLink。Ubuntu 20.04 (kernel 5.15)，NVIDIA driver 525.85，CUDA 11.7，cuDNN 8.6.0。底层通信使用 PyTorch distributed (NCCL backend)。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**：PyTorch Profiler 采集各 phase 的 computation/communication time。端到端训练迭代时间（iteration time）作为主要性能指标，所有方法在相同配置下归一化到 Vanilla 计算 speedup。
  
  **LUFFY 对 PyTorch 的修改（~4.5K 行 Python 代码）**：
  1. **Sequence Migration Controller**：新增集中式 controller 模块
     - 收集分布式信息：在 expert running 期间并行收集 token_to_sequence、token_to_gpu、sequence_to_gpu 映射
     - 迁移算法：对每个 sequence i，估算迁移到各 GPU j 的 token 拉取流量 f_{i,j} → 选择 top-q 候选 GPU H^i → 通过 cost model T_att(B, L) 评估 attention 计算时间增长 → 选择使计算成本增长最小的 GPU j*
     - 通过 `torch.distributed.rpc` 更新 sequence_to_gpu 哈希表，指导 combine phase 的 token 路由
  2. **Token Condensation Scheduler**：每 GPU 独立 CUDA stream
     - DGL 图构建：node = token (features: expert index + token embedding)，edge = token pair
     - 三步快速相似度测量：expert activation filtering → historical similarity lookup → cosine similarity 计算
     - 图剪枝 + 连通分量：根据自适应阈值 h_t 删除低相似度边 → 每子图保留 degree 最高的 token，其余凝聚
     - 维护 token_to_token 哈希表指导 dispatch/combine 的 token 替换
  3. **Cost Model**：$T_{att}(B, L) = (3BLd² + 2BL²d) / P$，其中 P 通过 profiling attention 层多次获得
  
  评估原理：每个 training iteration 各 phase 的时间通过 `torch.cuda.Event` 精确测量 → batch training time = attention compute + token condensation + dispatch all-to-all + expert compute + sequence migration + combine all-to-all → speedup = Vanilla_time / LUFFY_time

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源情况：论文未提供公开开源代码仓库。LUFFY 以 PyTorch plug-and-play 插件形式实现，依赖 DGL (Deep Graph Library) 和 PyTorch distributed。

  **Sequence Migration 调度全流程（以 4 GPU, MoE-TransformerXL, 1 个 training batch 为例）**：

  ```
  === 输入状态 ===
  - GPU 0-3 各持有 1 个 expert 和完整的 attention 参数
  - Batch 包含 B=8 个 sequences，长度不等
  - Expert parallelism: top-2 gating

  === Phase 1: Attention Computation (所有 GPU 本地) ===
  每个 GPU 处理其分配的 sequences:
    for seq in local_sequences:
        Q, K, V = Linear_QKV(seq)  # [L, d] → [L, 3d]
        attn_out = Softmax(QK^T/√d) V  # FlashAttention
    → 输出: token embeddings for all local sequences

  === Phase 2: Token Dispatch (All-to-All) ===
  Router 计算 gate → token 路由到 expert 所在 GPU:
    GPU 0 → tokens for Expert 1,2,3,4 → all-to-all scatter
    → GPU 0 收到: tokens routed to Expert 0 (local) + others' tokens

  === Phase 3: Expert Computation (与 Sequence Migration 并行) ===
  GPU 0 计算 Expert 0 的 FFN:
    out = expert_ffn(received_tokens)  # W_gate → SiLU → W_up → × → W_down
  同时，Controller 执行迁移算法:
    
    # 收集分布信息（与 expert running 并行）
    for each GPU g:
        gather(token_to_gpu[g])  # 每个 token 在哪个 GPU 被 expert 处理
    
    # Algorithm 1: Sequence Migration
    for each sequence i (i = 1..B):
        # Step 1: 估算迁移到各 GPU 的 combine 流量
        for each GPU j:
            f_{i,j} = count({token t in seq i | token_to_gpu[t] != j})
        H^i = top_q(argmin(f_{i,j}))  # 候选 GPU 集合
        
        # Step 2: 选择最小 attention cost 增长的 GPU
        for each GPU j in H^i:
            B_{j←i} = current_sequences_on_gpu(j) + [seq i]
            L_{j←i} = max_length(B_{j←i})
            s_{i,j} = T_att(B_{j←i}, L_{j←i}) - T_att(B_j, L_j)
        
        j* = argmax(s_{i,j})  # min cost growth
        
        # 检查容量: GPU 可容纳更多短序列，但有限的长序列
        if GPU j* has capacity:
            sequence_to_gpu[i] = j*
  
  === Phase 4: Sequence-Aware Combine (基于迁移决策) ===
  原始 Vanilla: 所有 token 拉回原 GPU → inter-GPU traffic 大
  LUFFY: Controller 广播 sequence_to_gpu 映射 →
    GPU 0 sequences: [seq_0, seq_3] (migrated here)
    → tokens of seq_3 pulled from GPU 1,2,3 to GPU 0
    → token pulling traffic 大幅减少
    → sequences 在迁移目标 GPU 上重构

  === Phase 5: Next Block Attention (优化的 batch) ===
  GPU 0 收到 seq_0 (len=250) + seq_3 (len=230):
    → 相似长度 → 仅需 padding 20 个 zeros
    → vs Vanilla: 混合长短序列 → padding 浪费大
  
  === 性能输出 ===
  - Communication time reduction: 1.76×-3.72× vs Vanilla
  - Computation time reduction: 1.16×-1.57× vs Vanilla  
  - Overall speedup: 1.51×-2.73× vs Vanilla (16 experts)
  ```

  **Token Condensation Scheduler 执行流程（单 GPU, CUDA Stream）**：

  ```
  === 输入 (Attention 输出后) ===
  - tokens: N 个 token embeddings [N, d]
  - gate_output: {token_idx → expert_idx}
  - historical_similarity: 来自 block (b-1) 的相似度缓存
  - loss_prev: 上一 iteration 的 loss 值

  === Step 1: DGL Graph Construction (CUDA Stream) ===
  g = dgl.graph((src_nodes, dst_nodes))
  g.ndata['expert'] = gate_output  # token → expert mapping
  g.ndata['embedding'] = tokens    # token embeddings

  === Step 2: Fast Edge Weight Computation ===
  g.apply_edges(edge_sim_calculation):
      for each edge (u, v):
          # 2a: Expert activation filter
          if g.ndata['expert'][u] != g.ndata['expert'][v]:
              return {'weight': 0.0}
          
          # 2b: Historical similarity lookup (O(1))
          s_prev = historical_cache.get((u, v))
          if s_prev is not None:
              if s_prev > S1: return {'weight': 1.0}
              if s_prev < S2: return {'weight': 0.0}
          
          # 2c: Real cosine similarity (O(d))
          emb_u = g.ndata['embedding'][u]
          emb_v = g.ndata['embedding'][v]
          sim = dot(emb_u, emb_v) / (norm(emb_u) * norm(emb_v))
          return {'weight': sim}

  === Step 3: Adaptive Threshold ===
  l_norm = (loss_ini - loss_prev) / loss_ini
  h_t = 1.0 / (1.0 + exp(l_norm))
  # 若 loss 下降大 → l_norm 大 → h_t 小 → 凝聚更多 token

  === Step 4: Graph Pruning + Component Selection ===
  # 移除 weight < h_t 的边
  g_sparse = g.edge_subgraph(g.edges()[g.edata['weight'] >= h_t])
  
  # 连通分量分析
  components = dgl.connected_components(g_sparse)
  
  # 每个分量保留 degree 最大的 token
  for comp in components:
      degrees = g_sparse.in_degrees(comp)
      rep = comp[argmax(degrees)]
      for token in comp:
          if token != rep:
              token_to_token[token] = rep

  === Step 5: Dispatch with Condensation ===
  for each expert_idx:
      # 只发送 representative tokens
      tokens_to_send = filter(representatives, expert_routing)
      all_to_all_send(tokens_to_send, expert_owner[expert_idx])

  === Step 6: Expert Computation ===
  # 更少的 token → 更少的计算
  for expert_idx, tokens_received:
      expert_out = expert_ffn(tokens_received)

  === Step 7: Combine with Expansion ===
  for each token:
      if token in token_to_token:
          out[token] = expert_out[token_to_token[token]]
      else:
          out[token] = expert_out[token]
  # token similarity preserved after expert: ~95% pairs change < 0.2
  ```

  **与 Baseline 的通信模式对比**：
  - Vanilla: dispatch all-to-all + combine all-to-all = 2 × all-to-all × 全量 tokens
  - LUFFY: dispatch (condensed tokens) + combine (migrated sequences) = 显著减少的 all-to-all 流量
  - 核心差异: LUFFY 不移动 expert 参数（保持最大 expert parallelism），而是通过 sequence migration 改变 combine 的 token 路由目标 + token condensation 减少 dispatch 的 token 数量

## BrownoutServe SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 **Triton 语言的 MoE 算子重写 + PagedAttention GPU kernel 优化**：
  1. **MoE 算子 Triton 重写**：BrownoutServe 的 MoE 模块（含 BrownoutMoE 的 expert FFN 计算、token routing、united expert 调用）全部使用 Triton 语言重写，替代传统的 C++/CUDA 实现。Triton 与 PyTorch 无缝兼容，简化了开发和维护。
  2. **PagedAttention GPU kernel 优化**：相比 vLLM 的 PagedAttention 实现，BrownoutServe 将 block table 从 CPU 移至 GPU，block table 操作（查询、映射、更新）全部实现为 GPU kernel 函数。这充分利用了 GPU 并行计算能力，有效减少 PagedAttention 的额外开销。

  实验比较（间接体现在端到端吞吐量/延迟中）：BrownoutServe (with fused MoE) vs vLLM (native, with fused MoE) 在 ShareGPT 和 Alpaca 上的吞吐量比较，其中 BrownoutServe 在使用 fused MoE 后仍能提升 1.07×-1.32×（Fig. 9），部分来源于 kernel 级优化的贡献。

- 后端平台是什么，配置是什么。
  4× NVIDIA A100-PCIE-40GB GPU（每卡 40GB HBM），Intel Xeon Gold 6238 CPU。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch + Triton kernel 语言。论文使用 ShareGPT 和 Alpaca 数据集进行 10 分钟连续推理吞吐量评估，250s burst trace 进行 SLO violation 评估。

  **修改/优化内容**:
  1. **MoE 算子 Triton 重写**：将 BrownoutMoE 中的 expert FFN 计算（矩阵乘法、激活函数）、token dispatch/combine 操作、united expert 调用全部用 Triton 实现。Triton 代码在 Python 层面编写 tile-level 计算逻辑，编译为高效 GPU kernel，替代了传统 hand-written CUDA C++ kernel，降低了开发复杂度同时保持性能。
  2. **PagedAttention block table GPU 化**：原 vLLM 的 block table 管理在 CPU 端进行，每次 attention 计算需 CPU→GPU 数据传输。BrownoutServe 将 block table 直接置于 GPU 显存，block table 的查询（lookup KV cache block index）、映射（map logical→physical block）、更新（eviction/new block allocation）操作全部实现为 GPU kernel。这消除了 CPU-GPU 数据传输延迟，利用 GPU 大规模并行性加速 block table 操作。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。

  **开源**: https://github.com/beyondHJM/BrownoutServe（Apache-2.0）

  **Kernel 调度评估原理（以 1 次 MoE forward pass, Qwen1.5-MoE-A2.7B, 60 experts 为例）**：

  **1. BrownoutMoE kernel 全流程**：
  ```
  输入: token hidden states [batch_size, hidden_dim], shape=(64, 2048)

  Step 1 - Gate Kernel (Triton):
      for each token t (GPU thread block):
          # 计算 token 与所有 60 experts 的 affinity
          s[t, :] = x[t] @ E_centroids.T  # [64, 60]
          top2_idx[t], top2_score[t] = topk(softmax(s[t, :]), k=2)
      → 输出: routing_map (token→expert 映射), routing_weights

  Step 2 - Token Counting & Sorting (GPU):
      A = [(expert_id, token_count, hidden_states), ...]  # 60 个 expert
      sort A by token_count descending (GPU radix sort)
      T = total_tokens * threshold  # S1 阈值
      partition A → S1 (hot experts), S2 (cold experts)

  Step 3 - S1 Original Expert FFN (Triton fused MoE kernel):
      for each expert e in S1:
          # 将 e 对应的 tokens 合并为单 batch
          tokens_e = gather(tokens routed to expert e)
          # FFN: gate_proj → up_proj → activation → down_proj
          h = tokens_e @ W_gate  # gate projection
          u = tokens_e @ W_up    # up projection
          out = (silu(h) * u) @ W_down
          # scatter 回原位置
      → 使用 fused MoE 实现时，多 expert 计算合并为一次 sparse GEMM

  Step 4 - S2 United Expert FFN (Triton kernel):
      对 S2 中 experts 按 way=k 分组
      对每组 group:
          合并所有 group 内 tokens → concat_tokens
          united_expert_out = UE(concat_tokens)  # Triton FFN kernel
          scatter united_expert_out 回原 token 位置

  输出: 所有 token 经过 MoE 后的 hidden states [64, 2048]
  ```

  **2. PagedAttention GPU Block Table Kernel 原理**：
  ```
  原 vLLM (CPU block table):
      CPU 维护 block_table[t] = [physical_block_ids]
      每次 attention: CPU → GPU copy block_table → GPU kernel 查询
      ↑ 通信开销随 seq_len 增长

  BrownoutServe (GPU block table):
      GPU 显存中维护 block_table (torch tensor on device)
      GPU kernel 直接访问:
          for each query token q:
              # 同一 kernel 内完成 block table lookup + attention
              physical_blocks = block_table[request_id]  # GPU-side lookup
              K_cache = gather KV from physical_blocks
              V_cache = gather V from physical_blocks
              out[q] = flash_attention(Q[q], K_cache, V_cache)
      → 消除 CPU→GPU 数据传输，减少 kernel launch 次数
  ```


## Accelerating MoE Model Inference with Expert Sharding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 MoEShard 的 **expert computation kernel 融合优化**，包含两层：
  1. **Token Concatenation Fusion**: 将同一 expert 的来自所有 GPU 的 token concatenate 为单个大 tensor 进行矩阵乘法，将 kernel launch 数从 |E|×|G| 降至 |E|（独立于 GPU 数量）。
  2. **MegaBlocks Block-Sparse Matrix Multiplication**: 将所有 expert shard 计算融合为单次大规模稀疏矩阵乘法（variable-sized sparse MM，基于 Gale et al. MegaBlocks [19]），使 kernel launch 数独立于 expert 数量。

  实验比较（Ablation, Section 4.4）:
  - **Varying experts (8→256, batch=250)**: MoEShard w/ MegaBlocks vs MoEShard w/o MegaBlocks。expert<64 时 MegaBlocks kernel 创建开销导致性能略低；expert≥64 时 MegaBlocks 优势递增。
  - **Varying batch size (10→450, 128 experts)**: MegaBlocks 版在所有 batch size 下均优于无 MegaBlocks 版，因 128 experts 时 MegaBlocks 效率更高。

- 后端平台是什么，配置是什么。
  4× NVIDIA A100 GPU（每卡 80GB HBM），NVLink 互联（双向 600 GiB/s），同一节点。CUDA 12.6。CPU: AMD EPYC 7543 32-core。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch + CUDA 12.6，在 BookCorpus 数据集上执行 forward pass，100 iterations averaged per layer。
  
  **修改/优化内容**:
  1. **Per-Expert Token Concatenation**: 在 MoEShard 的 Step 4 expert computation 中，对每个 expert e，将 W[0][e], W[1][e], ..., W[|G|-1][e] concatenate 为单个 tensor，只执行一次矩阵乘法（而非 |G| 次小乘法），计算完成后拆分回 per-GPU result。伪代码：`tokens_concat = cat([W[g][e] for g in G]); result_concat = tokens_concat @ W_i_shard @ W_o_shard; split result_concat back to W[g][e]`。
  2. **MegaBlocks Sparse MM**: 将所有 expert 的 (tokens, W_i_shard, W_o_shard) 打包为 block-sparse 格式，调用 MegaBlocks 的 variable-sized sparse matrix multiplication kernel 一次完成全部计算。这利用了 block-sparse 数据结构将多个独立的小矩阵乘法合并为一个 GPU kernel 调用。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。

  **开源**: https://github.com/sacs-epfl/moe-inference (Python + PyTorch, 13 commits on main)。

  **Kernel 调度评估原理（以 4 GPU, 128 expert, batch=250, seq=120 为例）**:

  1. **Without Fusion (baseline kernel scheduling)**: Step 4 中双重循环 `for g in G: for e in E:` → 每 (g, e) pair 触发一次矩阵乘法 kernel launch → 共 4×128 = 512 kernel launches。每个 kernel 处理小批量 token（部分 expert 可能 0 token），大量 GPU SM 空闲。

  2. **Token Concatenation Fusion**: 外层 `for e in E`，内层将在所有 GPU 上目标为该 expert 的 token 全部 cat → 每个 expert 一次 kernel launch → 共 128 kernel launches（独立于 GPU 数）。每个 kernel 处理的 token 量增大，GPU 利用率提升。

  3. **MegaBlocks Sparse MM Fusion**: 将所有 128 个 expert shard 的 token 组织为 block-sparse 格式（每 expert shard 为一个 sparse block，token 数可变），一次 `cublasGemmEx` 风格调用完成 → 1 kernel launch。MegaBlocks 内部使用 custom CUDA kernel 遍历 non-zero blocks 并分派到 SM 执行。

  4. **性能输出**: 每个 forward pass 中 `torch.cuda.Event` 记录开始/结束时间 → 100 iterations 取 per-layer 平均 TTFT。对比 with/without MegaBlocks 的 TTFT 差异衡量 kernel fusion 收益。

## Accelerating Distributed MoE Training and Inference with Lina

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 Lina 的 **Communication Micro-Op Scheduler**：将 all-to-all 和 allreduce 通信张量分区（tensor partitioning）为固定大小的 micro-ops（如 30MB chunk），通过 priority queue 运行时调度，保证 all-to-all micro-op 优先获满带宽，allreduce micro-op 仅在无 all-to-all 待发送时发射。同时引入 **all-to-all pipelining**：将 all-to-all 也分区为 micro-ops，使 FFN 计算可在第一个 all-to-all micro-op 完成后立即启动（token 粒度计算），消除计算等待时间。配合 **Expert Packing**（每 device 打包多个 expert，2^n 递增）对齐 FFN 与 all-to-all micro-op 时间，最大化 pipeline efficiency。

  实验比较:
  - **设计消融**: Baseline → +Priority → +Tensor Partitioning → +Pipelining → +Fixed Scheduling，在 2/4/8/16-expert 配置下比较 step time speedup
  - **Partition Size 敏感性**: 从 10MB 到 100MB 在 16-expert 模型上比较 step time
  - **Expert Packing 效果**: 比较 w/o Packing vs w/ Packing 的 pipelining efficiency（Transformer-XL: 33%→86%, GPT-2: 36%→85%, BERT2GPT2: 34%→79%）
  - **Overhead**: tensor partition/concatenation overhead 平均 1.02% step time; micro-op 传输 overhead 平均 1.7% 额外时间

- 后端平台是什么，配置是什么。
  4 节点 × 4 NVIDIA A100 GPU (40GB HBM)，节点间 100Gbps InfiniBand。Training 使用与 expert 数量等量的 GPU (2/4/8/16)。NCCL 2.10 底层通信。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch Profiler 采集 CUDA kernel 执行时间和 GPU 活动；training metrics 在 10-step warm-up 后 averaged over 50 steps；inference 在 test set 上平均。
  
  **修改内容**:
  1. **Tensor Partitioning**: 使用 LibTorch `chunk` 和 `cat` API 沿 token 维度将 gradient/activation tensor 分割为固定大小（30MB default）的 micro-ops；避免跨 gradient 混合 chunk 以简化 concat
  2. **Priority Queue Scheduler**: 每 device 单线程维护 priority queue；all-to-all micro-op 优先级高于 allreduce；当 backward pass 进入 combine computation 阶段时暂停 allreduce micro-ops 发射（预示 all-to-all 即将到来）
  3. **Expert Packing Coordinator**: MoE 模型中嵌入单线程 controller，在 forward pass 记录 all-to-all 和 FFN micro-op 时间 → 每 10 steps 调整 packing → 需要时插入 one-time synchronous all-to-all 交换 expert params（下次 iteration 生效）→ multi-stream parallel execution（多 expert forward/backward）
  4. **All-to-all Pipelining**: 将 all-to-all dispatch 分区为 micro-op → 每个 micro-op 完成后立即启动对应 token subset 的 Expert FFN → combine 阶段同理

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源情况**: 论文基于 DeepSpeed MoE (https://github.com/microsoft/DeepSpeed) 和 PyTorch 构建，约 7500 LoC C++/Python 修改，但论文未提供独立的 Lina 开源仓库链接。

  **Communication Micro-Op 调度全过程（以 backward pass 中一个 MoE layer 为例）**:
  1. **输入**: 计算 stream (Stream a) 完成 expert FFN 反向计算后，梯度分别进入 expert-parallel 通信 (Stream b, all-to-all) 和 data-parallel 通信 (Stream c, allreduce)
  2. **Tensor Partition**: 15MB gradient tensor → `tensor.chunk(chunk_size=30MB)` → 5 个 30MB micro-ops 入队
  3. **Priority Queue 调度逻辑**（单线程 per device）:
     ```
     while queue not empty:
       if queue has all-to-all micro-op:
         pop and launch all-to-all micro-op (NCCL all-to-all)
         wait for completion
       else if queue has allreduce micro-op AND combine_computation not yet started:
         pop and launch allreduce micro-op (NCCL allreduce)
       else:
         idle (等待下一 micro-op 入队或 combine 阶段结束)
     ```
  4. **All-to-all Pipelining**: all-to-all dispatch 分 3 个 micro-ops → micro-op 1 完成后 1/3 tokens 进入 FFN → micro-op 2 完成 +1/3 → micro-op 3 完成 +1/3
  5. **Expert Packing 决策**: 记录 FFN micro-op time vs all-to-all micro-op time → 若 FFN << all-to-all → packing_factor *= 2 → 下次 iteration 生效
  6. **输出**: 所有 allreduce micro-ops 完成后 optimizer step

  **Baseline 对比**:
  - Baseline 中 Stream b (all-to-all) 和 Stream c (allreduce) 分别独立发射完整的大张量通信原语 → NCCL 底层无协调地公平共享 InfiniBand 带宽 → median all-to-all slowdown 1.83x (worst 4.14x)
  - Lina: micro-ops 使 all-to-all 与 allreduce 不并发 → all-to-all 获得满带宽 → all-to-all time speedup 2.21x~2.39x; step time speedup 1.37x~1.73x vs DeepSpeed



## A Survey on Mixture of Experts in Large Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本论文为综述，不提供原始实验。它在系统计算的 Section 5.1 中综述了面向 MoE 动态稀疏性的 GPU kernel 与编译器优化：
  - **MegaBlocks [137]**：将 MoE 计算重新表述为 block-sparse 操作，开发专用 block-sparse GPU kernel，在不丢弃 token 的前提下高效处理动态负载
  - **PIT [139]**：面向 MoE 动态稀疏性的深度学习编译器，利用 Permutation Invariant Transformation（数学可证性质）将多个稀疏微 tile 变换为 GPU 高效密集 tile，再执行 dense GEMM，不改变计算结果
  - **ScatterMoE [138]**：通过 ParallelLinear 模块执行分散组的并行线性运算，避免 scatter-to-group 数据拷贝，减少内存占用，且易扩展到 FFN 以外的模块（如 Attention experts）
  - **定制 GPU kernel**：DeepSpeed-MoE [64]、FastMoE [129]、HetuMoE [134]、Tutel [130] 均针对 MoE 特有的 gate routing/input encode/output decode 操作开发定制 kernel，消除冗余计算和内存搬运

- 后端平台是什么，配置是什么。
  GPU 后端：NVIDIA GPU（MegaBlocks, ScatterMoE, PIT, DeepSpeed-MoE, FastMoE, Tutel 均针对 GPU 平台）。

- 评估性能的软件/脚本是什么。修改了什么。
  - **MegaBlocks**：修改了 MoE 前向/反向 kernel（scatter → block-sparse GEMM），基于 block-sparse 矩阵乘法实现
  - **PIT**：修改了深度学习编译器（tiling 机制），插入 PIT 变换规则在 operator 级别生成优化 kernel
  - **ScatterMoE**：修改了 MoE 的 scatter-to-group 流程，用 ParallelLinear 的 grouped GEMM 直接操作分散的 token 组
  - **DeepSpeed-MoE/FastMoE/Tutel**：修改了 gate routing/encode/decode 的 GPU kernel 实现

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源链接**（Table 4）：
  - MegaBlocks: https://github.com/stanford-futuredata/megablocks
  - ScatterMoE: https://github.com/shawntan/scattermoe
  - PIT：论文未明确给出独立开源链接
  - DeepSpeed-MoE: https://github.com/microsoft/DeepSpeed
  - FastMoE: https://github.com/laekov/fastmoe
  - Tutel: https://github.com/microsoft/tutel

  **MegaBlocks Block-Sparse Kernel 执行流程**：
  1. **输入**：token 序列经 Router 得到 (token → expert) 映射（sparse matrix S ∈ {0,1}^{T×N}）
  2. **Block 化**：将 token-expert 映射矩阵 S 划分为固定大小的 block（如 128×128）
  3. **稀疏性编码**：只保留非空 block，形成 Block-Sparse 表示（CSR/CSC 变体）
  4. **Block-Sparse GEMM**：对非空 block 执行 batched dense GEMM（每 block 内部 dense 计算）
  5. **结果组装**：将各 block 的 GEMM 输出按原始 token 顺序组装为最终输出
  6. **输出**：每个 token 的 expert 计算输出（不丢弃任何 token）

  **PIT Compiler Kernel 执行流程**：
  1. **输入**：包含 MoE 层的模型计算图
  2. **PIT Tiling**：识别 MoE 操作中满足 Permutation Invariant Transformation 属性的 operator，将稀疏分散的 micro-tile 按固定 tile 大小重新排列为 dense tile
  3. **Kernel 生成**：对重组后的 dense tile 生成标准高效 GEMM kernel
  4. **结果逆变换**：将 dense tile 输出恢复为原始 token 顺序
  5. **输出**：与原始 MoE 计算等价的结果，但 GPU 利用率更高

  **ScatterMoE ParallelLinear 执行流程**：
  1. **输入**：token embeddings x 和 (token → expert) 映射
  2. **分组**：按 expert 将 token 分组（保持原顺序，无需 scatter-to-group 拷贝）
  3. **ParallelLinear**：对每组 token 直接执行 grouped GEMM（PyTorch-native tensor 操作）
  4. **组装**：将各 expert 输出按 token 原序拼接
  5. **输出**：中间表示保持为标准 PyTorch tensor，便于扩展至非 FFN expert

## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本论文是综述，不提供原始实验。它在硬件级别（Section 5）综述了以下kernel/运行时计算优化：
  - **FLAME**：首个在FPGA上全面利用MoE稀疏性的加速框架。参数级使用M:N剪枝减少不必要计算，expert级通过CEPR（Circular Expert Prediction）进行稀疏激活预测，使用双缓冲机制在计算前一个expert时加载预测的expert。
  - **M3ViT**：基于多任务场景中attention计算重排序的FPGA架构，只激活与当前任务相关的稀疏"expert"通路，实现任务间零开销切换。
  - **Edge-MoE**：首个端到端FPGA实现的多任务ViT，包括GELU函数近似计算、统一线性层模块实现硬件资源高效复用。
  - **MoE-CSP**：设计了处理4-bit/8-bit量化权重的专用CUDA kernel，执行浮点计算加速。
  - **QMoE**：实现了自定义压缩格式和定制GPU kernel用于1-bit on-the-fly计算。

- 后端平台是什么，配置是什么。
  - **FPGA平台**：Xilinx/Intel FPGA（FLAME、M3ViT、Edge-MoE使用的目标平台）
  - **GPU平台**：NVIDIA GPU（MoE-CSP、QMoE的CUDA kernel目标）
  - 论文未统一规定硬件配置

- 评估性能的软件/脚本是什么。修改了什么。
  - FLAME：修改了FPGA上的expert激活路径模式（circular expert prediction替代线性预测），实现了双缓冲加载机制
  - M3ViT：修改了attention计算顺序以支持多任务场景的稀疏expert激活
  - Edge-MoE：修改了GELU函数的FPGA实现（近似方法降低复杂度）和线性层模块（统一设计实现复用）
  - MoE-CSP：新增了处理4-bit/8-bit量化权重+浮点计算的CUDA kernel
  - QMoE：新增了1-bit压缩格式和相应的GPU反量化kernel

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源情况**：各硬件方法的开源情况各不相同，综述未统一收集各方法的开源链接。主要的开源加速框架（如DeepSpeed-MoE、Tutel）中有部分kernel优化开源。

  **FLAME FPGA Expert预测与双缓冲Kernel执行流程**：
  1. 输入：经过M:N剪枝后的expert权重矩阵W_pruned，当前token的expert激活历史
  2. CEPR预测：基于循环预测模式，改变expert激活路径的patterning，预测下一层所需expert集合E_pred
  3. 双缓冲加载：在计算当前expert E_curr的同时，通过第二个buffer预加载E_pred的权重
  4. Expert计算：FPGA上的DSP/查找表执行稀疏矩阵乘法
  5. 输出：当前token的expert输出y_i

  **MoE-CSP量化CUDA Kernel执行流程**：
  1. 输入：INT4/INT8量化的expert权重W_q，FP16的输入激活值x
  2. Kernel内反量化：w_deq = dequantize(W_q[i])，转换为FP16
  3. 浮点矩阵乘法：y_i = matmul(x, w_deq)
  4. 输出：FP16精度的expert输出

## APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  APTMoE 的**需求优先级调度策略（Demand-Priority Scheduling Strategy）**负责协调三层加载阶段产生的 CUDA 数据移动 kernel，解决它们对同一 PCIe 带宽的竞争和互相阻塞问题。核心实现包括：
  - **优先级队列调度**：三个加载阶段分别维护队列，按需求紧迫度分配优先级——inter-expert phase（最高）> inter-layer phase（中等）> inter-stage phase（最低）。程序通过 PriorityQueue 管理，周期性查询 GPU 加载状态后动态选择最高优先级的待加载 kernel 发起。
  - **CUDA Event 前探机制**：在加载 stream 的倒数第二个 action 前插入 `torch.cuda.Event()`，利用 `event.query()` 检测加载进度，当 event 触发时表示前一个数据移动 kernel 仍在执行，可在此时决定下一个加载 kernel，隐藏 kernel launch overhead。
  - **双 Stream 重叠**：维护 `comp_stream`（计算 stream）和 `load_stream`（加载 stream），通过 `torch.cuda.Event()` 建立 inter-stream dependency，确保加载完成后再触发对应计算 kernel。
  实验比较吞吐量（tokens/s）、不同设备拓扑下的 speedup、强扩展性（4→16 GPU）。

- 后端平台是什么，配置是什么。
  NVIDIA A800 GPU (40GB)，Intel Xeon Gold 6348 CPU (28核)，GPU间通过PCIe Switch通信，节点内存1024GB。软件栈：Ubuntu 22.04.3 + PyTorch 2.0.0+cu117。

- 评估性能的软件/脚本是什么。修改了什么。
  APTMoE 基于自研 pipeline 框架（APTMoE/Runtime/PipelineRuntime/pipeline_runtime.py）实现，基线包括 GPipe、GPipeOffload、Mobius。核心修改：
  - `comm_scheduler.py`：实现 PriorityQueue 管理三层加载队列，torch.cuda.Event.query() 检测加载状态，周期性查询+动态调度
  - `offload.py`：实现三层加载阶段的数据移动决策，通过添加/移除 block 名称到对应队列管理加载
  - `R_solver.py`：实现 Equation 1 的最优 CPU/GPU 分配方案求解
  - 使用 `psutil.Process().cpu_affinity()` 绑定不同数量的 CPU 核心到特定进程以设置不同设备拓扑

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源**：https://github.com/Atopos-309/APTMoE

  **执行命令示例**：
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 ./main.py \
    --is_moe=True --num_training_steps=50 --model_config=S \
    --num_experts=16 --gini=0.3 --topo=C1+G2 --pipeline=APTMoE
  ```

  **Demand-Priority Kernel 调度执行流程**：
  1. **输入**：三层加载队列（inter-stage queue / inter-layer queue / inter-expert queue）中各包含待加载的 model block 名称
  2. **优先级决策**：调度器检查三层队列，按 inter-expert > inter-layer > inter-stage 优先级从非空队列中选择下一个加载目标
  3. **CUDA Event 插入**：在 load_stream 的倒数第二个数据移动 kernel 前插入 cuda_event，event 触发时发起新 kernel，隐藏 launch latency
  4. **Kernel 发起**：将选定的 model block 数据移动 kernel（host→device cudaMemcpy）提交到 load_stream
  5. **Inter-Stream 同步**：每个 model block 关联一个 torch.cuda.Event()，load_stream 完成数据移动后 record event，comp_stream 的对应计算 kernel 通过 stream wait 等待该 event
  6. **计算执行**：comp_stream 执行被加载 block 的 forward/backward 计算（MHA、gate、expert 等），与 load_stream 的下一次加载并行
  7. **输出**：每 iteration 的吞吐量（tokens/s），Step 结束后报告整体吞吐

  **调度关键点**：由于中断和恢复 CUDA kernel 执行极其困难且昂贵，APTMoE 选择在 kernel **启动前**而非执行中进行调度决策。通过 event.query() 的主动轮询机制，scheduler 可以在前一个加载 kernel 仍在执行时决定下一个加载内容，确保 PCIe 带宽得到最大化利用。

## Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 SYMI 为支持 per-iteration adaptive expert replication 引入的四项核心 collective communication / runtime scheduling 机制：
  
  1. **Intra+Inter Rank All-Reduce**：支持同一 rank 内多个 slots 持有同一 expert class 的 replica 时的梯度同步。分为三步：(a) 每个 rank 内选举一个 slot representative，其他 slot 将 gradient 加到 representative；(b) inter-rank all-reduce 仅在各 rank 的 representative 间执行；(c) representative 将 all-reduced gradients 广播回同 rank 的其他 slot。这使 expert 可以自由地分配在任意 slot，支持 intra-rank 和 inter-rank 同时 expert data parallelism，避免传统 NCCL all-reduce 的跨 rank 限制导致 up to 20% extra token drops。
  
  2. **Communication Group Pre-Registration**：由于 SYMI 每 iteration 的 expert placement 变化，NCCL 通信组也会变化。若每 iteration 动态创建通信组，在大集群（如 N=2048）中单次 NCCL group creation 耗时可能超过 1000s。SYMI 在初始化时预注册所有需要的 contiguous rank 通信组（仅需 N(N-1)/2 个），跨 expert 和 layer 复用，训练期间零 group creation overhead。
  
  3. **Gradient Collection Load-Balancing (Algorithm 2)**：SYMI Optimizer 为每个 (expert_class, optimizer_partition_node) 对选择一个 source rank 来发送 gradient shard。`get_source()` 优先选择本地 expert→optimizer 传输（零网络开销），远程传输则 round-robin 分配以避免 hotspot。最终通过 batch_isend_irecv 完成所有 expert 的梯度收集。
  
  4. **Expert Placement Materialization via Batch P2P**：SYMI Optimizer 计算 updated weights 后，通过 batch point-to-point communication 将 weights 发送到新 placement 对应的 expert slot。不引入额外数据搬运：发送到同一 slot 的数据量相同（无论 expert class 是否改变），因为每个 slot 始终接收一个完整的 expert weight。
  
  实验比较：
  - **Latency Breakdown**: 对比 SYMI vs DeepSpeed vs FlexMoE 各 rebalancing 频率下，training iteration 各阶段（FWD all-to-all, FWD compute, BWD compute, BWD all-reduce, Optimizer step, SYMI overhead）的耗时
  - **Communication Overhead**: SYMI 新增组件（popularity all-reduce + expert placement scheduler + metadata update）占总 iteration time 比例：1.06% (125M), 0.82% (350M), 0.70% (760M)
  - **FlexMoE Rebalancing Cost**: FlexMoE rebalancing iteration latency 为正常的 2.46x-4.10x

- 后端平台是什么，配置是什么。
  Azure 集群 16 × NC24ads-v4 instances，每 instance: NVIDIA A100 80GB GPU, PCIe 4.0 32 GB/s, 100Gbps ConnectX-5 NIC。底层通信库：NCCL (PyTorch distributed)。SYMI 基于 DeepSpeed 实现，optimizer offload 至 CPU host memory (ZeRO-1 风格)。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch Profiler 采集各阶段 latency breakdown；training loss logging per iteration；token survival rate 统计。
  
  **SYMI 对 DeepSpeed 的修改**:
  1. **Router Extension**: 在 MoE router 后添加 all-reduce collective 聚合 global expert popularity（metadata tensor 仅 E × 4 bytes，开销可忽略）
  2. **Intra+Inter Rank All-Reduce**: 修改 gradient synchronization 逻辑，支持 intra-rank 梯度累加 + inter-rank all-reduce (representative only) + intra-rank broadcast 三步流程
  3. **Communication Group Manager**: 预注册 contiguous-rank NCCL groups，训练期间通过查表获取所需 group 而非动态创建
  4. **SYMI Optimizer**: 替换原有 ZeRO-1 optimizer，实现 static uniform partitioning across ALL nodes（而非仅 expert 所在节点），gradient collection (Algorithm 2) 和 weight distribution (batch_isend_irecv)
  5. **Layer Metadata Store**: per-layer per-rank 存储 global expert popularity 数组，供 Expert Placement Scheduler 读取
  6. **Expert Placement Scheduler**: 实现 Algorithm 1（proportional allocation + rounding correction + contiguous assignment），每 iteration 计算新 placement

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源情况**: 论文未公开独立开源仓库。SYMI 基于 DeepSpeed (https://github.com/microsoft/DeepSpeed) 实现，但截至搜索时未找到公开代码链接（arXiv 2504.19925v2）。

  **SYMI Collective Communication 全流程（以一次 Optimizer Step 的 Grad/Weight Communication Phase 为评估原理）**:

  ```
  === 输入状态 ===
  - placement[t]: 当前 iteration 的 expert-to-slot mapping (长度 s*N 的数组)
  - gradients[t]: 各 slot 的 expert gradient (size G per expert instance)
  - optimizer_state: 均匀分片在 N 个节点 host memory 的 static optimizer partitions
  
  === Phase 1: Gradient Communication (collect grads to optimizer) ===
  1. Expert Gradient Sync (All-Reduce, within backward pass):
     输入: per-slot raw expert gradients [G each]
     处理:
       a. Intra-rank: for each expert_class, collect local slot grads → 
          rep_grad[rep_slot] = Σ grad[local_slots]
       b. Inter-rank: allreduce rep_grad across representative ranks → 
          synced_grad
       c. Intra-rank broadcast: grad[other_slots] = synced_grad / num_local_replicas
     输出: synchronized per-slot expert gradients
     性能: 相比传统 all-reduce 减少 inter-node traffic（同 expert replicas 优先同 rank）
     
  2. Gradient Collection by SYMI Optimizer:
     输入: synced gradients, optimizer partition layout
     处理:
       for each (expert_id, node_id) pair:
         src = get_source(expert_id, node_id)  
         # prefers local if expert_id maps to node_id's slot
         # else round-robin from remote candidates
         irecv(grad_shard[expert_id][node_id], from=src)
     输出: 每个 node 的 optimizer 收到所有 assigned experts 的 gradient shards
     通信模式: batch point-to-point (NCCL isend/irecv)
     评估指标: T_G^SYMI = (E/N)*(G/BW_pci) + ((sN-s)/N)*(G/BW_net)
  
  === Phase 2: Optimizer Update (local) ===
  3. Adam Step:
     输入: grad_shard[expert_id], optimizer_state[expert_id] (fp32 param, momentum, variance)
     处理: Adam update → updated_weight_shard[expert_id] = param - lr * (m_hat / (sqrt(v_hat) + eps))
     输出: updated fp16 weight shards
     性能: 纯 local 计算，无通信
  
  === Phase 3: Weight Communication (materialize NEW placement) ===
  4. Expert Placement Scheduling (for iteration t+1):
     输入: global_popularity[t] (from forward pass all-reduce, [E] array)
     处理: Algorithm 1
       goal = (popularity / sum(popularity)) * N * s
       exp_counts = clamp(floor(goal), min=1)
       rounding correction to sum = N*s
       contiguous assigment → placement[t+1]
     输出: placement[t+1] (长度 s*N 的 expert class ID 数组)
     开销: 纯 local 计算，< 0.1% iteration time
     
  5. Weight Distribution:
     输入: updated_weight_shards (distributed across N nodes), placement[t+1]
     处理:
       for slot in all_slots:
         expert_id = placement[t+1][slot]
         target_rank = slot_to_rank(slot)
         # collect all N shards for this expert's weight
         # identical data volume regardless of whether expert_id changed!
         if previous_expert[slot] != expert_id:
           # slot receives different expert's weights - but same size W!
           send(weight[expert_id], to=target_rank)
         else:
           send(weight[expert_id], to=target_rank)  # same as above!
     输出: 每个 GPU slot 获得下一 iteration 的 expert weights
     通信模式: batch point-to-point for all experts (NCCL isend/irecv)
     关键不变性: D_W^SYMI = s*N*W = D_W^static（通信量完全相等！）
     额外开销: 仅 locality shift → ΔT/T ≈ 1.52% extra cost (N=2048, E=64, s=2)
  ```

  **SYMI vs Baseline 通信量等价性证明**:
  
  对于 Grad Communication Phase:
  - Static: D_G^static = E*r * G/r * r = r*E*G = s*N*G
  - SYMI: D_G^SYMI = Σ r_i * G/N * N = s*N*G
  
  对于 Weight Communication Phase:
  - Static: D_W^static = E*r * W/r * r = r*E*W = s*N*W
  - SYMI: D_W^SYMI = Σ N * W/N * r_i = s*N*W
  
  SYMI 不引入任何额外数据搬运量。略微增加的通信成本仅来自 expert-optimizer locality 变化（expert 与 optimizer partition 不再总是同 node），在论文代表性配置（N=2048, E=64, s=2）下仅增加约 1.52% 通信时间。

## Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 Comet 的 **fine-grained fused MoE kernel**，核心包含三项 kernel 级优化：
  1. **Shared Tensor Based Dependency Resolving**：将 MoE layer0 的 shared tensor 沿 M（token）维度分解，将 layer1 的 shared tensor 沿 N（hidden）维度分解，并重新调度 GroupGEMM 的 tile 计算顺序——layer0 按 remote token 依赖最小化排序（local token 优先计算），layer1 按列方向交错计算（T^N 列完成后立即启动 reduce+通信）。这打破了通信(token级)和计算(tile级)之间的粒度不匹配。
  2. **Thread Block Specialization**：将通信（NVSHMEM I/O）和计算（CUTLASS GEMM）隔离到独立的 thread block，而非将它们垂直融合在同一 thread block 中。GEMM thread block 使用标准的 CUTLASS Hopper 实现（producer warp 用 TMA async 加载，consumer warp 执行 MMA），通信 thread block 独立执行 token 的 top-K reduce 和远程 NVSHMEM 读写。避免 fine-grained I/O 干扰计算 pipeline（尤其是 Hopper TMA 异步流水线）。
  3. **Adaptive Thread Block Assignment**：预编译多个不同 n^c/n^p（通信/计算 thread block 比例）的 kernel 变体，离线 profile 各配置下的最优分割点并存储为 metadata。运行时根据输入 token 长度 M 和并行策略（TP×EP）从 metadata 选择最优 kernel。最优 n^c 随 M 增大而增大（如 TP=8 时 M=4096→n^c=18, M=16384→n^c=26），随 TP 减小而增大（M=16384 时 TP=8→n^c=26, TP=4→n^c=46）。

  实验比较：
  - **Single MoE layer duration** (Figure 10): Comet vs Megatron-Cutlass, Megatron-TE, FasterMoE, Tutel，EP=8，M 从 256 到 16384 → Comet 1.28×-2.37× speedup，小 M 时优势更显著（kernel 内调度消除了 host 端 kernel launch 开销）
  - **MoE layer time breakdown** (Figure 11): EP=8, TP=1, E=8, topk=2, M=16384 → Comet hides 86.5% communication latency vs FasterMoE 29.2% and Tutel 68.6%
  - **Various parallelism** (Figure 12): E=8, topk=2, M=8192, EP×TP=8，不同 TP/EP 组合 → Comet 在所有配置下保持低延迟（shared tensor reschedule 消除 TP 引入的 fragmented GEMM 问题）
  - **Varying E and topk** (Figure 13): Comet 1.16×-1.83× speedup vs baselines
  - **Imbalanced token distribution** (Figure 14 left): std 从 0 到 0.05 → Comet 在所有分布下 consistently outperform
  - **L20 cluster (PCIe, 25 GB/s)** (Figure 14 right): Comet 1.19×-1.46× speedup vs baselines

- 后端平台是什么，配置是什么。
  **H800 集群**: 8× NVIDIA H800 GPU (80GB HBM)，NVLink 互联。CUDA 12.3, NVSHMEM 2.11, PyTorch 2.4.0, Megatron-LM (git-hash 6dbe4c)。
  **L20 集群**: 8× NVIDIA L20 GPU (46GB)，PCIe 桥互联，GPU-to-GPU 带宽约 25 GB/s。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: Megatron-LM 框架上的端到端 MoE 模型推理/训练。使用 PyTorch Profiler 采集时间 breakdown。模型: Mixtral 8x7B (E=8, N=4096, K=14336), Qwen2-MoE-2.7B (E=64, N=2048, K=1408), Phi-3.5-MoE (E=16, N=4096, K=6400)。

  **Comet 修改内容（~12k lines C++/CUDA + 2k lines Python）**:
  1. **CUTLASS GEMM kernel 优化**: 在 layer0 中将 GEMM 输入的 row indices 缓存到寄存器，减少 global memory 访问。利用 CUTLASS 编程模板生成高效 GroupGEMM kernel。
  2. **Shared Tensor Decomposition & Reschedule**: 
     - Layer0 (communication→computation pipeline): 将 shared tensor [M×topk, N] 沿 M 维度分解为 sub-tensors。Token 按 source rank 排序 → GroupGEMM tile 计算顺序设计为 local token tile 优先 → 在 remote token 传输期间 local tokens 已开始计算。
     - Layer1 (computation→communication pipeline): 将 shared tensor 沿 N 维度分解。GroupGEMM 改为 column-wise 执行（先计算所有 expert 的前 T^N 列 → 启动 reduce+通信 → 继续后续列），而非 sequential per-expert 执行。
  3. **NVSHMEM Fused Kernel**: 使用 NVSHMEM 的 Unified Virtual Address 实现 token 级 fine-grained remote I/O，替代 NCCL 的粗粒度 all-to-all。NVSHMEM buffer 大小 = M×N（BF16/FP16 时 2MN），shared across layers and experts。
  4. **Thread Block Specialized Kernel**: 在 Hopper 架构上，GEMM thread blocks 使用 CUTLASS 标准实现（producer warp TMA async load → shared memory → consumer warp MMA），通信 thread blocks 读取 GEMM 输出 → top-K reduce → NVSHMEM write/read。两套 thread blocks 由 GPU hardware scheduler 并发调度在同一 kernel 内。
  5. **Adaptive Assignment Metadata**: 预编译多个 (n^c, n^p) 组合的 kernel 变体，profile 后存储为 metadata，运行时查表选择。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源**: https://github.com/bytedance/flux（Project Page）。Comet 将开源，代码基于 Megatron-LM + CUTLASS + NVSHMEM。

  **Comet Fused Kernel 全流程（以 H800, EP=8, TP=1, E=8, topk=2, M=16384, 一个 MoE layer 为例）**:

  **=== MoE Layer0 (Communication→Computation Pipeline) ===**
  
  1. **输入**: 每个 rank 持有 M/W = 2048 tokens，经 Router 计算后需 dispatch 到持有对应 expert 的 rank
  
  2. **Shared Tensor Decomposition (沿 M 维)**:
     ```
     shared_tensor shape: [M×topk, N] = [32768, N]
     被分解为 M×topk = 32768 行（token 粒度）
     → tokens 按 source_rank 排序: local tokens 在前，remote tokens 在后
     ```
  
  3. **NVSHMEM Fine-grained Receive**: 通信 thread blocks 通过 NVSHMEM `nvshmem_getmem` 从 remote rank 逐 token 拉取数据到 shared tensor buffer。每个 token 独立可读——不等待全部 all-to-all 完成。
  
  4. **GroupGEMM Tile 重调度**:
     ```
     GroupGEMM 的 tile 计算顺序重新设计:
     先计算仅含 local tokens 的 tiles（无数据依赖，立即开始）
     → 再计算含部分 remote tokens 的 tiles（remote tokens 已通过 NVSHMEM 到达）
     → 最后计算纯 remote tokens 的 tiles
     在 GroupGEMM 执行早期 tiles 的同时，更多的 remote tokens 在并行到达
     ```
  
  5. **Thread Block Specialization (SM 分配)**:
     ```
     Total SMs = 132 (H800)
     通信 thread blocks (n^c): 执行 NVSHMEM remote read + token scatter
     计算 thread blocks (n^p): 执行 CUTLASS GroupGEMM
     # n^c/n^p 比例由 adaptive assignment metadata 决定
     
     GEMM TB (CUTLASS Hopper):
       Warp 0 (producer): TMA async copy global→shared memory
       Warp 1 (consumer): MMA (tensor core) shared→register→accumulator
     
     通信 TB:
       Warp 0..N: NVSHMEM get + scatter tokens to shared tensor buffer
     ```
  
  6. **输出**: 所有 expert 的 GEMM 完成 → layer0 output → 进入 layer1

  **=== MoE Layer1 (Computation→Communication Pipeline) ===**

  7. **Shared Tensor Decomposition (沿 N 维)**:
     ```
     shared_tensor shape: [M×topk, N] = [32768, N]
     被分解为 N/T^N 个列块（T^N 为 GroupGEMM tile N 维度大小）
     ```
  
  8. **Column-wise GroupGEMM + 通信重叠**:
     ```
     for col_block in [0, N/T^N):
       # 所有 expert 并行计算第 col_block 块
       for each expert on this rank:
         GEMM_tile_compute(expert, col_block)  # partial result along N dim
       
       # T^N 列完成后立即启动 reduce + 通信
       topk_reduce(partial_results[:, :col_block * T^N])
       NVSHMEM write tokens back to source ranks
     
     # 传统方案: 每个 expert 全部列计算完 → 才开始 reduce+通信
     # Comet: 每 T^N 列计算完 → 立即 reduce+通信 → 与剩余列计算重叠
     ```
  
  9. **Adaptive Assignment**: kernel 启动时，从 metadata 查表选择 (n^c, n^p) → 设定 thread block 数量 → 多 SM 并发执行
  
  10. **性能输出**: total MoE layer duration 记录为从 Router 开始到 combine 完成的时间。Comet hides 86.5% communication (Figure 11)，单层 1.96× speedup vs Megatron-Cutlass (Figure 10)。

  **与 Baseline 的关键差异**:
  - Megatron-Cutlass/TE: 零通信-计算重叠（通信完成→计算开始→通信开始，顺序执行）
  - FasterMoE: 将 expert 计算分 2 个 chunk 做 pipeline overlap（coarse-grained），仅 hide 29.2% 通信
  - Tutel: 通过优化 all-to-all primitive 和自适应并行 partial overlap，hide 68.6% 通信，但 expert 多时 scheduling overhead 增大
  - Comet: fine-grained token 级 fused kernel overlap，hide 86.5% 通信，且 kernel 内调度消除 host 端 overhead


## Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 **CPU Chunked Attention Verification Kernel**——专为 speculative decoding 在 offloading 场景下设计的 CPU 端 chunked attention 算子，以及 GPU-CPU 异构 kernel 调度流水线。核心内容包括：
  1. **CPU Chunked Attention Kernel**：处理 speculative decoding verification 中的 Q∈R^{n×d}, K∈R^{(l+n)×d}, V∈R^{(l+n)×d} 的 attention 计算，其中 n 为 draft tokens 数，l 为历史 tokens 数。针对 n>1（chunked）而非 n=1（单 token decode）优化。
  2. **Intel MKL 加速**：利用 Intel oneAPI Math Kernel Library 进行高效矩阵乘法，充分利用 CPU SIMD 和 MIMD 能力（区别于 GPU 的 SIMT 架构 + 手动 managed shared memory）。
  3. **Mask 内存压缩**：attention mask M∈{-∞,1}^{n×(l+n)} 中 draft-to-prefix 部分固定为 1，仅存储 draft-token 间相关部分（M 的右下角 n×n 子区域），大幅减少内存占用。

  实验比较：
  - 间接体现在端到端系统性能中——CPU Attention 在 iteration time 中占主导（Table 3: 4.29s actual vs 3.88s estimated）
  - Iteration breakdown (Figure 13): 随 draft length 增加 CPU Attention 时间占比增长，逐渐成为 target model 的瓶颈
  - Profiling estimator 精度: CPU Attention estimation error 10.6%

- 后端平台是什么，配置是什么。
  A30 环境: Intel Xeon Gold 6426Y CPU (2.43 TFLOPS, 357 GB/s memory bandwidth)。4090D 环境: Intel Xeon Gold 5418Y CPU (1.45 TFLOPS, 197 GB/s)。

- 评估性能的软件/脚本是什么。修改了什么。
  评估工具: PyTorch + Intel MKL。使用 APPS 和 CNN/DailyMail 数据集测量不同配置下各算子的实际执行时间（Table 3: Actual vs Estimated 对比）。

  **修改/优化内容**:
  1. **CPU Chunked Attention 实现**（替代 GPU chunked attention 和 naive CPU decode attention）：
     - GPU chunked attention 的替代方案：避免将 KV cache 从 CPU DRAM 反复传输到 GPU HBM
     - Naive CPU decode attention (n=1) 的替代方案：避免对每个 draft token 单独执行 attention（减少重复 KV cache 访问）
     - PyTorch CPU prefill attention 的替代方案：避免对已在 KV cache 中的 token 重复计算

  2. **Intel MKL GEMM 加速**: chunked attention 的计算核心为两个矩阵乘法——Q@K^T (attention scores) 和 softmax(scores)@V (weighted sum)——均通过 Intel MKL 的优化 GEMM 实现

  3. **Mask 压缩存储**：传统方案存储完整 n×(l+n) mask 矩阵，其中 l 远大于 n（如 l=512, n=5 时 mask 为 5×517）；SpecMoEOff 只存储 draft-to-draft 子区域 (n×n) 和 draft-to-prefix 的固定值（全 1），节省约 (l+n)^2 - n^2 的内存

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未公开独立开源仓库。CPU Chunked Attention kernel 基于 Intel MKL 实现于 SpecMoEOff 系统内部。

  **CPU Chunked Attention Kernel 全过程（以单 request, n=5 draft tokens, l=512 prefix tokens, d=4096 为例）**：

  1. **输入准备**：
     - Q ∈ R^{5×4096}: 5 个 draft tokens 通过 target model Q projection 得到
     - K ∈ R^{517×4096}: 512 prefix + 5 draft tokens 的 key，从 CPU DRAM KV cache 读取
     - V ∈ R^{517×4096}: 同上 value
     - M_draft ∈ {0,1}^{5×5}: 仅存储 draft tokens 间的 causal mask（下三角=1）
     - M_prefix: draft→prefix 全 1（不实际存储，计算时直接忽略 mask 加项）

  2. **Step 1 - Attention Scores (Intel MKL SGEMM)**：
     ```
     # Q@K^T: [5, 4096] @ [4096, 517] → [5, 517]
     scores_full = mkl_sgemm(Q, K.T) / sqrt(4096)  # [5, 517]
     # scores_full[:, :512]: Q vs prefix K (无 mask, 全 1)
     # scores_full[:, 512:]: Q vs draft K (需 causal mask)
     ```

  3. **Step 2 - Mask Application（仅处理 draft 部分）**：
     ```
     scores_full[:, 512:] += causal_mask(M_draft)  # M_draft 下三角=0, 上三角=-inf
     # prefix 部分无需 mask（所有 draft token 都可 attend 到所有 prefix token）
     ```

  4. **Step 3 - Softmax + Weighted Sum (Intel MKL SGEMM)**：
     ```
     attn_weights = softmax(scores_full, dim=-1)    # [5, 517]
     # attn_weights @ V: [5, 517] @ [517, 4096] → [5, 4096]
     output = mkl_sgemm(attn_weights, V)            # [5, 4096]
     ```

  5. **Batch 扩展**：对 b 个 requests，每个 request 的上述计算并行化（multi-threaded CPU parallelism），各 request 独立执行

  **与 Baseline 的对比**：

  | Kernel 方案 | Q@K^T 次数 | KV Cache 访问 | CPU-GPU 传输 | Mask 内存 |
  |------------|-----------|-------------|-------------|----------|
  | GPU chunked attention | 1 次 (b×n size) | ~0 (KV在GPU) | 需传输全部KV cache | 完整 n×(l+n) |
  | Naive CPU decode (repeat n times) | n 次 (b×1) | n× 重复读取 | 0 | 无 (per-token) |
  | PyTorch CPU prefill attention | 1 次 (全量) | 全部重读 + 重复计算 | 0 | 完整 n×(l+n) |
  | **SpecMoEOff CPU Chunked** | **1 次** | **1 次，无重复** | **0** | **仅 n×n** |

  核心优势：1 次 Q@K^T 覆盖全部 draft tokens，无 KV cache 重复访问，无 CPU-GPU 传输，mask 内存 O(n²) 而非 O(n·(l+n))。

  **GPU-CPU Kernel 调度流水线**：
  - GPU stream 1 (comp): GPU Other1 kernel → CPU Attention trigger → GPU Other2 kernel → GPU MoE kernel
  - GPU stream 2 (load): 异步 HtoD transfer 下一层 expert weights
  - CUDA Event synchronization: CPU Attention 完成 → record event → GPU Other2 等待 event
  - 两个 micro-batch 的 CPU Attention 和 GPU MoE 交错执行实现重叠

## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文使用 SimpleMoE (Tan et al. 2024) 的 ParallelLinear 操作实现 MLP 层的稀疏推理（top-K expert 选择和执行），使用 torch.nn (PyTorch) 实现 Attention 层的密集推理。核心运行时决策：(1) MLP 层使用 sparse inference kernel（ParallelLinear），因为 MLP 层 sparsity 高（激活比例 <30-40%），sparse kernel 在计算量减少上的收益超过动态路由开销；(2) Attention 层使用 dense inference kernel（torch.nn），因为 Attention 层 sparsity 低（激活比例 >60%），sparse kernel 的动态路由开销反而导致更慢。实验比较不同 expert sampling 策略（Threshold / TopK / Threshold-TopK）对 WikiText PPL 和 active param count 的 trade-off。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB GPU 和 H100-80GB GPU。训练使用 H100-80GB。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 SimpleMoE (Tan et al. 2024, arXiv:2403.08245) 的 ParallelLinear 操作进行稀疏推理。使用 torch.nn (PyTorch) 进行 dense inference。论文未修改 kernel 实现本身，而是做出 runtime 层面的调度决策——通过观察 Figure 5 发现 Attention 层 sparsity > 60% 时 sparse inference 反而比 dense inference 慢（因动态路由的中间 token 复制和 expert 输出聚合 overhead），因此采用混合策略：MLP 用 sparse，Attention 用 dense。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文代码未开源。SimpleMoE (Tan et al. 2024) 是开源项目 [ScatterMoE](https://github.com/shawntan/scattermoe)，提供了 ParallelLinear 的原生 PyTorch 实现。

  **Kernel 层面的输入-性能-输出流程**（以 DS-MoE-6B MLP 层, D_emb=4096, N_ffd=32, D_ffd=512, TopK=4 为例）：

  1. **Token 路由**：
     ```
     X: [B, 4096]                # input hidden states batch
     S = Softmax(X @ W_r)        # W_r: [4096, 32], Router 计算
     top_idx: [B, 4]             # 每 token 选 top-4 experts
     ```
  
  2. **ParallelLinear Sparse Forward（SimpleMoE）**：
     ```
     # 等价于对每个 selected expert 执行 matmul
     # 使用 torch.index_select 或 bmm 实现 batched expert computation
     for each expert e in top_idx:
       # 收集分配给 expert e 的 tokens
       X_e = gather tokens assigned to expert e    # [T_e, 4096]
       # 执行 FFN: W_up [4096, 512], W_down [512, 4096]
       H_e = GeLU(X_e @ W_up_e)                     # [T_e, 512]
       O_e = H_e @ W_down_e                         # [T_e, 4096]
       # 按 Router score 缩放
       O_e = S[token_to_e, e].unsqueeze(-1) * O_e
     # Scatter 回原 token 顺序
     O = scatter sum of all O_e back to [B, 4096]
     ```
     Note: ParallelLinear 内部通过 `torch.nn.functional.linear` 的 group 机制批量处理 experts，避免逐 expert Python loop 开销。

  3. **性能对比的 Kernel 层面原理**：
     - **Dense FFN**：`O = GeLU(X @ W_up) @ W_down` where W_up: [4096, 16384] (32×512), W_down: [16384, 4096]。计算量：2 × B × 4096 × 16384 = ~134M FLOPs/token。
     - **Sparse FFN (TopK=4)**：4 个 expert, W_up_e: [4096, 512], W_down_e: [512, 4096]。计算量：4 × 2 × B × 4096 × 512 = ~16.8M FLOPs/token。计算量减少 8×。
     - **Sparse Overhead**：token routing（gather/scatter）+ Router 计算。当 sparsity 高（MLP, ~70% tokens 无需计算）时，overhead << 节省的 FLOPs。但当 sparsity 低（Attention, ~30%）时，dynamic routing overhead 可能 > dense 节省。
  
  4. **混合策略的 Kernel 调度依据**（Figure 5, DS-MoE-3B ε=0.48）：
     - MLP 层：平均 active experts ~6-8 / 32，sparsity ~75-80%。使用 ParallelLinear sparse。
     - Attention 层：平均 active experts >5 / 8（60%+），且 attention 的 KV cache 计算已为 computation-heavy。使用 torch.nn dense（torch.nn.functional.scaled_dot_product_attention）。
     - 论文发现：Attention 层在 sparsity >60% 时 sparse inference 因 dynamic routing overhead 而比 dense 更慢。

  5. **Expert Sampling 策略对 Kernel 路由效率的影响**（Figure 4, DS-MoE-3B）：
     - Threshold：per-token 独立决定激活 expert 数（不同 token 可能不同 K），batch inference 时有 padding 浪费。
     - TopK：固定 K 值，统一 batch 内所有 token 的 expert 数，GPU 利用率高。
     - Threshold-TopK：先统计 batch 内平均激活 expert 数，用统一 K 值。兼顾自适应和 batch 效率。
     - 实验结论：Threshold 在 PPL/效率 trade-off 最优，但 TopK 和 Threshold-TopK 更适合实际部署。

## Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  为 DES 的 coreset selection 开发自定义 fused GPU kernel，将原有的 12 个碎片化算子（softmax、top-k、reduction 等）融合为仅 2 个 kernel。实验比较 fused kernel vs PyTorch baseline 的 coreset selection 延迟，验证消除 kernel launch 和 HBM traffic 开销的效果。

- 后端平台是什么，配置是什么。
  NVIDIA B200 GPU，CUDA 13.1，Intel Xeon 6960P CPU。

- 评估性能的软件/脚本是什么。修改了什么。
  NVIDIA Nsight Systems 进行 kernel profiling。自定义 CUDA kernel 替换 PyTorch 的碎片化算子链。
  Kernel 设计：
  - **Primary kernel**：融合 per-token softmax + Top-K filtering + weighted expert accumulation，使用 register-level computation 和 atomic instructions 更新 global saliency scores。
  - **Second kernel**：基于 threshold-governed ranking 执行 final expert masking。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供开源代码。自定义 fused kernel 的原理：
  ```
  输入: Router logits I (N×M tensor), Top-K, M_core
  PyTorch baseline 流程（12 kernels）:
    1. Softmax(I) → gate weights
    2. TopK(gate weights, K) → local selections
    3. Mask 非 Top-K 权重
    4. Sum over N dim → votes
    5. TopK(votes, M_core) → coreset
    6-12. ... 后续路由和计算
  
  Fused kernel 流程（2 kernels）:
    Kernel 1 (per-token fused):
      for each token n in parallel:
        softmax_n ← softmax(I_n)                 // register-level
        topk_idx_n, topk_val_n ← topk(softmax_n, K)  // register-level
        masked_weights_n ← mask_non_topk(softmax_n, topk_idx_n)
        for each expert i in topk_idx_n:
          atomicAdd(V[i], masked_weights_n[i])   // atomic to global memory
    Kernel 2 (final ranking):
      C ← topk(V, M_core)                        // select coreset
      output_mask ← create_mask(C, M)            // generate mask
  ```
  Fused kernel 实现 **6× speedup** over PyTorch baseline，通过消除冗余 HBM traffic 和 operator dispatch overhead。

## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 METRO 的 CUDA kernel，将 Algorithm 1（greedy expert-to-GPU assignment）实现为运行在单个 Streaming Multiprocessor (SM) 上的 GPU kernel。设计要点：(1) **单 SM 绑定**：由于算法并行度受限于 expert 数量（Qwen: 128 experts, DeepSeek-V3: 256 experts），加上 locking 进一步降低并发度至 <64，单个 A100 SM 即可提供足够的并行处理能力；(2) **SM-local shared memory**：将 per-GPU activated expert 计数器 L[1..G] 和对应的锁 l[1..G] 放置在 SM 共享内存中，实现快速访问；(3) **test-and-set lock**：使用简单的 test-and-set 自旋锁进行 GPU 线程间同步；(4) **全序锁获取**：通过按 GPU ID 全局顺序获取锁来避免死锁；(5) **CUDA Graph 预编译**：将 kernel 集成进 vLLM 的 decode phase CUDA Graphs，为 power-of-two batch sizes（up to 32 tokens/GPU）预编译，非 power-of-two 通过 padding 复用。

  此外 METRO 在通信层面改变了 kernel 间的调度：原 all-to-all dispatch 的 kernel 序列被替换为 all-gather + top-k + METRO routing + FFN + all-to-all combine 的序列。

  实验比较：(a) METRO routing kernel 延迟 vs FFN 层延迟 —— kernel 最多 26us vs FFN 最多减少 81us；(b) all-gather communication time vs all-to-all communication time — 两者无统计显著差异（NVLink latency 主导，bandwidth 开销 3us << NCCL launch ~100us）；(c) METRO top-k 额外开销（redundant computation on full token set）vs 原 top-k — 最多增加 3us (<1% 层时间)；(d) CUDA-based optimal algorithm (GPU push-relabel max-flow) vs METRO greedy — optimal 开销 290us+ (86.4%-103.8% FFN time) vs METRO <26us；(e) CPU-based optimal algorithm (Dinic max-flow) vs METRO — 含 CPU-GPU 数据传输 116-128us + 26.5-29.2us transfer (31.4%-41.3% FFN time)。

- 后端平台是什么，配置是什么。
  NVIDIA A100 40GB GPU（108 SM, 每个 SM 64 FP32 CUDA cores + 4 Tensor Cores），600 GB/s NVLink（8 GPU 在同一 NVLink domain）。NVIDIA B200 192GB GPU（模拟器建模），900 GB/s NVLink。CUDA kernel 运行在单 SM 上，计数器 L 和锁 l 驻留在 SM shared memory。

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 vLLM 框架中集成的 METRO CUDA kernel 和 NCCL 通信原语。修改内容：
  (a) **METRO routing CUDA kernel**：实现 Algorithm 1，单 SM 执行，test-and-set lock 同步，shared memory 计数器；
  (b) **CUDA Graph integration**：在 vLLM compilation framework 中为 power-of-two batch sizes 预编译包含 METRO kernel 的 CUDA Graphs；
  (c) **通信原语替换**：将 MoE layer 的 dispatch 阶段从 NCCL all-to-all 替换为 NCCL all-gather；
  (d) **Top-k 范围扩展**：top-k 从 per-GPU local tokens 扩展到全局 all-gathered tokens；
  (e) **Latency breakdown 测量**：通过 profiling 各组件（top-k, routing kernel, all-gather, FFN）的延迟来验证 METRO 的 overhead 可被 FFN 减少所抵消。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源代码。基于 vLLM 开源框架实现，以下是评估原理和 kernel 级全流程：

  **METRO CUDA Kernel 执行原理**：
  ```
  Kernel Configuration:
    - Grid: 1 block
    - Block: min(N, 64) threads (N = 128 for Qwen, 256 for DeepSeek)
    - SM: 1 (bound to single SM)
    - Shared Memory: L[G] (int32 array, G=8-16), l[G] (int32 locks)

  Kernel Pseudocode (per-thread, expert i):
    if T[i] == 0: return                      // skip experts with no tokens
    
    // Step 1: Determine candidate GPUs from placement matrix
    candidates = []
    for g in 0..G-1:
        if A[i][g] == 1: candidates.append(g)
    
    // Step 2: Acquire locks in total order (GPU ID ascending)
    for g in sorted(candidates):
        while atomicCAS(&l[g], 0, 1) != 0:   // test-and-set spinlock
            // spin
    
    // Step 3: Find GPU with minimum activated experts
    best_g = candidates[0]
    for g in candidates:
        if L[g] < L[best_g]: best_g = g
    
    // Step 4: Assign expert to best_g
    y[i][best_g] = 1
    atomicAdd(&L[best_g], 1)
    
    // Step 5: Release locks in reverse order
    for g in reverse(sorted(candidates)):
        atomicExch(&l[g], 0)
  ```

  **Kernel 输入到性能输出全流程**：
  ```
  Input:
    - A[128][8]: expert-GPU placement matrix (host->device, read-only, global mem)
    - T[128]: token count per expert (device, populated by top-k on all-gathered tokens)
    - G = 8: number of GPUs
  
  Execution (on single SM):
    1. Load A and T from global memory
    2. Initialize L[0..7] = {0} in shared memory
    3. Initialize l[0..7] = {0} in shared memory (0 = unlocked)
    4. Launch min(N,64) threads:
       - Each thread processes one expert i
       - Concurrently execute lock acquire-assign-release cycle
       - Total global memory reads: |A| entries (128 * avg_replicas)
       - Shared memory access: O(G) per lock acquire/release
    5. Write y[128][8] to global memory (output)
    6. Kernel overhead measured: 17-26us (varies with replication ratio)
  
  Communication sequence:
    NCCL All-gather: 2MB/GPU -> ~3us bandwidth + ~100us launch latency
    Top-K on all tokens: 17us->20us (vs original 17us->19us, +3us max)
    METRO Kernel: 17us->26us
    FFN (activated experts only): 230us->311us (varies with replication, ~81us reduction vs EPLB)
    NCCL All-to-all Combine: same as EPLB baseline
  ```

  关键结论：METRO routing kernel 的计算 overhead (17-26us) + top-k overhead (<3us) + 通信 overhead (~3us bandwidth) 总计最多约 30us，远低于其带来的 FFN 时间减少 (up to 81us)，净收益 ~50us/layer。在 30-layer Qwen3-30B 模型上，每 decode step 累积收益显著，最终端到端 decode latency 降低 11%-22%。

## ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 **ElasticMoE HMM Data Plane 中的五个低层 C++ primitives**，运行在 Ascend 910C NPU 上，为弹性缩放提供高效的设备内存管理和跨 NPU 数据传输基础：(1) **IpcSafeAllocator**：覆盖 PyTorch 默认 `TorchCachingAllocator`，用 `torch.ones()`、`torch.empty()`、`torch.full()` 等核心分配函数来分配 IPC 兼容的物理内存区域（Ascend IPC API 标记），使得张量可直接跨进程共享而不需额外拷贝；(2) **disk-copy**：按张量名称/partition index（TP rank）/layer type 选择性从磁盘加载权重到目标 NPU，避免同一张量在不同 NPU 上被多次从磁盘读取，最小化最慢的磁盘→NPU 链路使用；(3) **p2p-copy**：通过 Ascend HCCL 集合通信库（isend/irecv/broadcast）经 Unified Bus 或 RDMA 链路进行 NPU 间异步 P2P 传输，使用 `aclrtMemcpyAsync` API 直接 device-to-device 传输绕过 host memory，可选独立 stream 避免阻塞当前 NPU 计算上下文，比磁盘 I/O 快一个数量级；(4) **zero-copy**：通过 `rtIpcSetMemoryName()` 注册内存句柄 + `rtSetIpcMemPid()` 白名单目标进程 + UNIX domain socket 传输句柄 + `rtIpcOpenMemory()` 导入 + `torch::from_blob()` 封装，实现跨进程零拷贝张量共享，避免数据实际传输；(5) **vpage-remap**：通过 `aclrtMallocPhysical` 分配非连续物理页 → `aclrtReserveMemAddress` 预留连续虚拟地址范围 → `aclrtMapMem` 映射物理页到虚拟地址空间，使 kernel 将 expert 权重视为连续张量（满足 GEMM kernel 要求），而底层物理放置灵活可重映射，缩放时只需更新映射而无需重新分配和拷贝整个缓冲区。实验通过 Ablation 量化各组件贡献（Table 1/Table 3）：逐步禁用 IPCAlloc→HCCL→PreInit→ZeroCopy，测量 DP3↔DP4 缩放时间和 peak memory。ElasticMoE full：scale-up 2.43s / 0 downtime / 275.2 GB peak memory。

- 后端平台是什么，配置是什么。
  **Ascend 910C NPU**（每颗 64 GB HBM），部署在 Huawei CloudMatrix384 supernode 中。使用 **Huawei CANN (Compute Architecture for Neural Networks)** API 进行设备内存管理、HCCL (Huawei Collective Communication Library) 进行集合通信和 P2P 传输。Ascend Unified Bus (UB) 提供 NPU 间 non-blocking all-to-all 高带宽互联，也支持 RDMA 跨节点链路。

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 ElasticMoE HMM data plane 的 C++/PyBind11 实现，通过 PyBind 暴露给 Python 层调用。核心评估方式为 Ablation study：在 DP3→DP4 scale-up 上逐步禁用各 primitive，重复 3 次取均值。评估使用以下自定义组件：(a) IpcSafeAllocator 覆盖 PyTorch 默认分配器；(b) p2p-copy 使用 HCCL `init_process_group` 建立跨设备通信域，`isend/irecv/broadcast` 进行异步传输，`aclrtMemcpyAsync` 进行设备间拷贝；(c) zero-copy 使用 Ascend `rtIpcSetMemoryName/rtIpcOpenMemory` + `rtSetIpcMemPid` 白名单机制；(d) vpage-remap 使用 `aclrtMallocPhysical/aclrtReserveMemAddress/aclrtMapMem` 进行虚拟内存映射；(e) disk-copy 使用 CANN API 按 filter（name/partition/layer）选择性磁盘加载。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供独立开源代码。ElasticMoE 的 kernel 级别 primitives 评估原理和执行流程：

  ```
  === Ablation 实验流程 (Table 1: DP3→DP4 scale-up) ===
  
  Step 1: IPC 兼容内存分配 (IpcSafeAllocator)
    输入: PyTorch tensor shape, dtype
    执行: 拦截 torch.ones/empty/full → 调用 CANN aclrtMalloc (IPC-compatible flag)
    → 返回物理内存地址
    输出: 可跨进程共享的张量
    禁用效果: scale time 2.43→3.14s (+29%), peak mem 275.2→290.0 GB (+5.4%)
  
  Step 2: P2P 数据传输 (p2p-copy)
    输入: source NPU tensor handle, target NPU id, tensor size
    执行: 初始化 HCCL domain → target NPU 通过 aclrtMalloc 分配目标张量
    → aclrtMemcpyAsync 异步传输 (Unified Bus/RDMA)
    → 可选独立 stream 避免阻塞计算
    输出: target NPU 上的张量副本
    禁用效果: scale time +IPCAlloc 3.14→+HCCL 10.42s (+232%, 慢一个数量级)
    替代方案: 退化为从磁盘逐张量加载
  
  Step 3: Zero-Copy 跨进程张量共享 (zero-copy)
    输入: IpcSafeAllocator 分配的张量, source process PID, target process PID
    执行: 
      (a) HMM 通过 rtIpcSetMemoryName(tensor_ptr, name) 注册内存句柄
      (b) HMM 通过 rtSetIpcMemPid(pid) 白名单目标进程
      (c) 通过 UNIX domain socket (ZMQ) 传输句柄名称到 IMM
      (d) IMM 通过 rtIpcOpenMemory(name) 导入物理指针
      (e) IMM 通过 torch::from_blob(ptr, shape, dtype) 封装为 PyTorch tensor
    输出: 两个进程引用同一物理内存（零数据拷贝）
    禁用效果: scale time +HCCL+PreInit 62.78→+ZeroCopy 67.40s, 引入 67.40s downtime
  
  Step 4: 虚拟 Expert 管理 (vpage-remap)
    输入: expert 权重张量列表, 每个 expert 的物理页大小, 目标虚拟地址布局
    执行:
      (a) aclrtMallocPhysical 为每个 expert 分配独立非连续物理页
      (b) aclrtReserveMemAddress 预留连续虚拟地址空间
      (c) aclrtMapMem 将各物理页绑定到虚拟地址对应偏移
      (d) 缩放时: 更新虚拟→物理映射指向新页 (本地分配或 p2p-copy 接收)
      (e) 旧映射保持活跃直到新推理实例接管
      (f) 过渡完成后: 解绑旧物理页, 释放
    输出: kernel 视角为连续张量（满足 GEMM 对齐要求），物理内存灵活可重映射
    收益: 避免 EP 重配置时的大缓冲区重新分配和全量拷贝，降低 peak memory 和延迟
  
  === 性能输出 (Ablation Table 1) ===
  完整 ElasticMoE:       scale-up 2.43s, downtime 0, peak mem 275.2 GB
  - IPCAlloc:            scale-up 3.14s, downtime 0, peak mem 290.0 GB
  - IPCAlloc - HCCL:     scale-up 10.42s, downtime 0, peak mem 290.0 GB
  - IPCAlloc - HCCL - PreInit: scale-up 62.78s, downtime 0, peak mem 290.0 GB
  - All disabled (no ZeroCopy): scale-up 67.40s, downtime 67.40s, peak mem 290.0 GB
  ```

  关键结论：(a) ZeroCopy 对消除 downtime 最关键——无 ZeroCopy 时 downtime=scale time；(b) HCCL P2P 比磁盘加载快约一个数量级（10.42s vs 62.78s）; (c) PreInit（IMM 预热实例）贡献最大延迟改善，从 62.78s 降到 10.42s；(d) IPCAlloc 主要降低 peak memory 而非延迟（-5.4% peak mem）；(e) 全部四个机制联合作用才能实现低延迟、零停机缩放。

## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 D2MoE 的 **Parallel Loading Dequantization Kernel**，针对 MWQ 嵌套量化结构在边缘 GPU 上的反量化性能优化。核心包括：(1) **加载并行性**：将量化数据从磁盘直接传输到 GPU global memory，与激活值从 global memory 向 L2 cache 的移动重叠（DMA engine + CUDA stream 并行）。(2) **计算并行性**：expert 反量化在 CUDA Cores 上执行，与 Tensor Cores 上的 expert 矩阵计算同步进行（Figure 8），利用 GPU 分离的 CUDA Core 和 Tensor Core 资源。(3) **优化 binary operation**：借鉴 Any-Precision LLM [29]，避免传统 bit-transpose 方法中 INT → FP 的多轮类型转换链，binary residual (±1) 通过单个 bit extract + conditional sign assignment 实现，每 element 仅需 1 次 FMA。

  实验比较（Figure 12，dequantization overhead 分析）：D2MoE-V1 在 LLaMA-MoE-3.5B 和 Mixtral 8×7B 上测量 dequantization 的计算开销、峰值内存开销和延迟开销。4 requests 时计算开销 20.77%、延迟开销 18.56%；32 requests 时因 MWQ 嵌套权重复用增加分别降至 16.77% 和 5.3%。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 3060 (6GB, Ampere, 3584 CUDA Cores, 112 Tensor Cores) 和 Jetson AGX Orin 64GB (2048 CUDA Cores, 64 Tensor Cores)。CUDA 基于 Ampere 和 Ada Lovelace 架构。GPU 存储层级：NVMe SSD (3.5 GB/s) → GPU Global Memory → L2 Cache → Shared Memory → Registers。使用 Triton 进行 I/O-compute 并行编程。

- 评估性能的软件/脚本是什么。修改了什么。
  自研 D2MoE 引擎 (~2,500 LOC Python + CUDA)。主要修改：(a) 实现 MWQ 专用 CUDA dequantization kernel，将 per-group scale/zero-point 应用与 binary residual 累加融合为单一 kernel；(b) 使用 CUDA stream 异步重叠 disk→GPU 数据传输与 dequantization kernel 执行；(c) 利用 Triton 协调 dequantization + expert FFN GEMM 的 pipeline 执行。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供公开开源代码仓库。Kernel 原理如下：

  ```
  === MWQ Dequantization Kernel 全流程 ===

  Input (per expert, loaded from disk to GPU global memory):
    Q_W_b1: int8 [s, h] (b_1-bit quantized, e.g. INT2 packed)
    z_b1: int8 [s, h/128] (per-group zero points, group_size=128)
    s_b1: fp16 [s, h/128] (per-group scales)
    Q_W_bk: packed 1-bit [s, h] (binary residual for k=2..K)
    s_bk: fp16 [s, h/128] (per-group scales for k=2..K)

  Step 1: Parallel Loading (CUDA Stream I/O)
    cudaMemcpyAsync(Q_W_b1..bK, disk, sizes, H2D, load_stream)
    activation X moves: global mem → L2 cache (comp_stream)

  Step 2: Dequantization Kernel (CUDA Cores, per-group parallel)
    for each group g_id (128 elements):
      # Base asymmetric dequant to FP16
      for idx in group:
        W_fp16[idx] = (int(Q_W_b1[idx]) - int(z_b1[g_id])) * s_b1[g_id]
      
      # Binary residual accumulation (k = 2..K):
      for k in 2..K:
        for idx in group:
          sign_bit = (Q_W_bk_packed[idx/8] >> (idx%8)) & 0x01
          W_fp16[idx] += (sign_bit ? 1.0 : -1.0) * s_bk[g_id]
          # 仅 1 次 bit extract + 1 次 FMA per element per bit-level
          # vs 传统 bit-transpose: unpack → int8→int32→fp32→fp16 (4 ops)
      
      store W_fp16[group] to shared memory

  Step 3: Expert FFN (Tensor Cores, overlaps with Step 2 of next expert)
    for expert e in I/O-complete queue:
      GEMM(W_fp16 @ X) using Tensor Core FP16 MMA
      # CUDA Cores dequantize next expert while Tensor Cores compute current

  === 传统 bit-transpose vs 本文 binary ops ===
  传统 (per INT2):
    unpack_2bit(packed) → int8_val → int32_val → fp32_val → fp16_val → dequant
    (5+ operations per element, multiple type conversions)
  本文 (binary residual path, k≥2):
    bit_extract(packed) → sign → fp16_val += sign * scale
    (2 operations: 1 logical + 1 FMA, zero type conversion)
  ```

  性能：dequantization overhead 随 request 数增加从 ~20% 降至 ~5%（权重复用），临时 FP16 内存立即释放，不影响 peak memory。