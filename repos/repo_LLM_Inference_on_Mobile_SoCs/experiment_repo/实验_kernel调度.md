## Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：HeteroInfer 在 kernel 调度层面的核心贡献包括以下四个部分：
    
    **(1) GPU-NPU 异构 Kernel 调度的 Layer-level 策略**：根据算子的计算亲和性将 Matmul 分配给 NPU（利用其 systolic array 的 weight-stall 范式），将 RMSNorm、SwiGLU、Attention 的非 Matmul 部分分配给 GPU（利用其线性性能和灵活性）。利用计算不变量 `[M,N] × [N,K] → [[K,N] × [N,M]]^T` 交换张量顺序以适配 NPU-2 (order-sensitive performance)。
    
    **(2) Tensor-level GPU-NPU 并行 Kernel 调度**：
    - Weight-centric partition：沿权重张量行维度拆分，GPU 和 NPU 各计算一部分子张量，partition ratio 由 offline solver 确定（如 [28672,4096] × [4096,1] decoding 场景采用 3:1 GPU:NPU 比）
    - Activation-centric partition：为支持 NPU 不支持的动态序列长度，将激活张量沿序列长度维度拆分为多个标准形状子张量 + 一个动态形状子张量，标准形状部分在 NPU 顺序执行，动态部分卸载到 GPU
    - Hybrid partition：结合 activation-centric（处理动态形状）和 weight-centric（处理 NPU 的 shape-sensitive 性能退化）
    
    **(3) Fast Synchronization（微秒级 GPU-NPU 同步）**：利用 UMA（统一内存架构）的共享地址空间消除数据拷贝；利用 LLM 逐层相同操作的特性，使 GPU kernel 等待时间可预测；CPU sync thread 先 sleep(predicted_wait) 再轮询 output tensor flag bit（数微秒），替代 clFinish 的 ~400μs 固定开销。Prefill 阶段为 NPU-dominant（NPU 执行覆盖 GPU + sync），decoding 阶段为 GPU-dominant（NPU 执行被 GPU 覆盖，利用 GPU queue ordering 避免额外提交开销）。
    
    **(4) Offline Profiler + Online Solver 协同**：Profiler 在有限搜索空间（仅 LLM 权重形状、NPU stage performance 限制最小尺寸、预定义标准序列长度）内离线测量 GPU/NPU kernel 的执行时间、内存带宽和同步开销（< 20 分钟完成）。Solver 在线根据实际序列长度，利用 GPU-1 (linear performance) 和 NPU-1 (stage performance) 特性插值估计非标准形状延迟，通过 min-max 优化选择最优 partition strategy 和 partition ratio。

  - 实验比较：(1) GPU 性能特征：不同张量大小下的 FLOPS 线性增长 → 饱和曲线（Characteristic GPU-1）；(2) NPU 性能特征：stage performance（不同张量尺寸下的阶梯状延迟曲线）、order-sensitive performance（`[14336,4096]×[4096,K]` vs `[K,4096]×[4096,14336]` 对比，最高 6× 性能差）、shape-sensitive performance（输入行列比对 NPU 效率的影响）；(3) SoC 内存带宽：单处理器 vs. 多处理器并发时的可用带宽（单 GPU/NPU/CPU ~40-45 GB/s，GPU+NPU 并发 ~60 GB/s）；(4) NPU 计算图生成时间随张量形状变化的开销；(5) 各框架 prefill speed（固定 seq_len 64/256/1024）、decoding rate 对比；(6) fast synchronization 消融（Hetero-layer 和 Hetero-tensor 在 prefill 和 decoding 阶段有/无 fast sync 的性能对比，decoding 阶段 Llama-8B 有 4.01× 收益）；(7) prefill ablation：Naïve NPU → +activation-centric → +tensor reorder → +weight-centric → +fast sync 的逐级收益；(8) 与手游并发的 GPU interference 实验（FPS 变化 + prefill/decoding 速度变化）。

- 后端平台是什么，配置是什么。
  - 主平台：Qualcomm Snapdragon 8 Gen 3 SoC
    - CPU：Arm CPU（不作为计算后端，仅用于控制面和同步）
    - GPU：Adreno 750，FP16 ~1 TFLOPS 实际 / 2.8 TFLOPS 理论峰值
    - NPU：Hexagon NPU，FP16 ~10 TFLOPS 实际 / 17 TFLOPS 理论峰值，内含多个 32×32 systolic array，采用 weight-stall 计算范式
    - 内存：统一内存架构 (UMA)，理论带宽 68 GB/s，最大可达 ~61.9 GB/s（连续大块 memcpy）
  - 辅助平台：Qualcomm Snapdragon 8 Elite（内存带宽与 8 Gen 3 相同，prefill 性能提升约 10.5%）
  - 参考 SoC 规格对比 (Table 2)：MediaTek Dimensity 9300 (Mali-G720 GPU, APU 790 NPU)、Apple A18 (Bionic GPU, Neural Engine)、Nvidia Orin (Ampere GPU, DLA)、Tesla FSD (FSD GPU, FSD D1)

- 评估性能的软件/脚本是什么。修改了什么。
  - GPU kernel：使用 OpenCL 开发优化后的 GPU kernel
  - NPU 算子：通过 Qualcomm QNN (Qualcomm AI Engine Direct SDK) [43] 集成 NPU 算子
  - 对比框架：llama.cpp [11] (CPU/GPU, W4A16)、MLC-LLM [34] (GPU, W4A16)、MNN-LLM [55] (GPU, W4A16)、llm.npu [56] (NPU, INT8/FP16 mixed)、PowerInfer-2 [60] (NPU, W4A16 + sparse)
  - 模型量化：HeteroInfer 采用 W4A16 (weight-only quantization)——权重以 INT4 存储，计算时 dequantize 为 FP16 执行。Decoding 阶段仅使用 NPU 的 TOPS（因 NPU 当前不支持 W4A16 的 decoding）
  - 核心修改：(1) 新增 GPU 和 NPU 间的 tensor partitioning 和 result merge 逻辑；(2) 新增 fast synchronization 机制（替代 clFinish）；(3) 新增 memory pool 管理 host-device 共享缓冲区（绕过 device driver 组织）；(4) 新增 control plane decider 根据 solver 输出动态选择执行策略

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未明确说明 HeteroInfer 自身代码是否已开源。通过 web search 未找到公开 GitHub 仓库。使用的依赖组件：OpenCL（GPU kernel 开发标准）、Qualcomm QNN SDK（Qualcomm 官方 NPU 工具链，https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk）。

  - **Kernel 调度评估原理和 Kernel 输入到性能输出全过程（以 FFN-down layer prefill 为例）**：
    
    **原理概述**：
    HeteroInfer 的 kernel 调度本质是在每个 Matmul 算子上求解一个 min-max 优化问题：在 GPU 和 NPU 之间分配计算负载（partition），使得两者的执行时间尽可能接近（max 最小化），同时同步开销（T_sync + T_copy）可控。决策空间受 NPU 的三阶段特性（stage / order-sensitive / shape-sensitive performance）和 SoC 内存带宽约束。
    
    **全过程（以 Llama-8B FFN-down layer, weight=[4096,14336], activation=[14336,320] 为例）**：
    
    ```
    阶段 0：Offline Profiling — 构建性能查找表
    ─────────────────────────────────────────────
    输入: Snapdragon 8 Gen 3 SoC
    
    1. GPU Profiling (OpenCL):
       for weight_shape in {[4096,4096], [28672,4096], [4096,14336], ...}:
         for act_shape in {[4096,1], [4096,128], [4096,256], [4096,512], [4096,1024], ...}:
           gpu_kernel = compile_opencl_kernel(weight_shape, act_shape)
           t_gpu = measure_execution_time(gpu_kernel)
           mem_bw_gpu = measure_memory_bandwidth(gpu_kernel)
           记录到 lookup_table
    
    2. NPU Profiling (QNN):
       for weight_shape in {同上}:
         for act_shape in {预定义标准序列长度: 64, 128, 256, 512, 1024}:
           npu_graph = qnn_build_graph(weight_shape, act_shape)
           t_npu = measure_execution_time(npu_graph)
           mem_bw_npu = measure_memory_bandwidth(npu_graph)
           记录到 lookup_table
       
       约束：
         - 子张量尺寸 ≥ 32（NPU 32×32 systolic array 的 stage performance 下界）
         - 激活形状仅标准长度（NPU 静态图限制）
    
    3. Synchronization Profiling:
       t_sync_clfinish = ~400 μs (clFinish)
       t_sync_fast = ~tens of μs (fast sync: sleep + polling)
       t_copy = 0 (UMA, 共享地址空间, 无数据拷贝)
    
    总耗时: < 20 分钟
    
    阶段 1：Online Solver — 为当前 seq_len 选择策略
    ─────────────────────────────────────────────
    输入: seq_len = 320, weight=[4096,14336]
    
    1. 判定 seq_len 是否匹配标准形状:
       320 不匹配任何标准长度 (256, 512, ...)
       → 需要 dynamic shape 支持
    
    2. 枚举候选策略并估计延迟:
       
       策略 A — No Partition (NPU-only with Padding):
         activation pad to 512
         T_npu = lookup([4096,14336], [14336,512])         // 从 profiling 表查
         T_total = T_npu
    
       策略 B — Activation-centric:
         partition activation: [14336,320] → [14336,256](标准) + [14336,64](动态)
         T_npu = lookup([4096,14336], [14336,256])         // NPU 标准图
         T_gpu = estimate([4096,14336], [14336,64])        // GPU-1: linear 插值
         T_total = max(T_npu, T_gpu) + T_sync
    
       策略 C — Hybrid (Activation-centric + Weight-centric):
         step 1 — Activation partition: [14336,320] → [14336,256] + [14336,64]
         step 2 — Weight partition for NPU sub-task:
           因 NPU-3 shape-sensitive: [4096,14336] × [14336,256] NPU 效率仅 ~0.5-1.5× GPU
           weight partition ratio = 2:3 (GPU:NPU = 40%:60%)
           GPU_sub: [4096*0.4, 14336] × [14336,64]    → T_gpu1
           NPU_sub: [4096*0.6, 14336] × [14336,256]   → T_npu1
         T_total = max(T_gpu1, T_npu1) + T_sync
    
       （策略 D — GPU-only、策略 E — NPU-only 同样枚举）
    
    3. 选择 T_total 最小的策略 → Hybrid (2:3 weight partition)
    
    阶段 2：Kernel 执行 — GPU-NPU 并行计算
    ─────────────────────────────────────────────
    输入: 激活张量 A[14336, 320], 权重张量 W[4096, 14336]
    
    ┌─ GPU 路径 ─────────────────────────────────────┐
    │ 1. 权重分片:                                    │
    │    W_gpu = W[0:1638, :]   // 4096×0.4 ≈ 1638 行│
    │                                                 │
    │ 2. 激活分片:                                    │
    │    A_gpu = A[:, 256:320]  // 序列长度 64 的部分  │
    │                                                 │
    │ 3. OpenCL Kernel Launch:                        │
    │    cl_mem buf_W = clCreateBuffer(W_gpu)         │
    │    cl_mem buf_A = clCreateBuffer(A_gpu)         │
    │    cl_mem buf_O = clCreateBuffer([1638, 64])    │
    │    clSetKernelArg(kernel_matmul, ...)           │
    │    clEnqueueNDRangeKernel(kernel_matmul, ...)   │
    │    // 无 clEnqueueWriteBuffer (UMA 共享内存)     │
    │                                                 │
    │ 4. 执行: GPU 计算 [1638, 14336] × [14336, 64]  │
    │    → 输出 O_gpu[1638, 64]                       │
    └─────────────────────────────────────────────────┘
    
    ┌─ NPU 路径 ─────────────────────────────────────┐
    │ 1. 权重分片:                                    │
    │    W_npu = W[1638:4096, :]  // 剩余 2458 行     │
    │                                                 │
    │ 2. 激活分片:                                    │
    │    A_npu = A[:, 0:256]      // 标准形状          │
    │                                                 │
    │ 3. QNN Graph Execution (预生成图):               │
    │    qnn_graph = prebuilt_graphs["ffn_down_256"]  │
    │    qnn_tensor_set_input(qnn_graph, W_npu, A_npu)│
    │    qnn_graph_execute(qnn_graph)                 │
    │    // 内部: 32×32 systolic array                │
    │    //   tile 分解 → weight stall 计算 → 累加     │
    │                                                 │
    │ 4. 执行: NPU 计算 [2458, 14336] × [14336, 256] │
    │    → 输出 O_npu[2458, 256]                      │
    └─────────────────────────────────────────────────┘
    
    ┌─ Fast Synchronization (NPU-dominant prefill) ──┐
    │ // NPU 计算量大, GPU 执行被 NPU 覆盖             │
    │                                                 │
    │ T0: NPU submit → GPU submit                     │
    │ T1: CPU sync thread:                            │
    │     predicted_wait = T_npu - T_gpu_estimated    │
    │     usleep(predicted_wait - margin)  // ~80-100μs│
    │     while (O_gpu.flag_bit == 0):  // 轮询 flag  │
    │       continue  // 仅数微秒                     │
    │ T2: O_gpu.flag_bit = 1 → CPU 通知 NPU          │
    │     // 此时 NPU 大概率已完成或即将完成            │
    │                                                 │
    │ 同步开销: 数微秒 ~ 数十微秒 (vs. clFinish ~400μs)│
    └─────────────────────────────────────────────────┘
    
    ┌─ Result Merge ─────────────────────────────────┐
    │ // 合并 GPU 和 NPU 的输出                       │
    │ O_final[0:1638, :]   = O_gpu[1638, 64]         │
    │ O_final[1638:4096, :] = O_npu[2458, 256]       │
    │ // O_final 形状: [4096, 320]                     │
    │ → 传给下一 decoder layer                         │
    └─────────────────────────────────────────────────┘
    ```
    
    **Decoding 阶段的 Kernel 调度差异**：
    ```
    输入: 激活 [4096, 1]（单 token）, 权重 [28672, 4096] (FFN-up)
    
    GPU path: matrix-vector [28672, 4096] × [4096, 1]
      → GPU 线性性能 + 高内存带宽, 约 511 μs
    NPU path: 同样计算
      → NPU-1 stage performance（32×32 systolic array 对小激活效率低）, 约 693 μs
    
    → Weight-centric partition (GPU:NPU = 3:1):
      GPU: [28672*0.75, 4096] × [4096, 1]  ≈ 75% 工作量
      NPU: [28672*0.25, 4096] × [4096, 1]  ≈ 25% 工作量
      → 并行执行, 最大化内存带宽 (43.3 → 59.5 GB/s)
    
    Fast Sync (GPU-dominant):
      NPU submit → GPU submit (kernel_1)
      → NPU 完成 → enqueue GPU kernel_2
      → GPU queue ordering 确保 kernel_1 先于 kernel_2
      → 无额外同步提交开销
    ```

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：论文在 Qualcomm Hexagon NPU 上实现了自定义的低级 kernel，绕过闭源 QNN SDK，直接通过 reverse engineering 发现并使用 FP16 HMX 未公开指令。核心 kernel 实现包括：
    **(1) Mixed-Precision Dequantization GEMM Kernel**：在 HVX 向量单元上执行 INT4→FP16 运行时反量化 + 权重重组 → 连续写入 TCM → HMX 矩阵单元执行 FP16 tile MatMul。使用 vlut16 LUT 指令替代传统的 mask-unpack-convert 多指令序列进行 INT4→FP16 转换，并使用 vlut16 实现 4 组 scales 的广播。离线阶段执行 pre-quantization permutation（权重重排为 HMX tile layout）、tile-group quantization（group_size=32，即 2×16 tile）、post-quantization super-group coalesce（8 group → 1 super-group，256 INT4 填满 1 个 128-byte HVX 向量寄存器）。
    **(2) LUT-Based FlashAttention Kernel**：在 Hexagon NPU 上实现 FP16 FlashAttention（Algorithm 1，含 HMX MatMul + HVX LUT-exp + HVX rowmax/rowsum）。使用 vgather 指令实现 LUT-based exp（64 KiB TCM，32768 entries），利用 safe softmax 使所有 exp 输入 ≤0，忽略符号位 + 左移一位生成 vgather 字节偏移。Attention MatMul 使用 FP16 HMX，关键累加（rowsum, rescale）使用 FP32，其余保持在 FP16。
    **(3) DMA-based Memory Management**：利用 DMA 引擎（≥60 GB/s 读取带宽）进行 1D/2D 异步预取（prefetch）。HVX scatter/gather 和全部 HMX 指令仅访问 TCM；通过 12fetch 指令从 DDR → L2 cache，DMA 从 DDR → TCM。
    **(4) CPU-NPU Shared Memory Communication**：使用 rpcmem（kernel dmabuf wrapper）共享物理内存，消除 CPU-NPU 数据拷贝。FastRPC 启动远程 NPU session，NPU 端线程持续轮询共享内存区域接收 CPU 计算请求。手动 cache maintenance（因 Snapdragon SoC 仅单向 coherence）。
  - 实验比较：(1) GEMM dequantization ablation：baseline（column-major + scatter dequant）→ +HMX layout → +super-group coalesce（最终 "Ours"）→ "no dequantization" 性能上界。最终方法加速 9.65–19.04×，仅比上界慢 27%；(2) Softmax ablation：LUT-based exp vs. FP32 polynomial exp vs. FP16 polynomial exp，query_len ∈ {1,4,16} × KV_len ∈ {1024,4096,16384}；(3) HMX vs. HVX FP16 GEMM 吞吐量对比（Table 2: 12 TFLOPS vs. 33 GFLOPS，~365× 差距）；(4) DMA vs. HVX 内存读取带宽对比（60 vs. 26 GB/s）；(5) 不同 batch size (1/4/8/16) 下 end-to-end decoding throughput；(6) NPU vs. GPU（llama.cpp OpenCL backend）prefill 和 decoding throughput 对比；(7) 与 QNN FP16 参考数据的 prefilling 对比；(8) CPU utilization / memory consumption 随 batch size 变化。

- 后端平台是什么，配置是什么。
  - Qualcomm Snapdragon 8 Gen 2（Hexagon NPU V73）：OnePlus Ace3。NPU 32-bit 虚拟地址空间限制 2 GiB（导致 ≥3B 模型无法运行）。
  - Qualcomm Snapdragon 8 Gen 3（Hexagon NPU V75）：OnePlus 12
  - Qualcomm Snapdragon 8 Elite（Hexagon NPU V79）：OnePlus Ace5 Pro
  - NPU 内部：HVX 向量单元 4-6 个（每单元 32×1024-bit 寄存器），HMX 矩阵单元 1-2 个（FP16 tile=32×32, 2 KiB/tile），8 个 scalar VLIW 硬件线程。TCM 8 MiB（software-managed on-chip memory），L2 cache 1 MiB。
  - FP16 HMX GEMM 峰值：~12 TFLOPS（1024³ GEMM，所有数据在 TCM）。DMA 读取 DDR 带宽 ≥60 GB/s。HVX 向量单元内存读取带宽 <30 GB/s。
  - 对比 GPU：Adreno GPU 通过 llama.cpp OpenCL backend（commit 1caae7f，含 Q4_0 优化 kernel）

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 NPU operator library（编译为 Hexagon DSP 独立 shared object）+ llama.cpp Hexagon NPU backend（CPU 侧集成，约 7K 行 C/C++ + inline assembly）
  - 工具链：Hexagon SDK 6.0.0.2 的 LLVM toolchain（非 QNN）。依赖 libcdsprpc.so（Android vendor library，提供 dmabuf 分配/映射接口）和 FastRPC（Hexagon SDK 远程 session 管理）
  - 关键修改：
    (1) **HMX FP16 GEMM kernel**：通过 reverse engineering 二进制库发现并使用未公开的 HMX 指令——hmx_load_activation（从 TCM 加载 activation tile）、hmx_load_weight（加载 weight tile）、hmx_matmul_accumulate（tile MatMul + 累加）、hmx_store（输出 tile 到 TCM）、hmx_scale_bias（per-channel scale/bias）
    (2) **HVX dequantization kernel**：利用 vlut16（16-entry LUT，8-bit index → 16-bit output）和 vgather（gather 64×2-byte elements from TCM）等 SIMD 指令实现高效反量化和 exp 计算
    (3) **FP16 FlashAttention kernel**：HMX MatMul (QK^T, PV) + HVX LUT-exp + HVX rowmax/rowsum + HVX rescale（Algorithm 1）
    (4) **NPU backend 集成**：rpcmem shared buffer 替代 llama.cpp 默认 buffer 类型；FastRPC session 管理；shared memory polling 通信；manual L2 cache invalidate/writeback
    (5) **CPU fallback**：lm_head/logits 因 NPU 32-bit 地址空间限制（vocab_size × batch_size × FP16 过大）回退到 CPU

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：主仓库 https://github.com/haozixu/llama.cpp-npu（MIT 许可证）；算子库 https://github.com/haozixu/htp-ops-lib。无 QNN 依赖。FP16 HMX 指令通过 reverse engineering 获取，论文已公开发布。

  - **Kernel 评估原理和 GEMM Dequantization 全过程（以 Qwen2.5-1.5B Attention Q-proj, weight=[2048,2048], batch=4 为例）**：

    **原理概述**：
    该 kernel 将 "INT4 量化权重 → 运行时反量化 → FP16 MatMul" 全流程在 NPU 上高效实现。核心策略：HVX（向量单元）使用 LUT 指令高效完成反量化，HMX（矩阵单元）使用 tile MatMul 获得 12 TFLOPS 算力。离线权重 layout 变换使反量化结果连续写入 TCM（消除 scatter），super-group coalesce 使每次 HVX 操作填满 128-byte 寄存器。

    ```
    # ========== 离线权重准备 ==========
    输入: FP16 W [2048, 2048]
    输出: Q4_0 量化权重（super-group 格式），存于 rpcmem shared buffer

    Step 1 — Pre-quantization permutation（图 4/6）:
      W_perm = rearrange_to_HMX_tile_layout(W)
      # 外层: 64×64 tiles [2048/32=64 rows, 2048/32=64 cols], column-major
      # 内层: 每 tile 内每两行 cross-lane shuffle（匹配 HMX memory layout）

    Step 2 — Tile-group quantization (group_size=32, 2×16 tile):
      在 permuted 内存顺序上，每连续 32 元素:
        s = max_abs(group_32) / 7.0
        quant = round(group_32 / s), clip [-7, 7]
        pack: (quant[2i] & 0x0F) | ((quant[2i+1] & 0x0F) << 4)
      存储: [16 bytes INT4] + [2 bytes FP16 scale] = 18 bytes/group

    Step 3 — Super-group coalesce（图 7）:
      8 groups → 1 super-group:
        [128 bytes INT4 (256 values)]  → fills 1 HVX register
        [16 bytes FP16 scales (8 scales)]

    # ========== 运行时 GEMM Kernel ==========
    输入: FP16 activation A [4, 2048]（batch=4, decoding），Q4_0 super-group 权重
    输出: FP16 C [4, 2048]

    # NPU 端执行（Hexagon DSP）:

    for tile_j in range(64):  # 64 weight tile columns

        # === HVX 反量化阶段 ===
        for sg in range(8):  # 8 super-groups per tile column

            # 1. DMA 预取当前 super-group 到 TCM
            dma_prefetch(TCM_ADDR_W + sg*144, DDR_W + sg*144, 144)

            # 2. vlut16: INT4 → FP16 直接转换（图 9）
            #    LUT 16-entry: [FP16(-7), ..., FP16(7)]
            v_int4 = vload(TCM_ADDR_W + sg*144, 128)      # 128 bytes
            v_fp16_lo = vlut16(v_int4, LUT_INT4_TO_FP16)  # 低 4-bit → 16×FP16
            v_fp16_hi = vlut16(v_int4 >> 4, LUT_INT4_TO_FP16) # 高 4-bit

            # 3. vlut16: scales broadcast
            v_scales = vload(TCM_ADDR_W + sg*144 + 128, 16) # 8 FP16 scales
            v_scales_lo = vlut16(CONST_IDX_LO, v_scales_lut)
            v_scales_hi = vlut16(CONST_IDX_HI, v_scales_lut)

            # 4. HVX FP16 multiply
            v_deq_lo = vmpy_f16(v_fp16_lo, v_scales_lo)
            v_deq_hi = vmpy_f16(v_fp16_hi, v_scales_hi)

            # 5. 连续写入 TCM（已是 HMX layout, 无需 scatter!）
            vstore(TCM_TILE_ADDR + sg*64,      v_deq_lo)
            vstore(TCM_TILE_ADDR + sg*64 + 128, v_deq_hi)

        # === HMX MatMul 阶段 ===
        # 激活 tile: [4, 32]（pad batch 4→32）, 权重 tile: [32, 32]
        hmx_load_activation(TCM_ACT_TILE)     # load [32,32] activation tile
        hmx_load_weight(TCM_TILE_ADDR)         # load [32,32] weight tile
        hmx_matmul_accumulate()                 # tile MatMul, FP32 accumulate
        # ... 循环所有 tile rows ...
        hmx_store(TCM_OUT_TILE)                # output FP16 [32,32]

    # DMA writeback: TCM → DDR
    dma_writeback(DDR_C, TCM_OUT, [4, 2048])

    # ========== 性能对比（OnePlus 12, V75, GEMV 等价 workload）==========
    # Baseline (column-major + scatter):
    #   每个 group 独立反量化 → scatter 写入 TCM（非连续，多次 memory transaction）
    # Ours:
    #   连续写入 TCM（HMX layout）+ LUT 替代指令序列
    #   加速: 9.65–19.04× (取决于矩阵大小)
    #   仅比"no dequantization"上界慢 27%（接近理论上限）
    ```

  - **LUT-Based FlashAttention Softmax 全过程**：
    ```
    # LUT 预计算（系统初始化，64 KiB TCM, 0.8% TCM 容量）:
    for i in range(32768):
        lut_exp[i] = FP16(exp(-i / 256.0))

    # 在线 exp（每个 FlashAttention tile, Algorithm 1）:
    # S = QK^T [B_q, B_kv], FP16
    # m_new = max(m_old, rowmax(S))
    # S' = S - m_new  (all ≤ 0, safe softmax)

    for each 64-element chunk:
        # 构造 vgather byte offset:
        #   x ≤ 0 → bit[15]=1 (sign)
        #   abs_x = x & 0x7FFF (ignore sign)
        #   offset = abs_x << 1 (×2 for FP16 2-byte alignment)
        v_offsets = vand(v_chunk, 0x7FFF)
        v_offsets = vasl(v_offsets, 1)
        v_exp = vgather(lut_exp_base, v_offsets)  # 24-48 packets latency (V75)

    # P = LUT_Exp(S - m_new)
    # l_new = exp(m_old-m_new)*l_old + rowsum(P, FP32)
    # O_new = diag(exp(m_old-m_new))*O_old + MatMul(P, V, FP32 accum)

    # 性能（OnePlus 12, Fig 14）:
    #   query=16, KV=16384: 2.19× vs FP32 exp, 1.60× vs FP16 poly exp
    #   query=1,  KV=1024:  1.26× vs FP32 exp
    ```

  - **CPU-NPU 通信机制**：
    ```
    # 初始化:
    FastRPC → start remote NPU session
    分配 rpcmem shared buffers (权重, KV cache, 激活, 通信 ring buffer)
    NPU thread → 进入 polling loop (轮询 command flag)

    # 每次 forward:
    CPU: fill activations → cache writeback → set READY flag
    NPU: detect READY → cache invalidate → execute kernels → cache writeback → set DONE
    CPU: detect DONE → cache invalidate → continue

    # lm_head 回退（因 NPU 32-bit 地址空间限制）:
    # vocab_size ~128K, batch>1 → logits tensor 过大
    # → CPU 执行 lm_head MatMul
    # batch=16 时 CPU logits 占比 ≥50%
    ```
