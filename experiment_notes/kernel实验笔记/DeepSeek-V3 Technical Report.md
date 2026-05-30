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
