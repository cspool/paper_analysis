## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

- baseline方法是什么？
  现有低比特 KV cache 推理系统分为两类，均未能高效利用 Tensor Cores：(1) **Non-fused attention with separated kernels（Kivi）**：将 mixed-precision attention 分解为多个独立 kernel（dequantization kernel + attention kernel），各 kernel 独立 launch，中间数据反复读写 global memory。虽然灵活支持多种 attention variant，但增加了 launch overhead、inflated memory traffic，破坏了 on-chip data reuse；(2) **Fused attention on CUDA Cores only（Atom, QServe）**：将 dequantization 和 matmul 都实现在 CUDA Cores 上（FMA 指令），虽避免了 non-fused 的中间数据问题，但完全忽略了 Tensor Cores——现代 GPU 的主要算力来源（A100: Tensor Cores 312 TFLOPS FP16 vs CUDA Cores 19.5 TFLOPS FP32）。CUDA Cores 同时处理 dequantization（memory-bound）和 matmul（compute-bound），导致 register bandwidth 竞争、L1/L2 争抢、occupancy 下降，尤其在 GQA 等 arithmetic intensity 较高的 attention variant 下性能严重退化（QServe 在 GQA 下 speedup 从 MHA 的 3.5× 跌至 1.4×）。两类 baseline 都未能解决三个关键挑战：(C1) 低比特数据 layout 与 Tensor Core fragment layout 不匹配——量化后的 packed 数据直接 dequantize 会产生乱序的 register 分布，无法直接送入 Tensor Core mma；(C2) dequantization 频繁 stall warp 执行——FlashAttention 原始的单 warp 沿 N 维策略使 dequantization 序列化；(C3) 缺乏通用的系统级优化——不同量化算法使用不同的 scaling granularity（tensor-wise vs channel-wise），现有 mixed-precision kernel（Marlin, Ladder）仅针对静态权重，无法处理动态生成的低比特 KV cache。

  全栈执行例子（Baseline / QServe on A100, 128K context, LLaMA-3.1-8B GQA decode）：
  - 算法pipeline：在线 INT4 quantization（tensor-wise/channel-wise）→ KV cache 存储为 packed INT4 → decode 时每 token 执行 CUDA Core-only fused attention kernel（FMA dequantization + FMA GEMV/GEMM）。因 CUDA Cores 同时承担 dequantization（memory-bound，~50% kernel time）和 matmul（compute-bound），Tensor Cores 完全闲置
  - 系统框架：基于 FlashAttention kernel 修改，集成到 HuggingFace Transformers/vLLM serving pipeline；支持 paged attention memory management
  - 编译框架：论文未明确说明（CUDA 手工 kernel）
  - kernel调度：Block-wise tiling（Q tile + KV tile）→ cp.async 加载 packed KV + 量化参数 → CUDA Core FMA dequantization（INT4→FP16, per-element scale+zp）→ CUDA Core FMA matmul（QK^T, PV）→ online softmax → output write-back。全程仅在 CUDA Cores 上执行，Tensor Cores 无负载。Dequantization 消耗近 50% 的 kernel execution time
  - 硬件架构：NVIDIA A100 GPU（80GB HBM, 312 TFLOPS Tensor Cores FP16, 19.5 TFLOPS CUDA Cores FP32），Tensor Cores 利用率 ~0%

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BitDecoding 通过 **cooperative use of Tensor Cores + CUDA Cores** 将 Tensor Cores 引入低比特 KV cache 解码。四项设计分别对应 baseline 的三个核心挑战：

  **1. Layout Induction via Hardware Instructions（解决 C1: Layout 不匹配）**：
  利用 ldmatrix 的 thread-to-register 映射天然产生 Tensor Core 的 interleaved fragment layout。在 Residual Kernel 中，ldmatrix 加载 FP16 KV tile 后，各线程在寄存器内完成量化和 INT16 packing——因为 ldmatrix 建立的 interleaved 映射在打包过程中被"隐式保存"。Packing Kernel 以相同 ldmatrix 配置加载 packed 数据后，解量化结果自动对齐 Tensor Core 寄存器，无需全局 reshape。这比 Marlin 的离线 layout transformation kernel 和 Ladder 的迭代搜索快 3 个数量级（prefill: 0.06ms vs 58ms/4.79ms, decode: 0.008ms vs 0.41ms/0.65ms, Table II）。

  **2. Warp Parallelism Strategy（解决 C2: Dequantization stall）**：
  将 FlashAttention 的单 warp 沿 N 维改为多 warp（W_m=1, W_n>1），SM warp scheduler 交替调度多个 warp 执行 dequantization，消除单 warp 的序列化 stall。配合 Cooperative Softmax（register→shared memory→register cross-warp reduction），以仅 0.5% overhead 恢复多 warp 下的计算正确性。TC utilization 从 10.91%（W_n=1）提升到 19.66%（W_n=4）+ correctness valid（Table III）。

  **3. Asynchronous Pipeline（解决 C2 进阶: CUDA-Tensor Core 协调）**：
  Packing Kernel 中实现 register-level 异步流水线：第 i 个 tile 的 Tensor Core mma 与第 i+1 个 tile 的 ldmatrix + CUDA Core dequantization 重叠执行，持续 producer-consumer flow。Dequantization overhead 从 baseline 的 ~50% 降至 <15%（4-bit）和 <35%（2-bit）（Fig. 15）。

  **4. Residual Kernel with Unified Quantization（解决 C3: 通用性）**：
  基于 N_r 对齐的 KV cache partitioning 统一支持 tensor-wise 和 channel-wise 量化——沿 seq_len 维做 channel-wise，沿 hidden dim 维做 tensor-wise，均在 residual block 内执行。Warp-level reduction（__shfl_xor_sync + shared memory buffer）高效计算 scale/zp。支持 MHA/MQA/GQA 全 attention variant（通过 Query Transformation reshape）。

  **5. Architecture-specific Optimizations**：
  - Hopper：利用 STSM + wgmma_SS 指令对，dequantized 数据经 shared memory 直接供 Tensor Core 消费，wgmma 异步执行
  - Blackwell：利用原生 MXFP4/NVFP4 mma 指令，直接在 packed 4-bit 数据上做 GEMM，完全消除 dequantization

  全栈执行例子（BitDecoding on H100, 128K context, LLaMA-3.1-8B GQA decode）：
  - 算法pipeline：
    1. Prefill 后 KV Cache Partitioning：N_r = 8 × W_n × R（e.g., R=4 for 4-bit, W_n=4 → N_r=128），X_pack = X[:L - (L mod 128)]（量化+pack），X_res = X[L-128:]（FP16 residual）
    2. Per decode step:
       a. Query Transformation: Q [1, 4, 8] → [4, 8]（gq=4 for LLaMA-3.1-8B GQA hq=32, hkv=8）
       b. Packing Kernel: cp.async 异步加载 Q tile + K_pack/V_pack tiles + K_p/V_p params → ldmatrix 加载 packed data + lop3 75316420 remapping（CUDA Cores, 与上一 tile 的 mma 重叠）→ mma QK^T（Tensor Cores wgmma_SS, B from shared memory via STSM）→ Cooperative Softmax（cross-warp sTMP reduction）→ P → sAcc → ldmatrix reload → mma PV（Tensor Cores）→ output
       c. Residual Kernel: 若 res_len == N_r，将满的 residual block 量化+pack → 追加到 packed cache
    3. Decode 持续至 EOS
  - 系统框架：CUDA/PTX 手工 kernel 集成到 HuggingFace Transformers attention backend；与 FlashAttention-3 兼容（Hopper warp-specialized pipeline）；支持 paged attention
  - 编译框架：论文未明确说明（手工 CUDA kernel，无编译框架修改）
  - kernel调度：Packing Kernel 异步流水线（ldmatrix+Dequant [CUDA Cores] || mma [Tensor Cores]）；Cooperative Softmax（register/shared memory cross-warp sync）；Residual Kernel（ldmatrix→quantize→pack→write，fused 单 kernel）；Hopper 优化（STSM+wgmma_SS, TMA 异步数据加载）
  - 硬件架构：H100 GPU（80GB HBM, 989 TFLOPS Tensor Cores FP16, 60 TFLOPS CUDA Cores FP32）；Tensor Cores 利用率 ~19.66%（实测，受 dequantization 限制但远超 CUDA Core-only baseline 的 ~0%）

  核心创新总结：BitDecoding 不是简单地"把 dequantization 放到 CUDA Cores、matmul 放到 Tensor Cores"，而是通过 **layout induction（ldmatrix→quantize→ldmatrix→dequant→mma 的闭环对齐）** 和 **warp-level parallelism（多 warp dequantization + cooperative softmax + 异步流水线）** 两个系统级设计，使这种分工真正高效。这种设计对任意低比特位宽、任意量化粒度、任意 attention variant、任意 GPU 代数都是高效和通用的。
