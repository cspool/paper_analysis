## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了三个核心 GPU kernel：(1) **Residual Kernel**：fused computation + quantization + packing，将新生成 token 的 FP16 KV tensor 通过 ldmatrix 加载到寄存器，在 Tensor Core fragment 上执行 mma 后，各线程在寄存器内完成量化（tensor-wise 或 channel-wise）和 INT16 bit-packing，输出 interleaved layout-compatible 低比特数据到 global memory；(2) **Packing Kernel**：fused dequantization + Tensor Core GEMM，用与 Residual Kernel 相同的 ldmatrix/mma 配置加载 packed 低比特数据，经 lop3 指令高效 dequantization（75316420 pattern remapping），随后送入 Tensor Core mma，并实现了 CUDA Core 与 Tensor Core 的 register-level 异步流水线（ldmatrix+Dequant 与 mma 重叠）；(3) **Multi-warp Cooperative Softmax**：多 warp 并行 softmax，用 shared memory buffer（sTMP, sAcc）做 cross-warp reduction 和跨 warp 同步，在 Hopper 上利用 WGMMA 直接从 shared memory 消费数据。实验比较：(1) kernel-level：vs FP16 FlashDecoding-v2（speedup baseline）、Kivi（non-fused low-bit）、QServe/Atom（CUDA Core-only fused）、FlashDecoding-v3（Hopper optimized），在 Single/Batches/Page 三种 setting 下，跨 Blackwell(H100 equiv. RTX 5090/RTX PRO 6000)、Hopper(H100)、Ada(RTX 4090)、Ampere(A100) 四代 GPU；(2) end-to-end：vs Kivi 和 QServe 在 LLaMA-2-7B、LLaMA-3.1-8B/70B、Qwen3-8B/14B 上的解码延迟和吞吐。

- 后端平台是什么，配置是什么。
  - Blackwell: RTX 5090、RTX PRO 6000（原生支持 MXFP4/NVFP4 低精度 Tensor Core，消除 dequantization 开销）
  - Hopper: NVIDIA H100（80GB HBM，支持 WGMMA 指令、warp-specialized pipeline、TMA 异步拷贝）
  - Ada: NVIDIA RTX 4090（带宽受限 GPU）
  - Ampere: NVIDIA A100（80GB HBM，高带宽 GPU）。多 GPU 实验：8×A100 用于 LLaMA-3.1-70B
  - 所有 kernel 使用 CUDA + PTX 内联汇编编写，基准 kernel（FlashDecoding-v2/v3、QServe、Kivi）对比评估

- 评估性能的软件/脚本是什么。修改了什么。
  基于 CUDA/PTX 手工编写的自定义 attention kernel，集成到 PyTorch 推理 pipeline 中作为 attention 后端。核心修改/实现：

  1. **Residual Kernel（量化+打包）**：
  - 输入：prefill 后的 FP16 KV tensor 和新生成 token 的 FP16 K/V
  - 过程：ldmatrix 加载 FP16 KV tile → Tensor Core mma 执行 QK^T 或 PV → 线程级 min/max reduction（`__shfl_xor_sync` 做 warp-level reduction）→ 计算 scale/zero-point → 各线程在寄存器内量化并 pack 为 INT16 → 写出到 low-bit KV cache global memory
  - 关键优化：利用 ldmatrix 建立的 interleaved register layout，量化后自动保持 Tensor Core compatible layout，无需额外 layout transformation；scale/zero-point 存储为 compact half2 格式

  2. **Packing Kernel（去量化+GEMM）**：
  - 输入：Low-bit packed KV cache（K_pack, V_pack）、FP16 Q tile、量化参数（K_p, V_p）
  - 异步数据移动：Q 和 K_pack/V_pack 用 `cp.async.cg` 从 global→shared memory；K_p/V_p 用 `cp.async.ca` 做 byte-aligned 细粒度 copy；Hopper 上用 `tma.copy` 替代
  - Shared→Register：ldmatrix 加载 K_pack/V_pack 到 Tensor Core register layout；用 sizzling scheme（col_id = row_id ^ col_id）消除 bank conflict
  - Dequantization：lop3 指令执行 75316420 pattern bitwise 映射，高效转换 INT4/INT2→FP16；对齐 Tensor Core 期望的 interleaved fragment layout
  - 异步流水线：第 i 个 tile 在 Tensor Core 上执行 mma 的同时，第 i+1 个 tile 通过 ldmatrix 加载并 dequantize（CUDA Core）——CUDA Core 和 Tensor Core 重叠执行
  - 输出：FP16 attention output O tile

  3. **Multi-warp Cooperative Softmax（Algorithm 1）**：
  - W_m=1（解码 Q length 小），增加 W_n 提高并行度
  - sTMP ∈ R^{W_n}：cross-warp reduction 计算 row-wise max（先 intra-warp register reduction，再 inter-warp shared memory reduction）
  - sAcc ∈ R^{T_m×T_n}：暂存 Tensor Core 寄存器中的 attention scores P，通过 ldmatrix 重载确保 MMA alignment
  - sTMP 和 sAcc 复用同一 shared memory 指针（因 W_n 小）

  4. **Hopper 优化**：
  - 利用 `STSM` PTX 指令将 dequantized FP16 值写入 shared memory
  - 利用 `wgmma_SS` 指令（B 矩阵在 shared memory）执行 Tensor Core GEMM
  - 异步特性使存储与计算重叠

  5. **Blackwell 优化**：
  - 使用原生 MXFP4/NVFP4 mma 指令，直接在 packed 4-bit 数据上执行 GEMM
  - 跳过 lop3-based register remapping（无需显式 dequantization）
  - Block-scaling factor 布局由 Sect. IV-A 的 layout-agnostic 方法自动对齐

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/OpenBitSys/BitDecoding

  **评估原理与 Kernel 执行全流程（以单层 GQA attention decode step 为例）**：

  ```
  输入：
    Q ∈ R^{1×hq×d}（当前 token 的 query，decode 阶段 seq_len=1）
    K_pack ∈ R^{L×hkv×d_packed}（低比特 packed Key cache，含 scale/zp metadata）
    V_pack ∈ R^{L×hkv×d_packed}（低比特 packed Value cache）
    K_res, V_res ∈ R^{res_len×hkv×d}（FP16 residual KV cache）
    量化参数：bit_width β ∈ {2,4}, group_size（channel-wise 时）

  输出：
    O ∈ R^{1×hq×d}（当前 token 的 attention output）

  Step 1: Query Transformation
    // 将 Q 从 [1, gq, hkv] reshape 为 [gq, hkv]（gq = hq/hkv）
    // 在 GQA 下饱满 Tensor Core tile，提高 warp occupancy

  Step 2: Residual Kernel（可选，当 res_len == N_r 时触发）
    // 将 residual KV cache 中满 N_r 个 token 量化并迁移到 packed cache
    ldmatrix 加载 FP16 residual K/V → 线程内 min/max reduction
    → __shfl_xor_sync warp-level reduction → scale/zp 计算
    → 各线程在寄存器内 quantize + pack → 写入 K_pack/V_pack global memory
    // 每 decode step 仅当 res_len == N_r 时触发一次

  Step 3: Packing Kernel（主要 decode kernel）
    // Block-wise tiling: Q tile T_m, KV tile T_n
    for c = 0 to ceil(L/T_n):
        // --- Asynchronous Global→Shared Memory ---
        cp.async.cg: Q tile, K_pack[c], V_pack[c] → SMEM
        cp.async.ca: K_p[c], V_p[c]（量化参数）→ SMEM

        // --- Pipeline Stage 1: Load + Dequant (CUDA Cores) ---
        ldmatrix: K_pack[c] → registers (interleaved Tensor Core layout)
        ldmatrix: K_p[c]   → registers (量化 scale/zp)
        // lop3 bitwise remapping (INT4/INT2 → FP16): 75316420 pattern
        K_fp16 = dequant(K_pack[c], K_p[c])

        // --- Pipeline Stage 2: MMA (Tensor Cores) ---
        // 与下一个 tile 的 ldmatrix+dequant 重叠
        S = Q_fp16 @ K_fp16^T            // Tensor Core mma, output T_m × T_n
        // Cooperative Softmax (cross-warp via sTMP, sAcc)
        m_new = max(m_old, rowmax(S))    // sTMP cross-warp reduction
        P = exp(S - m_new)               // element-wise on CUDA Cores
        sAcc = P                          // store to SMEM for re-alignment
        P' = ldmatrix(sAcc)               // reload for MMA alignment
        O_new = P' @ V_fp16 + diag(exp(m_old - m_new)) @ O_old
        // (Hopper: wgmma_SS 直接从 sAcc shared memory 消费 P')

    // --- Residual KV Cache Attention ---
    // 对 res_len 个 FP16 residual token 执行标准 FlashAttention
    O += FlashAttention(Q, K_res, V_res)

  输出 O 到下一层 Transformer
  ```

  **评估指标与原理**：
  - Kernel-level speedup：以 FP16 FlashDecoding-v2 latency 为基准，normalized speedup = T_baseline / T_method
  - 三种 setting：
    - Single：batch_size=1，模拟边缘长上下文推理
    - Batches：大 batch_size + simple padding
    - Page：大 batch_size + paged attention memory management
  - Nsight Compute profiling：分析 dequantization overhead、Tensor Core utilization、memory throughput
  - End-to-end：HuggingFace Transformers 中替换 attention backend，测量 token/s 吞吐和逐 token 延迟

  **关键性能数据**：
  - Blackwell (RTX 5090, NVFP4): up to 8.6× vs FP16 FlashDecoding-v2, up to 4.3× vs QServe
  - Hopper (H100): up to 8.0× (v3 with wgmma), 4.1× (v2)
  - Ada (RTX 4090): ~4× (4-bit), ~7× (2-bit)
  - Ampere (A100): up to 3×; Kivi 和 QServe 在 A100 上甚至比 FP16 baseline 更差
  - End-to-end (LLaMA-3.1-8B, 128K): 3× latency reduction; >4× higher throughput than QServe
  - Dequantization overhead: BitDecoding <15% (4-bit), <35% (2-bit) vs Atom/QServe ~50%
