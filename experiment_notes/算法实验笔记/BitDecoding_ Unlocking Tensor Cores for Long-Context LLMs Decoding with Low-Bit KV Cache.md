## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 BitDecoding 系统，通过 cooperative use of Tensor Cores + CUDA Cores 实现低比特 KV cache 的高效解码。核心算法 pipeline 包括：(1) **Layout Induction 方法**：利用 ldmatrix 硬件指令的 thread-to-register 映射自动 induce Tensor Core compatible 低比特 packing layout——各线程在寄存器内量化和打包，保持 FP16 interleaved fragment layout，解量化后直接匹配 Tensor Core 寄存器，无需全局 reshape 或离线 layout transformation；(2) **Residual Block Size 对齐**：定义 residual block size N_r = P_n × W_n × R（R = ω/β 为 packing ratio），使低比特 KV cache fragment 精确对齐 Tensor Core warp-level tiling，饱和计算单元；(3) **75316420 Pattern Remapping**：基于 lop3 指令的 bitwise 操作，将 INT4/INT2 数据高效转换为 FP16，对齐 Tensor Core 的 interleaved 计算模式；(4) **Query Transformation**：将 [1, (gq, hkv)] reshape 为 [gq, hkv]，在 GQA/MQA 下饱满 Tensor Core tile；(5) **Warp Parallelism Strategy**：W_m=1（decode Q length 小），增加 W_n 提高 warps 并行度，SM warp scheduler 交替调度多个 warp 的 dequantization，避免 stall；(6) **Multi-level Memory Hierarchy Cooperative Softmax**：register→shared memory→register 的跨 warp reduction 和同步。实验比较：(1) kernel-level speedup vs FP16 FlashDecoding-v2、Kivi、QServe、Atom；(2) end-to-end 吞吐和延迟 vs Kivi、QServe；(3) 精度 trade-off（LongBench accuracy vs throughput）；(4) 各组件 ablation（layout induction、warp parallelism、pipeline optimization 的 speedup breakdown）；(5) 量化+打包延迟 overhead vs Marlin/Ladder。

- 硬件平台是什么，配置是什么。
  Blackwell (RTX 5090, RTX PRO 6000)：原生 MXFP4/NVFP4 低精度 Tensor Core。Hopper (H100 80GB)：WGMMA 指令 + warp-specialized pipeline + TMA。Ada (RTX 4090)：带宽受限。Ampere (A100 80GB)：高带宽。多 GPU：8×A100 for LLaMA-3.1-70B。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B (MHA)、LLaMA-3.1-8B (GQA)、LLaMA-3.1-70B (GQA)、Qwen3-8B (GQA)、Qwen3-14B (GQA)。精度 benchmark：LongBench（bilingual multi-task long-context understanding benchmark），评估平均准确率。Kernel benchmark：synthetic workloads 下不同 seq_len（最高 128K）、不同 batch_size、不同 attention variant（MHA/MQA/GQA）下的 latency 和 speedup。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/OpenBitSys/BitDecoding

  **算法 Pipeline 核心流程**：

  **Phase 1: KV Cache Partitioning（prefill 后）**
  ```
  输入：Prefill 后的 FP16 KV cache X ∈ R^{L×d}
  参数：bit_width β ∈ {2,4}, 量化粒度 granularity ∈ {tensor-wise, channel-wise}

  // 计算 packing ratio 和 residual block size
  R = 16 / β                    // ω=16 (INT16 pack), e.g., R=8 for 2-bit
  N_r = P_n × W_n × R           // P_n=8 (mma.m16n8k16), W_n: warp count along N

  // 分区
  N_p = L - (L mod N_r)          // 对齐 packed 部分长度
  res_len = L mod N_r            // residual 部分长度

  X_pack = X[:N_p]               // 将被量化+打包为低比特
  X_res  = X[N_p:]               // 保留 FP16 residual KV cache
  ```

  **Phase 2: Quantization & Packing（Residual Kernel）**
  ```
  // 对 X_pack 中每个 N_r 大小的 block 执行
  for each block of size N_r:
      // Step 1: ldmatrix 加载 FP16 KV tile（建立 interleaved layout）
      registers = ldmatrix(FP16_KV_block)

      // Step 2: 线程级 reduction（warp-level via __shfl_xor_sync）
      local_min, local_max = thread_level_reduction(registers)
      warp_min = __shfl_xor_sync(local_min)
      warp_max = __shfl_xor_sync(local_max)

      // Step 3: 计算 scale 和 zero-point
      if granularity == tensor-wise:
          scale = (warp_max - warp_min) / (2^β - 1)
          zero_point = round(-warp_min / scale)
      elif granularity == channel-wise:
          // 沿 seq_len 维（在 residual block 内）channel-wise 量化
          scale[d] = (max_d - min_d) / (2^β - 1)  // per channel
          zero_point[d] = round(-min_d / scale[d])

      // Step 4: 各线程在寄存器内量化+pack（保持 interleaved layout）
      for each thread's register values:
          quantized = clamp(round(fp16_val / scale) + zero_point, 0, 2^β-1)
          packed = pack 16/β 个 quantized values → INT16

      // Step 5: 写出到 low-bit KV cache
      K_pack[N_p:N_p+N_r] = packed_K
      V_pack[N_p:N_p+N_r] = packed_V
      K_params += {scale, zero_point}  // half2 格式存储
  ```

  **Phase 3: Autoregressive Decoding（Packing Kernel, 每生成 token 执行）**
  ```
  输入：Q ∈ R^{1×hq×d}, K_pack, V_pack, K_params, V_params, K_res, V_res

  // Step 1: Query Transformation（GQA/MQA）
  Q_reshaped = reshape(Q, [gq, hkv, d])  // gq = hq/hkv

  // Step 2: Packed KV Cache Attention（main body）
  for c in 0..ceil(N_p / T_n):           // T_n: KV tile size along N
      // 异步 Global→Shared Memory
      Q_tile = cp.async.cg(Q_reshaped) → SMEM
      K_tile_pack = cp.async.cg(K_pack[c*T_n:(c+1)*T_n]) → SMEM
      V_tile_pack = cp.async.cg(V_pack[c*T_n:(c+1)*T_n]) → SMEM
      K_tile_param = cp.async.ca(K_params[c*T_n:(c+1)*T_n]) → SMEM
      V_tile_param = cp.async.ca(V_params[c*T_n:(c+1)*T_n]) → SMEM

      // Pipeline: ldmatrix + dequant (CUDA Cores) overlap with mma (Tensor Cores)
      for each warp tile in K_tile_pack:
          // Stage A (CUDA Cores): Load & Dequant
          reg_K = ldmatrix(K_tile_pack[tile])     // 加载 packed INT16
          reg_Kp = ldmatrix(K_tile_param[tile])   // 加载 scale/zp (half2)
          reg_Kfp16 = lop3_75316420_remap(reg_K)  // bitwise remapping
          reg_Kfp16 = reg_Kfp16 * reg_Kp.scale + reg_Kp.zp  // dequant

          // Stage B (Tensor Cores): Matmul（与下一个 tile 的 Stage A 重叠）
          S = mma(Q_tile, reg_Kfp16)              // T_m × T_n

          // Cooperative Softmax (cross-warp)
          m_new = max(m_old, rowmax_warp_reduce(S, sTMP))
          P = exp(S - m_new)                      // element-wise (CUDA Cores)
          sAcc[tile] = P                           // store to SMEM
          P_aligned = ldmatrix(sAcc[tile])         // reload for MMA alignment

          reg_Vfp16 = ldmatrix(V_tile_pack[tile]) → dequant
          O_new = mma(P_aligned, reg_Vfp16) + diag(exp(m_old - m_new)) @ O_old
          O_old, m_old = O_new, m_new              // online update

  // Step 3: Residual KV Cache Attention（标准 FlashAttention）
  O += FlashAttention(Q_reshaped, K_res, V_res)

  // Step 4: 更新 Residual KV Cache
  K_res = concat(K_res, new_K_token)
  V_res = concat(V_res, new_V_token)
  if len(K_res) == N_r:
      // 触发 Residual Kernel：量化满的 residual block → packed cache
      quantize_and_pack(K_res, V_res) → append to K_pack, V_pack
      K_res, V_res = [], []  // 清空 residual
  ```

  **张量计算细节**：

  布局对齐数学表达：
  $$X = X_{\text{pack}} \cup X_{\text{res}}, \quad X_{\text{pack}} = X[:L-N_r], \quad X_{\text{res}} = X[L-N_r:]$$
  $$N_r = P_n \times W_n \times R, \quad R = \omega / \beta$$

  Bank conflict-free shared memory layout：
  $$\text{col}_{id} = \text{row}_{id} \oplus \text{col}_{id}$$

  Cooperative Softmax（Algorithm 1）：
  $$S_i = Q_i K_j^T, \quad S_i \in \mathbb{R}^{T_m \times T_n}$$
  $$m_i^{new} = \max(m_i, \text{rowmax}(S_i, sTMP))$$
  $$P_i = \exp(S_i - m_i^{new}), \quad P_i \in \mathbb{R}^{T_m \times T_n}$$
  $$sAcc = \text{tiled\_copy\_r2s}(P_i)$$
  $$P_i' = \text{tiled\_copy\_s2r}(sAcc)$$
  $$O_i^{new} = P_i' V_j + \text{diag}(e^{m_i - m_i^{new}}) O_i$$

  Hopper WGMMA 关键路径（PTX 级别）：
  ```
  ldmatrix → dequant（CUDA Cores）
  STSM → shared memory（存储 dequantized FP16）
  wgmma_SS → Tensor Cores（B matrix from shared memory）
  // STSM 和 wgmma 利用 Hopper 异步执行重叠
  ```

  **关键性能与精度 trade-off（Table I）**：
  | KV Cache | Throughput      | LongBench Acc |
  |----------|-----------------|---------------|
  | FP16     | 49.25 tok/s     | 48.25         |
  | INT4     | 147.21 (2.98×)  | 48.16 (-0.2%) |
  | INT2     | 209.48 (4.25×)  | 47.38 (-2.7%) |

  **量化+打包 Overhead（Table II）**：
  | Inference Phase | Marlin | Ladder | BitDecoding |
  |-----------------|--------|--------|-------------|
  | Prefill         | 58.02ms| 4.79ms | 0.0599ms    |
  | Decode          | 0.41ms | 0.65ms | 0.008ms     |

  **Multi-warp Ablation（Table III）**：
  W_n=1 → TC utilization 10.91%, latency 3.746ms（低效）
  W_n=4 + Coop Softmax → TC utilization 19.66%, latency 0.613ms（高效，correctness valid）
