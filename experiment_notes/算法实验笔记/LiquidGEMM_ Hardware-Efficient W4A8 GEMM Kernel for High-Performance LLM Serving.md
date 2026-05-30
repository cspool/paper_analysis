## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是LiquidQuant (LQQ)，一种硬件高效的两级W4A8量化算法。LQQ通过rotation-based transformation将INT8值映射到UINT8域再量化为UINT4，配合two's complement特性设计dequantization，仅需两条32-bit硬件指令（IMAD + XOR）处理四个元素，避免溢出。离线量化流程：FP16 → 第一级per-channel量化到INT8（限制在[-119, 119]范围）→ 第二级shift到UINT8域 → per-group量化到UINT4（group size=64）。激活量化动态使用SmoothQuant per-token量化。实验比较的算法baseline包括：QServe（W4A8，group size=128）、TRT-W4A16、TRT-W8A8、TRT-FP8、TRT-FP16。准确率评估使用WikiText2 perplexity、PIQA/ARC/HellaSwag/WinoGrande zero-shot accuracy。性能评估通过系统级吞吐量和kernel级延迟。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU（80GB HBM），Intel Xeon Platinum 8457C CPU，2.9TB RAM。软件：PyTorch 2.4.0，CUDA 12.4。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA（30B, 7B/2 13B/2 70B），LLaMA3-8B，Mistral-7B，Mixtral-8×7B，Yi-34B。数据集：WikiText2（perplexity），PIQA/ARC/HellaSwag/WinoGrande（zero-shot accuracy）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：LiquidGEMM未提供开源代码。论文说明LiquidGEMM已部署为ByteDance Seed生产LLM serving基础设施的主要GEMM kernel。baseline系统QServe开源（https://github.com/mit-han-lab/omniserve），TensorRT-LLM开源（https://github.com/NVIDIA/TensorRT-LLM）。

  算法pipeline（两级量化+dequantization）：

  离线量化阶段：
  ```
  // Step 1: FP16 → INT8 (per-channel, 第一级量化)
  W_fp16: shape [K, N]
  for c in range(N):  // per-channel
    s_i8[c] = (max(W_fp16[:,c]) - min(W_fp16[:,c])) / (119 - (-119))
    Q_i8[:,c] = clamp(round(W_fp16[:,c] / s_i8[c]), -119, 119)

  // Step 2: INT8 → UINT4 (per-group, 第二级量化, LiquidQuant)
  group_size = 64
  for c in range(N):
    for g in range(0, K, group_size):
      group = Q_i8[g:g+group_size, c]
      min_val = min(group)  // 负数
      Q_u8 = group - min_val  // shift到UINT8域 [0, max-min]
      s_u8 = round(max(Q_u8) / 15)  // scale factor, ≤16
      Q_u4 = round(Q_u8 / s_u8)  // UINT4 [0,15]

  // Precompute dequantization constant
  a = 128 + min(Q_i8)  // precomputed per-group, ∈ [0,255]
  ```

  在线dequantization（kernel main-loop内）：
  ```
  // Dequantize four UINT4 elements using two hardware instructions:
  // Input: packed UINT4 in 32-bit register reg_in
  //         s_u8: per-group scale factor (broadcast to 32-bit)
  //         a: precomputed offset (128+min, broadcast to 32-bit)

  // Unpack: expand 8 × 4-bit → 2 × 32-bit registers (QServe method)
  reg_lo = unpack_low_4bits(reg_in)
  reg_hi = unpack_high_4bits(reg_in)

  // Dequantization with two instructions per 4 elements (Equation 12):
  // Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80
  result_lo = IMAD(reg_lo, s_u8, a)   // multiply-add, 1 instruction
  result_lo = XOR(result_lo, 0x80)    // flip MSB, 1 instruction

  result_hi = IMAD(reg_hi, s_u8, a)
  result_hi = XOR(result_hi, 0x80)

  // First-level dequantization in epilogue:
  // W_fp16 ≈ Q_i8 * s_i8 (back to FP16)
  ```

  关键特性：LQQ利用two's complement同余性质（i ≡ j mod 2^8 → 相同二进制表示）消除溢出。dequantization全部在UINT8域内计算，确保中间结果∈[0,255]。XOR 0x80等价于条件性地加/减128，将UINT8结果映射回INT8的二进制表示，可直接用于Tensor Core MMA。
