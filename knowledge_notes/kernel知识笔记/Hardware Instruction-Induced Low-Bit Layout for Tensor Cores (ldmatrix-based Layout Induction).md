## Hardware Instruction-Induced Low-Bit Layout for Tensor Cores (ldmatrix-based Layout Induction)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layout Induction 是 BitDecoding 提出的利用 ldmatrix PTX 指令的 thread-to-register 映射自动为低比特（INT4/INT2）量化数据生成 Tensor Cores 兼容 packed layout 的方法。核心洞察：ldmatrix 从 shared memory 加载数据到 register 时，按 Tensor Cores fragment 的 interleaved pattern 分布到各线程。如果每个线程在其本地 registers 内完成 quantization + packing，那么写回的低比特 packed data 隐式保留了 FP16 interleaved layout——在解量化时无需全局 reshape，直接匹配 TC 寄存器期望。这避免了 Marlin（离线 layout transformation kernel）和 Ladder（迭代搜索）的大量预处理开销。

从kernel调度角度拆解术语。

```
// Residual Kernel: ldmatrix → compute → quantize → pack → store
// 输入: FP16 K/V tile，输出: packed INT16 K/V（layout-compatible）

Step 1: ldmatrix 加载 FP16 tile → registers（自动 interleaved layout）
  regs[0:7] = ldmatrix.sync.aligned.m16n8k16.shared.b16(K_tile_smem)
  // 每个线程持有 8 个 FP16 值，遵循 mma.m16n8k16 fragment mapping

Step 2: MMA computation（可选，如 QK^T 或 P V）
  accum = mma.sync.aligned.m16n8k16(Q_reg, K_reg, accum)

Step 3: 线程内量化（保持 interleaved layout）
  for each thread's 8 values:
      local_min = min(regs)
      local_max = max(regs)
  warp_min = __shfl_xor_sync(local_min)   // warp-level reduction
  warp_max = __shfl_xor_sync(local_max)
  scale = (warp_max - warp_min) / (2^β - 1)
  zero_point = round(-warp_min / scale)
  for each thread's values:
      q = clamp(round(fp16_val / scale) + zero_point, 0, 2^β-1)

Step 4: Pack to INT16（layout preserved）
  packed = pack R quantized values → INT16  // R = 16/β
  store packed to global memory (K_pack / V_pack)

// Packing Kernel: ldmatrix → dequant → mma（对称的逆过程）
// 使用相同 ldmatrix/mma 配置 → dequantized 值自动对齐 TC fragment
```

术语一般如何实现？如何使用？

实现在 BitDecoding 的 Residual Kernel 和 Packing Kernel 中（~300 行 CUDA PTX）。关键要求：Residual Kernel 和 Packing Kernel 必须使用相同的 ldmatrix variant、相同的 mma variant、相同的 warp tiling 配置。比 Marlin（离线预处理：prefill 58ms, decode 0.41ms）和 Ladder（4.79ms, 0.65ms）快 3 个数量级——BitDecoding 仅 0.06ms (prefill) 和 0.008ms (decode)。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
