## lop3-based Layout Remapping for Low-Bit Dequantization (75316420 pattern)

术语是什么？通过联网搜索让回答具体和精准。

lop3-based Layout Remapping 是 BitDecoding 中用于高效 INT4/INT2→FP16 dequantization 的 register-level 位操作技术。虽然 ldmatrix-based layout induction 保证 packed data 与 Tensor Cores 兼容，但 naive INT→FP16 static_cast 在 GPU 上极慢（Kim et al., 2022 指出 low-bit cast 是已知的性能瓶颈）。BitDecoding 的方案：ldmatrix 加载 packed INT16 到 register → cast 为 INT32 → 用 NVIDIA PTX 的 lop3 指令（arbitrary 3-input lookup table / bitwise logic）按 75316420 pattern 重新排布 bits → 使得后续 FP16 转换高效。75316420 是经验确定的 bit permutation pattern，将 packed INT4 values 重新映射为符合 Tensor Cores interleaved register layout 且利于 FP16 exponent field manipulation 的顺序。相比 Marlin 的 layout transform kernel（独立 kernel, prefill 58ms overhead），lop3 remapping 完全在 register 内完成，zero extra memory traffic。

从 kernel 调度角度拆解术语：

```
// === lop3-based Layout Remapping（在Packing Kernel dequant阶段） ===
// 输入：INT16 packed data（8个INT4值，或16个INT2值）
//       已通过ldmatrix加载到register

// Step 1: Cast INT16 → INT32
uint32_t packed = (uint32_t)ldmatrix_result[thread_idx];

// Step 2: lop3 bit permutation (75316420 pattern)
// lop3: PTX bitwise logic instruction with arbitrary 3-input truth table
// 将bits按75316420模式重新排列
// 效果：
//   原始bits: b7 b6 b5 b4 b3 b2 b1 b0 b15 b14 b13 b12 b11 b10 b9 b8 ...
//   → 重排后更利于后续提取和FP16转换
uint32_t remapped;
asm volatile("lop3.b32 %0, %1, 0, 0, 0x88;"  // 具体imm根据pattern确定
             : "=r"(remapped) : "r"(packed));

// Step 3: Extract 4-bit values and convert to FP16
// pattern保证extract的4-bit value在FP16 mantissa field位置
uint32_t fp16_bits = (remapped_val << 10) | 0x6400;  // exponent=1024
FP16 deq = __int2float_rn(fp16_bits) - 1024.0f;       // 高效INT→FP16

// Step 4: Apply scale and zero-point
FP16 result = deq * scale + zero_point;
```

lop3 指令是 SM70+(Volta) 开始支持的通用 PTX 位操作指令，接收 3 个 32-bit 输入和一个 8-bit 真值表（lookup table, 256 entries），按真值表对输入逐 bit 输出。BitDecoding 利用其表达能力在一两条指令内完成复杂的 bit permutation，避免多条 shift/mask/or 指令。

术语一般如何实现？如何使用？

lop3 remapping 作为 CUDA inline PTX assembly 嵌入 Packing Kernel。75316420 pattern 根据 mma variant 和 bit-width 在 compile time 确定（不是 runtime 计算）。对于不同 GPU 世代和不同 bit-width (4-bit/2-bit)，pattern 可能不同，由 BitDecoding 的 unified instruction configuration 自动选择。Blackwell 架构上 lop3 remapping 被绕过（因原生 mxfp4 mma 不需要 software dequantization）。参考实现：https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

