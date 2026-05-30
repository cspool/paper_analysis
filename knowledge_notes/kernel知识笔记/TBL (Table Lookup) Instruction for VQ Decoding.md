## TBL (Table Lookup) Instruction for VQ Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TBL（Table Lookup）指令是移动 CPU（ARM 架构）的 SIMD 硬件指令，用于将索引值映射到查找表中的预存值。典型规范：5-6 bit 索引 → 8-bit 值（如 ARM NEON 的 `vtbl`/`vtbx` 指令系列）。在 GPTVQ 中，TBL 指令是 VQ 解码的核心算子——每个 VQ 维度需要一个 TBL 调用将 6-bit 质心索引映射到 8-bit signed integer 值。2D VQ 需要 2 条 TBL 指令（每维一条），结果相加后乘以 scale。TBL 指令将 16 个 8-bit 表项存于 128-bit NEON 寄存器中，单周期完成 16 路并行查表（one register = 16 × 8-bit = 128 bits，正好一个 64-entry codebook 需要 4 个 NEON 寄存器）。相比通用 gather/scatter 指令（如 SVE gather），TBL 指令延迟更低、吞吐更高，是移动端高效 VQ 解码的硬件基础。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// ARM NEON TBL-based 2D VQ Decode Kernel (伪代码)
// 输入: packed 6-bit indices, 64-entry LUT (8-bit signed), FP16 scale
// 输出: dequantized FP16 weights

// NEON 寄存器布局:
// v0-v3: 4 × 128-bit = 64 × 8-bit LUT entries
// v4: packed 6-bit indices (16 indices per cycle)

void vq_decode_2d_neon(
    const uint8_t* packed_indices,  // 6-bit packed
    const int8_t* lut_dim0,         // 64 entries × 8-bit
    const int8_t* lut_dim1,         // 64 entries × 8-bit
    const float16_t* scale,         // per-block FP16 scale
    float16_t* decoded_weights,     // output
    int num_weights
) {
    // 加载 LUT 到 NEON 寄存器 (64 entries = 4 × 128-bit regs)
    int8x16_t lut0_0 = vld1q_s8(lut_dim0);       // entries 0-15
    int8x16_t lut0_1 = vld1q_s8(lut_dim0 + 16);  // entries 16-31
    int8x16_t lut0_2 = vld1q_s8(lut_dim0 + 32);  // entries 32-47
    int8x16_t lut0_3 = vld1q_s8(lut_dim0 + 48);  // entries 48-63
    
    // 同理加载 dim1 LUT
    
    for (int i = 0; i < num_weights; i += 16) {
        // Step 1: 从 packed format 解包 16 个 6-bit indices
        uint8x16_t indices_packed = vld1q_u8(packed_indices + i*6/8);
        uint8x16_t idx = unpack_6bit(indices_packed);  // 解包
        
        // Step 2: TBL 查表（维度 0）
        // vtbl: 用 idx 的低 4-bit 选择寄存器，高 2-bit 选择表段
        int8x16_t val_dim0 = vtbx4_s8(
            vtbx4_s8(vtbl4_s8(lut0_0, idx_low), lut0_1, idx_low),
            lut0_2, lut0_3, idx_low
        );
        
        // Step 3: TBL 查表（维度 1）
        int8x16_t val_dim1 = /* 同理用 lut_dim1 查表 */;
        
        // Step 4: 合并两维 + 反量化
        int8x16_t val_sum = vaddq_s8(val_dim0, val_dim1);  // v1 + v2
        float16x8_t decoded = vcvtq_f16_s16(             // int8 → float16
            vmovl_s8(vget_low_s8(val_sum))
        );
        decoded = vmulq_f16(decoded, vdupq_n_f16(*scale));  // × scale
        
        vst1q_f16(decoded_weights + i, decoded);
    }
}
```

关键设计决策：
- 6-bit index 限制 codebook ≤ 64 entries（= 2^6），精确匹配 TBL 指令的寻址能力
- 2D VQ 每 weight 需 2 次 TBL 查表 + 1 次加法 + 1 次乘法
- Packed 6-bit 格式：16 个 weights × 6 bits = 96 bits，占 12 bytes（vs 16 × 8 bits = 16 bytes）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TBL 指令在 ARM NEON 中为 `vtbl`（索引在 0..n-1 范围内返回表值，超出返回 0）和 `vtbx`（索引超出范围保留目标寄存器原值）。x86 等效指令为 `pshufb`（SSSE3+）。在 GPTVQ 中，TBL 是推理引擎 C 代码中通过 NEON intrinsics（`vld1q_s8`, `vqtbl1q_s8` 等）调用的。Codebook 存为 INT8 格式（8-bit signed），对应 TBL 的 8-bit 输出。关键限制：(1) TBL 仅支持 5-6 bit index（取决于实现），因此 VQ codebook 必须 ≤ 64 entries；(2) 解包 6-bit indices 的 overhead 是 TBL 之外的额外开销；(3) TBL 的 128-bit 限制意味着更大 codebook 需要多次 TBL 调用。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---
