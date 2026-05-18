## LUT-based Dequantization and Softmax on NPU Vector Units

术语是什么？通过联网搜索让回答具体和精准。

LUT-based computation在NPU上使用向量单元的查表指令替代复杂数学计算，克服NPU向量单元通用计算能力弱、缺乏专用数学函数单元的瓶颈。论文两项LUT技术：(1) LUT dequantization：HVX的vlut16指令将4-bit INT4直接查表映射为FP16，避免传统mask-unpack-convert多指令序列；(2) LUT Softmax：HVX的vgather指令从预计算64KiB FP16 exp LUT收集值，替代多项式exp2展开，消除VLIW顺序依赖并减少指令数。两者利用NPU向量单元的硬件LUT指令以低延迟完成原本昂贵计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**LUT Softmax (safe softmax, FP16):**
```
// 预计算: 64KiB exp LUT in TCM, 仅x≤0的32768 entries
// LUT[half_bits] = exp(FP16_val) for FP16_val ≤ 0
// safe softmax: S - rowmax ensures all inputs ≤ 0

LUT_Exp(S_sub_max):  // all ≤ 0
    off = (S_sub_max << 1) & 0xFFFE   // 忽略MSB (sign=1), left-shift for 2B offset
    return vgather(LUT_base + off)     // 一次收集64个FP16

// FlashAttention on-chip softmax (Algorithm 1):
for KV_tile j:
    S_i = HMX_MatMul(Q_i, K_j^T, Acc=FP32)  // [Bq, Bkv] FP16
    m_new = max(m_old, HVX_rowmax(S_i))
    P_i = LUT_Exp(S_i - m_new)               // vgather
    l_new = exp(m_old-m_new)*l_old + HVX_rowsum(P_i, Acc=FP32)
    O_i = diag(exp(m_old-m_new))*O_old + HMX_MatMul(P_i, V_j)
```

LUT Softmax vs F32 exp: 1.26-2.19× speedup; vs F16 exp: up to 1.60×。

**LUT Dequantization (vlut16):**
```
// vlut16: 每个8-bit index查16-entry table, 输出16-bit; 生成一对128B registers
LUT_INT4_TO_FP16 = [FP16(-8),..., FP16(7)]  // 16 entries, 32B
// Scale广播: 将4组scales放LUT content, constant indices查表
// 一次vlut16完成4组scale广播到全register

dequant(W_q, W_s):
    W_fp16 = vlut16(W_q_high, LUT4_TO_FP16)  // INT4→FP16, 无qfloat overhead
    scales = vlut16(indices, scale_LUT)        // broadcast
    return W_fp16 * scales
```

vlut16直接产生IEEE-754 FP16 (vs HVX默认qfloat格式需额外转换)，进一步减少指令。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LUT在系统初始化时预计算（64KiB exp LUT占TCM 0.8%），无运行时overhead。通用性：(1) LUT dequant支持任意4-bit编码(FP4, NF4, IQ4_NL)，仅换table content；(2) LUT Softmax依赖safe softmax保证输入≤0，否则需128KiB全范围LUT；(3) vgather延迟较高(24-48 VLIW packets on V75)，需与无关指令交错编排；(4) 类似思想可用于ARM SVE/SSVE、x86 AVX-512等有LUT指令的向量ISA。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

