## INT4 Concatenation for SIMD Mobile Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT4 Concatenation是Squat论文在SIMD-based MKMP Multiplier中使用的技术，通过将两个4-bit权重值拼接存入单个16-bit寄存器，利用ARM CPU的`mla`指令（16-bit × 16-bit → 32-bit INT32累加器）在单指令内同时执行两个子字节乘加操作。传统方法将4-bit数据零扩展至8-bit（byte boundaries），浪费了一半的SIMD计算带宽。Concatenation技术将相邻行权重拼接后与共享激活值相乘，配合bit-shift和row-wise summation恢复正确结果。低比特优先策略（low-bit priority strategy）均匀利用位宽，最小化冗余零。该技术也可以推广到activation-activation的矩阵乘法中。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
INT4 Concatenation的寄存器级操作（ARM NEON）：
```
// 传统方法（baseline）：4-bit → 8-bit extension
// 每个4-bit权重占用8-bit寄存器空间，SIMD利用率50%
uint8x16_t w_4bit = vld1q_u8(w_ptr);  // 16个8-bit槽，只用了低4bit
// 浪费了16×4bit = 64bit的SIMD带宽

// INT4 Concatenation方法：
// w_row_i: [w_i0|w_i1|...|w_i7] (8个4-bit权重 in 32-bit)
// w_row_j: [w_j0|w_j1|...|w_j7] (相邻行, 8个4-bit权重 in 32-bit)
// concat: [w_i0|w_j0|w_i1|w_j1|...|w_i7|w_j7] (16个4-bit权重 in 64-bit)

// Step 1: 拼接加载
uint16x8_t w_concat;  // 8个16-bit槽 = 128-bit NEON寄存器
// w_concat[0] = (w_row_i[0] << 4) | w_row_j[0]  // low-bit priority
// w_concat[1] = (w_row_i[1] << 4) | w_row_j[1]
// ...

// Step 2: 激活值广播
uint16x8_t x_broadcast = vdupq_n_u16(x_shared);  // 同一激活值复制8份

// Step 3: 并行乘加 (mla: 16-bit × 16-bit → 32-bit累加)
int32x4_t acc_lo = vmull_s16(vget_low_s16(w_concat), vget_low_s16(x_broadcast));
int32x4_t acc_hi = vmlal_s16(acc_lo, vget_high_s16(w_concat), vget_high_s16(x_broadcast));

// Step 4: 内部拆分 (bit-shift恢复)
// acc中的每个32-bit值是 (w_i << 4 | w_j) * x_shared
// 需要shift来分离w_i*x_shared和w_j*x_shared
int32x4_t result_i = vshrq_n_s32(acc_hi, 4);  // 右移恢复高位结果
int32x4_t result_j = vandq_s32(acc_hi, mask_low4);  // mask提取低位结果
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
INT4 concatenation要求：(1) weight matrix的行数为偶数（相邻行配对拼接）；(2) 16-bit中间寄存器提供冗余（实际4-bit值远超需求但方便SIMD操作）；(3) low-bit priority策略确保位宽均匀利用。该技术理论上将4-bit GEMM的计算量减半（乘法和加法各减少50%），INT4 multiplier节省50% INT8 multiplier硬件资源。适用于所有支持16-bit乘法的ARM NEON处理器（ARMv7+）。也可应用于activation-activation矩阵乘法。限制：需确保16-bit乘法不溢出32-bit累加器（batch size或accumulation depth受限）。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

---
