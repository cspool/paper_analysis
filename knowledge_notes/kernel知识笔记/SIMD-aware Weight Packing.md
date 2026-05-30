## SIMD-aware Weight Packing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SIMD-aware Weight Packing 是 TinyChat 针对 CPU SIMD 架构（ARM NEON、x86 AVX）提出的 INT4 权重排列优化策略。传统顺序排列（w0, w1, ..., w31）下，每个 4-bit 权重的解包需要 3 条标量指令（shift + AND + FMA scaling），32 个权重共 96 条标量指令。SIMD-aware packing 将权重重新排列为交错顺序，使得单条 SIMD 指令可并行解包整个寄存器宽度的权重。以 ARM NEON 128-bit 为例：将 32 个 4-bit 权重排列为 (w0, w16, w1, w17, ..., w15, w31)，一个 128-bit 寄存器可同时解包全部 32 个权重，仅需 3 条 SIMD 指令（AND 提取低位、shift+AND 提取高位、FMA scaling）。通用规则：对于 2^n-bit SIMD 寄存器，相邻权重的索引差为 `1/8 × 2^n`，因为每个寄存器可存 `1/8 × 2^n` 个 8-bit 解包后的权重。GPU 端采用不同排布：每 8 个权重打包为 (w0, w2, w4, w6, w1, w3, w5, w7) 顺序（参照 Kim et al., 2022）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ARM NEON 128-bit SIMD-aware unpacking 的伪代码：
```c
// 传统顺序 packing: [w0|w1|w2|w3|...|w31] (每 4-bit 一个 weight)
// 每个权重需要独立的标量指令解包，无 SIMD 利用

// SIMD-aware packing: [w0|w16|w1|w17|w2|w18|...|w15|w31]
// 128-bit 寄存器 = 32 个 4-bit weight indices

// 解包过程（3 条 NEON SIMD 指令）：
uint8x16_t packed = vld1q_u8(packed_weights_ptr);  // 加载 128-bit

// Step 1: 提取低 4-bit (w0, w16 的低 4bit, w1, w17 的低 4bit, ...)
uint8x16_t low_nibbles = vandq_u8(packed, vdupq_n_u8(0x0F));

// Step 2: 提取高 4-bit (w0, w16 的高 4bit, w1, w17 的高 4bit, ...)
uint8x16_t high_nibbles = vshrq_n_u8(packed, 4);

// Step 3: 查表 LUT + FMA scaling
// 将解包后的 indices 用作 LUT 索引，查得 FP16 值后乘以 scale
// 使用 NEON FMA 指令一次性完成乘加

// 对比标量方法: 32 weights × 3 instructions = 96 instructions
// SIMD 方法: 3 SIMD instructions, 理论加速 32×
```
实际收益：ARM CPU 上 SIMD-aware packing 额外提供 ~1.2× 加速（相比未做 SIMD-aware packing 的 bulk dequantization）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 量化后的 INT4 权重在保存时按目标平台的 SIMD 宽度进行重排（offline 预处理）；(2) 不同平台使用不同的 packing layout——ARM NEON 128-bit 用 32-weight 交错，AVX 256-bit 用 64-weight 交错，GPU 用 8-weight 交错；(3) 推理时直接按 packed layout 加载，用对应 SIMD intrinsic 解包。TinyChat 在 CPU 后端（C++ 实现）中针对 ARM NEON 和 x86 AVX 分别实现了 SIMD-aware packing 和解包 kernel。代码：https://github.com/mit-han-lab/llm-awq。这种技术的通用性：任何需要在 CPU SIMD 上执行低比特推理的 weight-only 量化方案（GGUF/Q4_0, GPTQ, AWQ 等）都可受益。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

在 Squat 中，INT4 Concatenation 是 SIMD-aware packing 的扩展——将相邻行4-bit权重拼接入16-bit寄存器（不零扩展到8-bit），利用 ARM `mla` 指令（16-bit×16-bit→32-bit累加器）在单指令完成两个子字节乘加。与 AWQ 的交错排列不同，Squat 的拼接策略专注于最大化 SIMD 寄存器的位宽利用率（100% vs 零扩展方案的50%），理论计算量减半。配合 bit-shift + row-wise summation 恢复正确结果。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT Dequantization（查找表反量化）是一种将量化权重从低比特格式恢复到高精度格式的 GPU kernel 实现技术。当量化格式的值不是均匀分布（如 NormalFloat、FP 格式），无法通过简单的代数公式（如 `w_fp16 = scale * (w_int - zero_point)`）完成反量化时，需要使用预先存储的查找表（LUT）将每个量化索引映射到对应的浮点值。AFPQ 论文在推理系统中使用 LUT 来完成 NF4/NF3 值到 FP16 的转换：NF 格式的 16 个（NF4）或 8 个（NF3）候选值预先存储在 GPU 的 constant memory 或 register 中，反量化时通过量化索引查表得到对应的 FP16 值。LUT 之后，再用 scale_pos/scale_neg 进行非对称缩放得到最终 FP16 权重。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AFPQ 论文中 NF4-asym LUT dequantization kernel 的执行过程：
```
# GPU Kernel: NF4-asym Dequantization
# 输入: packed_4bit_weights (byte array), scales_pos (FP16), scales_neg (FP16)
# 常量: NF4_LUT[16] = [-1, -0.6962, ..., 0.7230, 1.0]  # 16 个 FP16 值

# Constant Memory 中预存 LUT
__constant__ half NF4_LUT[16] = {...};

__global__ void dequant_nf4_asym_kernel(
    uint8_t* packed_w,    // 每 2 个 NF4 → 1 byte
    half* scales_pos,     // FP16, 每组一个
    half* scales_neg,     // FP16, 每组一个
    half* w_fp16,         // 输出: 反量化后的 FP16 权重
    int group_size,       // = 128
    int num_groups
) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    
    // 加载当前 group 的 scale
    half s_pos = scales_pos[group_id];
    half s_neg = scales_neg[group_id];
    
    // 每个线程处理多个元素
    for (int i = tid; i < group_size; i += blockDim.x) {
        int byte_idx = group_id * (group_size / 2) + i / 2;
        uint8_t byte = packed_w[byte_idx];
        
        // 提取两个 NF4 索引
        uint8_t idx;
        if (i % 2 == 0) idx = byte & 0x0F;
        else            idx = (byte >> 4) & 0x0F;
        
        // LUT 查找
        half val = NF4_LUT[idx];
        
        // 非对称反量化
        if (val > 0)       w_fp16[...] = s_pos * val;
        else if (val < 0)  w_fp16[...] = s_neg * val;
        else               w_fp16[...] = 0;
    }
}
```
关键设计：(1) LUT 存在 constant memory 中（所有线程同时读取同一地址时零延迟）；(2) packed 4-bit 格式每 byte 存 2 个权重，减少显存带宽；(3) 非对称 scale 的 branch 基于 val 的符号（非判断 weight 原始符号），避免额外存储符号 bit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LUT dequantization 的实现方式：(1) CUDA constant memory——适合 LUT 大小 ≤ 64KB（NF4 LUT 仅 32 bytes）；(2) Shared memory——当 LUT 较大或不同 warp 需不同 LUT 时使用；(3) Register——极小的 LUT（如 NF3 的 8 个值）可展开到寄存器。在 AFPQ 的 FasterTransformer 集成中，NF4-asym kernel 替换了原有的 INT4 dequant kernel（后者通过 `w_fp16 = scale * (w_int - zero_point)` 的代数计算完成，无需 LUT）。LUT 方法的局限性：(1) 适用于非均匀量化格式（NF、FP），但比 INT 的代数反量化多一次 memory read；(2) 可能增加 register pressure（如果 LUT 被编译器展开到寄存器）。AFPQ 论文观察到 NF4-asym 推理延迟（265ms）高于 INT4（174ms），部分由 LUT 和额外的 scale branch 导致，并指出可以通过 kernel 优化缩小差距。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs

---
