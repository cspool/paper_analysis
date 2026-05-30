## XNOR and Bit-count Operations（同或与位计数运算，1-bit 卷积硬件实现）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XNOR 和 bit-count（popcount）是二值化神经网络（BNN）中将浮点卷积乘加运算（MAC）替换为纯位操作的核心计算原语。当权重和激活都被二值化为 {+1, -1}（通常映射为 {1, 0}）后，卷积中的逐元素乘法退化为同或（XNOR）运算：`w=+1, a=+1 → +1·+1=+1, XNOR(1,1)=1`；`w=+1, a=-1 → +1·-1=-1, XNOR(1,0)=0`。即 XNOR 输出 1 时等价于乘法结果 +1，输出 0 时等价于 -1。累加操作退化为位计数（popcount）：统计 XNOR 结果中 1 的个数（得 +1 计数），最终求和 = `2*popcount(XNOR(x_b, w_b)) - n`。硬件上，XNOR 为单比特门操作（vs FP16 MAC 需多周期浮点单元），popcount 可用专用指令（x86 POPCNT, CUDA __popc）或 LUT 高效实现。理论加速比：32x 内存节省（32-bit → 1-bit），64x 能量节省（浮点乘加 → 位操作）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 BI-DiffSR 的二值化卷积 kernel 概念为例（论文未给出定制 CUDA kernel，以下为通用 1-bit Conv 设计）：
```cuda
// 1-bit 卷积 kernel 概念: bit-packed GEMM
// packed 格式: 每 byte 存 8 个 1-bit 值 (1=+1, 0=-1)
__global__ void binarized_conv2d(
    const uint8_t* packed_act,    // 1-bit 激活, packed
    const uint8_t* packed_w,      // 1-bit 权重, packed
    int32_t* output,              // int32 累加结果
    int H, int W, int C_in_pack, int K,
    float w_scale                  // 权重缩放因子 ||w||_1/n
) {
    int h_idx = blockIdx.y, w_idx = blockIdx.x;
    int c_out = threadIdx.x;
    int accum = 0;
    
    for (int cp = 0; cp < C_in_pack; cp++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                uint8_t act_byte = packed_act[h_idx][w_idx][cp];
                uint8_t w_byte = packed_w[c_out][cp][kh][kw];
                
                // XNOR: 按位同或, ~(a ^ b)
                uint8_t xnor_result = ~(act_byte ^ w_byte);
                
                // popcount: CUDA intrinsic __popc (32-bit)
                int ones = __popc((uint32_t)xnor_result);
                accum += 2 * ones - 8;  // ones - (8-ones)
            }
        }
    }
    output[h_idx*W + w_idx][c_out] = accum;
    // output * w_scale (后续 FP 乘法恢复量级)
}
```
关键优化：(1) 每 byte 操作同时处理 8 个 1-bit 值；(2) XNOR 和 popcount 均在寄存器完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPU（x86）：`_mm512_xor_si128` + `_mm_popcnt_u64` / `std::popcount`（C++20）。CUDA：`__vpopcnt4()`（SM 8.0+, 32-bit）、`__popcll()`。已知开源框架：daBNN（ARM NEON）、Larq Compute Engine（TF Lite）、BNN-PYNQ（FPGA）。PyTorch 无原生 1-bit kernel，BI-DiffSR 未实现定制 CUDA kernel（仅标注理论加速比），实际部署需自定义 CUDA kernel 或导出到 BNN 专用推理引擎。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---
