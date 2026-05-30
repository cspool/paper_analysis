## MoE-Specific CUDA Kernels (Quantized Expert Computation)

术语解释
针对MoE量化expert的专用CUDA kernel，处理低精度权重（INT4/INT2/INT1）与浮点激活值的矩阵乘法，在kernel内完成反量化+浮点计算，实现真正的推理加速。

术语是什么？
量化的MoE模型如果不使用专用kernel，只能在计算前统一反量化为FP16，然后再用标准GEMM——这种情况下只能节省内存，无法加速计算。专用kernel在kernel内部融合反量化和矩阵乘法：
- **MoE-CSP**：处理4-bit/8-bit量化权重的CUDA kernel，kernel内反量化 + FP32计算
- **QMoE**：1-bit压缩格式 + 专用GPU kernel，on-the-fly反量化
- 通用方案：W4A16 kernel（4-bit权重 + 16-bit激活）

从kernel调度角度拆解术语。
```
# MoE-CSP量化kernel伪代码
__global__ void moe_quantized_expert_kernel(
    int8_t* W_q,       // INT4量化权重 [d_ffn, d_model/2]
    half* x,           // FP16输入激活值 [d_model]
    half* y,           // FP16输出 [d_ffn]
    float* scales,     // 量化scale [d_ffn]
    int d_model, int d_ffn
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_ffn) return;
    
    float acc = 0.0f;
    for (int col = 0; col < d_model/2; col++) {
        // 读取打包的INT4值（2个INT4打包为1个INT8）
        int8_t packed = W_q[row * d_model/2 + col];
        // 解包为两个INT4
        int4_t w0 = (packed & 0x0F) - 8;  // 反量化的一部分
        int4_t w1 = (packed >> 4) - 8;
        // 反量化 + MAC（乘累加）
        acc += scales[row] * (float(w0) * __half2float(x[2*col]) + 
                              float(w1) * __half2float(x[2*col+1]));
    }
    y[row] = __float2half(acc);
}

# QMoE 1-bit kernel的关键差异
# 权重为1-bit，反量化仅涉及±1乘法
# acc += (bit == 1 ? scale : -scale) * x[col]
```

术语一般如何实现？如何使用？
- CUDA C++编写，使用__half（FP16）、int4_t等数据类型
- 需要处理内存对齐（INT4打包格式）和bank conflict
- 性能关键：shared memory使用优化、warp-level同步
- 开源实现：bitsandbytes、GPTQ、AutoGPTQ等库中的量化kernel
- 典型加速比：W4A16 kernel 1.5x-3x vs FP16 GEMM

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models

---
