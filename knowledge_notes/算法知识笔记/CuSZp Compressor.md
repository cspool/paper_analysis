## CuSZp Compressor

术语解释
CuSZp 是超快速的 GPU 端 error-bounded lossy compression 框架，发表于 SC'23。核心创新是将完整的压缩/解压阶段融合到单个 CUDA kernel 中，消除 kernel launch 和数据移动开销，在 NVIDIA A100 上实现约 300 GB/s 的端到端吞吐。

术语是什么？
CuSZp 的核心设计特点：
- **单 kernel 设计**：压缩和解压各自融合为一个 CUDA kernel 函数，避免多次 kernel launch 和中间数据传输
- **多种编码模式**：Fixed（固定长度编码）、Plain（维度感知 delta 编码 + 固定长度编码）、Outlier（delta 编码 + 异常值保留）——适用于不同数据特征
- **支持 1D/2D/3D 数据**，FP32 和 FP64
- **极高性能**：SC'23 论文报告在 A100 上平均压缩吞吐 93.63 GB/s，解压吞吐 120.04 GB/s；比 cuSZ 快 95.53×，比 cuSZx 快 55.18×。
- **后续版本**：cuSZp2 (SC'24)、cuSZp3 (SC'25)，持续优化编码模式和压缩比

从算法pipeline角度拆解术语：
CuSZp 在 MoE expert GPU 端解压中的使用：
```
# GPU 端使用 CuSZp 解压 expert 权重
# 输入: compressed_expert_data (从 CPU 通过 PCIe 传输)
# 输出: expert_weights (用于 GPU FFN 计算)

# CuSZp 单 kernel 解压伪代码
__global__ void cuszp_decompress_kernel(
    uint8_t* compressed_data,
    float* output_weights,
    int n_elements,
    float error_bound
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 单 kernel 完成: 解码 → 反量化 → 反预测
    // 输出含 bounded error 的权重
}

# 调用示例
cuszp_decompress(compressed_data, d_expert_weights, n_params, ê);
# 之后直接使用 d_expert_weights 进行 GEMM 计算
output = expert_ffn(d_expert_weights, input_activation);
```

术语一般如何实现？如何使用？
- 语言：CUDA C/C++，提供 C/C++ API 和 Python API
- 平台：NVIDIA GPU (A100/H100 tested)
- 开源：github.com/szcompressor/cuSZp
- 使用模式：CPU 端压缩（SZ3）→ PCIe 传输 → GPU 端解压（CuSZp）→ GPU 计算
- 在 MoE offloading 场景中：压缩由 CPU 端 SZ3 完成（因非激活 expert 存储在主存），GPU 端 CuSZp 负责快速解压以最小化推理延迟

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference
