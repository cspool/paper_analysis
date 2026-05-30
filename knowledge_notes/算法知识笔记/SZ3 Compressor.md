## SZ3 Compressor

术语解释
SZ3 是模块化的 CPU 端 error-bounded lossy compression 框架，由 Argonne National Laboratory 等机构开发，是 SZ 系列压缩器（SZ → SZ2 → SZ3）的最新主版本。发表于 IEEE Transactions on Big Data (2022)。

术语是什么？
SZ3 采用"预测-量化-编码"三阶段管线，核心模块化设计允许独立替换各阶段组件：
- **预测器（Predictor）**：Lorenzo predictor（利用多维数据空间梯度预测）、线性回归 predictor、样条 predictor 等可插拔
- **量化器（Quantizer）**：基于用户指定的 error bound ê 进行有界量化，支持 Absolute/Relative/PW_REL 等多种 error bound 模式
- **编码器（Encoder）**：Huffman 编码、变长编码等
- **关键优化**：自适应量化索引预测（IPDPS 2025 最新工作），可提升压缩比最高 95%

从算法pipeline角度拆解术语：
SZ3 在 MoE expert 压缩中的使用：
```
# 假设 expert 权重矩阵 W 形状: [d_model, d_ff] = [4096, 14336]
# 使用 SZ3 压缩（ABS mode, error bound ê = 0.01 * mean(|W|)）

# API 调用
sz3_config = SZ3_Config(
    error_bound_mode="ABS",     # 绝对误差模式
    error_bound=ê,              # 误差界
    predictor="Lorenzo",        # 预测器类型
    encoder="Huffman"           # 编码器类型
)
compressed_data = sz3_compress(W.flatten(), sz3_config)

# 压缩比计算
CR = W.size * 4 / len(compressed_data)  # 假设 FP32 (4 bytes)

# 解压
W_reconstructed = sz3_decompress(compressed_data, W.shape)
```

术语一般如何实现？如何使用？
- 语言：C++，提供 C API
- 平台：CPU (x86/ARM)，多线程并行
- 数据格式：支持 1D/2D/3D float/double 数组
- 典型使用场景：HPC checkpoint 压缩、科学数据传输、Federated Learning 模型压缩
- 开源链接：github.com/szcompressor/SZ3
- 2025 年仍作为 SOTA error-bounded compressor 被广泛使用和扩展

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

---
