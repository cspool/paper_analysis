## Uniform Quantization（均匀量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Uniform Quantization（均匀量化）是将连续的浮点数值映射到等间距离散值的量化方式。其核心公式为：W_int = clamp(round(W/s) + z, 0, 2^N - 1)，反量化 Ŵ = (W_int - z)·s。其中 s 为步长（step size/scaling factor），z 为零点（zero point），N 为目标位宽。均匀量化的量化级别呈等差数列分布，与vector quantization（使用codebook中非均匀离散值）相对。均匀量化的优势在于硬件友好性——标准的INT MAC单元可直接处理均匀量化的整数值，无需额外的codebook查表或比特转置，因此被GPTQ、AWQ、OmniQuant、EfficientQAT等主流方法广泛采用。但在极低位宽（2-bit）下，均匀量化因表达能力有限（仅4个离散值），精度低于QuIP#等vector quantization，后者的复杂codebook设计能更精确逼近原始分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在EfficientQAT的Block-AP中，均匀量化以per-group方式应用于每个Linear层权重：
```
# 每组g个权重共享s(FP16)和z(N-bit)
for group in split(weight, group_size=g):
    s = group.abs().max() / (2^N - 1)          # 初始化步长
    z = -group.min() / s                        # 初始化零点
    # 前向量化（包含在计算图中，可反向传播）
    W_int = clamp(round(W/s) + z, 0, 2^N - 1)   # N-bit 整数
    W_hat = (W_int - z) * s                      # 反量化为FP16
    output = matmul(x, W_hat)                    # FP16矩阵乘法
```
平均位宽公式：bits/param = N + (N+16)/g。例如g=64时：2-bit → 2.28 bits/param，3-bit → 3.30 bits/param，4-bit → 4.31 bits/param。高效硬件兼容性是均匀量化的核心优势：与vector quantization（AQLM、QuIP#）不同——vector量化仅限于weight-only，且需专用kernel处理codebook查表——均匀量化可同时压缩权重和激活，兼容MLC-LLM、AWQ、BitBLAS、Marlin、T-MAC等标准推理框架。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
均匀量化的实现层次：(1) 对称均匀量化（z=0，仅需s）：最简单，但假设权重零均值对称分布；(2) 非对称均匀量化（含s和z）：可处理偏态分布，额外存储一个N-bit零点每group；(3) per-tensor/per-channel/per-group：共享s/z的粒度越细精度越高但存储开销越大。在EfficientQAT中，per-group g=64在精度和存储间取得平衡（2-bit w2g64压缩比≈82% vs FP16）。实现上，PyTorch中通过自定义autograd Function打包量化/反量化为单一操作符，推理时使用packing格式存储（每8个4-bit权重打包为1字节，或每4个2-bit权重打包为1字节）。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

SLiM (ICML 2025) 提出了均匀量化的概率化最优 scaling 方法 SLiM-Quant，解决了传统 symmetric uniform quantization 中 AbsMax 对 outlier 敏感、Grid Search 次优且昂贵的痛点。SLiM-Quant 将 non-convex 的 MSE 最小化目标通过概率化重表述转化为在权重直方图上的数值积分，利用多网格策略高效找到全局最优 α*，使 uniform quantization 精度达到 group quantization 水平，同时保留 uniform 的硬件友好性（单 scale per tensor, 6% 实测加速 vs group）。

---
