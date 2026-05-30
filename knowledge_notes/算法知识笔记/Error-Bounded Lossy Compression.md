## Error-Bounded Lossy Compression

术语解释
Error-Bounded Lossy Compression（有界误差有损压缩）是一类保证重建数据与原始数据之间绝对误差不超过预设阈值（error bound ê）的有损压缩算法。区别于传统量化方法产生不可控、不可预测的误差，error-bounded 压缩在压缩比和重建精度之间提供可配置、可预测的权衡。

术语是什么？
Error-bounded lossy compression 的核心 pipeline：原始数据 → 预测（利用空间/时间相关性预测每个值）→ 量化（基于 error bound ê 控制量化步长，保证 |重建值 - 原始值| ≤ ê）→ 编码（Huffman/变长编码压缩量化残差）。代表性实现包括 SZ3（CPU）、CuSZp（GPU）、ZFP 等。关键特性：
- **有界误差保证**：对任意参数 θ_i 和其重建值 θ'_i，有 |θ_i - θ'_i| ≤ ê（绝对误差界）或 |θ_i - θ'_i|/|θ_i| ≤ ê_rel（相对误差界）
- **高压缩比**：利用数据空间相关性（如相邻参数值相近），通过预测器去除冗余，可实现远超简单量化的压缩比。例如 CuSZp 在 A100 上可实现 ~300 GB/s 端到端吞吐
- **可控制的精度-压缩权衡**：增大 ê 获得更高压缩比但更大误差；减小 ê 获得更高精度但更低压缩比

从算法pipeline角度拆解术语：
```
# Error-Bounded Lossy Compression Pipeline (SZ3 风格)
输入: 原始数据张量 X ∈ R^n, error bound ê (绝对误差界)

# Step 1: 预测 (Prediction)
X_pred = Predictor(X)  # Lorenzo predictor / linear regression / spline
                        # 利用相邻数据点的值预测当前点

# Step 2: 计算残差 (Residual)
R = X - X_pred  # 预测误差

# Step 3: 有界量化 (Bounded Quantization)
# 量化步长 q 的选择保证解量化后误差 ≤ ê
q = 2 * ê  # 线性量化器步长与 error bound 的关系
Q = round(R / q) * q  # 量化残差，每个值误差 ≤ ê

# Step 4: 编码 (Encoding)
compressed_data = Encode(Q)  # Huffman / 变长编码

# 解压过程 (逆过程)
Q' = Decode(compressed_data)
X_reconstructed = X_pred + Q'
# 保证: |X_i - X_reconstructed_i| ≤ ê for all i
```

在 MoE 推理 offloading 场景中的应用：
```
# Expert 压缩 offloading 流程
原始 expert 权重 W ∈ R^{d_model × d_ff}

# CPU 端压缩
compressed_W = SZ3_compress(W, error_bound=ê)

# PCIe 传输（数据量 = 原始 size / 压缩比 CR）
transfer_size = |W| * sizeof(float) / CR

# GPU 端解压
W_reconstructed = CuSZp_decompress(compressed_W)
# 保证: |W_ij - W_reconstructed_ij| ≤ ê

# GPU 上 FFN 推理（使用含 bounded error 的权重）
output = W_reconstructed @ input
```

术语一般如何实现？如何使用？
- **科学数据压缩场景**：SZ3/CuSZp 最初设计用于 HPC 科学模拟数据压缩，数据量巨大且有损压缩可接受
- **ML 模型压缩场景**：MoE expert 参数压缩是新应用方向，利用 error bound 保证推理精度可控
- **常见实现**：SZ3（C++，CPU 多线程）、CuSZp（CUDA，GPU）、ZFP（C，CPU/GPU）
- **配置参数**：error bound mode（ABS/REL/VR_REL/PW_REL）、error bound value ê、预测器类型（Lorenzo/Linear/Polynomial）
- **开源链接**：SZ3 (github.com/szcompressor/SZ3), CuSZp (github.com/szcompressor/cuSZp)
- **与量化的对比**：量化固定位宽（如 4-bit）→ 误差由位宽和数据分布决定，不可控；Error-bounded 压缩固定误差界 → 位宽和压缩比自适应，误差可保证

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

---
