## Walsh-Hadamard Transform (WHT / 沃尔什-哈达玛变换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Walsh-Hadamard Transform (WHT) 是一种正交变换，其变换矩阵仅由 +1 和 -1 元素组成。对于维度 N=2^n，WHT 矩阵 H_N 通过 Kronecker 积递归构造：H_2 = 1/√2 [[1, 1], [1, -1]]，H_N = H_2 ⊗ H_{2^{n-1}}。行的集合构成正交的 Walsh-Hadamard 基（满足 H_N^T H_N = H_N H_N^T = I_N）。与 DFT/DCT/DHT 等正弦基不同，WHT 的基函数为方形波（square-wave patterns with sharp transitions），天然适合表示突变/尖峰信号（如模型权重的异常值）。WHT 满足快速计算特性：H_{2^n} 可在 O(n log n) 时间内计算，且由于元素仅为 ±1，计算仅涉及加法和减法，无需矩阵乘法。对于非 2 的幂维度，存在已知 Hadamard 矩阵的特定维度值（如 12、20、28 等），一般情况 N=2^n·m 下可用 H_N = H_{2^n} ⊗ H_m（H_m 为已知 Hadamard 矩阵）。在 QWHA 论文中，WHT 用作 PEFT 适配器的变换核，将稀疏系数矩阵 F 展开为全秩权重更新 ΔW = F H^{-1}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 QWHA 中 WHT 的核心用法——适配器权重更新计算流程：

```
# 给定: 稀疏系数矩阵 F ∈ R^{d_out × d_in}（仅 p 个非零元）
#       WHT 矩阵 H ∈ R^{d_in × d_in}（预计算并缓存，跨层共享）
#       H^{-1} = H^T（WHT 的正交性）

# 前向传播时权重更新 ΔW 的计算：
# 方法1（训练时）: 先展开 F 通过 H^{-1}
ΔW = F @ H^{-1}        # 矩阵乘法，O(d_out × d_in × d_in)
Y = X @ (W_Q + α·ΔW)^T

# 方法2（推理时，利用 WHT 的快速性）: 先对激活 X 做 WHT
X_transformed = X @ H^{-1}   # 通过 fast Hadamard kernel，O(d_in log d_in) 仅用加减法
ΔW_X = F_sparse_matmul(X_transformed)  # 稀疏矩阵乘法，O(p)
Y = W_Q @ X + α·ΔW_X

# WHT 矩阵预计算（一次性，跨层复用）：
H_2 = 1/√2 * [[1, 1], [1, -1]]
for n in 2..log2(N):
    H_{2^n} = H_2 ⊗ H_{2^{n-1}}  # Kronecker product
H_N = normalize(H_{2^n})          # 确保正交：H_N^T H_N = I
```

WHT 在 QA-PEFT 中相比 DCT/DHT 的优势体现在"能量集中"特性：WHT 系数的 Pareto hill index η 最小（分布最陡），意味着量化误差 ΔW_Q 的能量最大比例集中在最少 WHT 系数中。这使得稀疏适配器 F 能用少量参数高效重建量化误差，特别是大振幅异常值。对比：WHA (WHT) 捕获 18.12% 异常值系数，DHA (DHT) 17.06%，DCA (DCT) 仅 7.23%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WHT 在深度学习中的应用包括：(1) QuIP#/QuaRot 等量化方法中用作 incoherence processing（随机 Hadamard 旋转使权重矩阵更接近 i.i.d. 高斯分布，降低量化难度）；(2) QWHA 中作为 PEFT 适配器变换核；(3) 快速 Hadamard 变换可通过 fused CUDA kernel 实现（Dao-AILab 的 fast-hadamard-transform），利用 WHT 的 ±1 结构仅用加减法避免矩阵乘法，显著减少计算开销。H 矩阵可预计算一次并缓存在 GPU 内存中，跨所有同维度线性层复用。对于 WHT 系数和能量的理论等价性：||W||_F^2 = Σσ_i^2 = ||F||_F^2 = ||H'WH||_F^2（正交变换保持 Frobenius 范数）。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，Walsh-Hadamard 旋转矩阵 H 被用作 QA-SFT 中的可选旋转配置 R_i ∈ {H, I}。RoSTE 的关键创新在于自适应旋转策略：对每一层，通过比较 Walsh-Hadamard 旋转与无旋转的量化误差（公式 12）来逐层决策是否使用 H。随机旋转矩阵构造为 R(ζ) = H · Diag(r(ζ))，其中 H 为 Walsh-Hadamard 矩阵，r(ζ) ∈ {-1,1}^d 为随机符号向量。Proposition 4.4 证明使用 H 旋转后，权重量化误差从 O(d·max_i w_i²) 降至 O(log(4d/δ)/2 · ‖w‖²)，有效抑制了 outlier 值对量化精度的危害。旋转分为两类：(1) offline mergeable rotations (R1, R2) 可融进权重矩阵在训练前完成；(2) online rotations (R3, R4) 通过 fast Hadamard CUDA kernel 实现在线变换，用于消除 KV cache 的 activation outlier。
