## Binary-Coded Quantization (BCQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Binary-Coded Quantization (BCQ) 是一种非均匀量化方案，将权重矩阵 W ∈ R^{m×n} 表示为 q 个二值基矩阵 B_i ∈ {-1,+1}^{m×n} 与实值缩放因子 α_i ∈ R 的线性组合：Ŵ = Σ_{i=1}^q α_i B_i。其中 q 为量化比特数。参数通过最小化 Frobenius 重建误差 e = ‖W - Ŵ‖_F² 获得。当 q=1 时退化为标准二值量化：B₁* = sign(W), α₁* = ⟨W, B*⟩ / ‖B*‖_F²。当 q>1 时采用贪心初始化 + 交替优化：(1) 贪心：逐比特计算残差 R_i = W - Σ_{j=1}^i α_j B_j，B_{i+1} = sign(R_i)；(2) 交替优化：最小二乘更新 α = (B^T B)^{-1} B^T W，二分搜索重分配 B_i。BCQ 的核心优势是二值结构天然适合硬件加速——每个比特平面的运算化为对激活值的加减操作（因 B_i ∈ {-1,+1}），无需 centroid lookup 等非均匀量化的额外开销。相关工作包括 LUT-GEMM（利用二值模式查表加速 GPU GEMM）、iFPU（利用指数预对齐降低浮点运算复杂度）、FIGLUT（硬件加速器 LUT-based GEMM）。BCQ 也是 ShiftAddLLM 和 AnyBCQ 的底层量化格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BCQ 在 AnyBCQ 中 q=2 至 4 的逐层量化流程（group-wise asymmetric BCQ, g=128）：

```
# q-bit BCQ 量化伪代码
def BCQ_quantize(W, q, g=128, T=20):
    # W ∈ R^{m×n}, 按 g=128 列分组
    for group in range(0, n, g):
        W_g = W[:, group:group+g]
        B = []        # 比特平面列表
        alpha = []    # 缩放因子列表
        residual = W_g.clone()
        
        # 贪心初始化
        for i in range(q):
            B_i = sign(residual)  # B_i ∈ {-1,+1}^{m×g}
            alpha_i = dot(residual, B_i) / norm(B_i)^2
            residual = residual - alpha_i * B_i
            B.append(B_i); alpha.append(alpha_i)
        
        # 交替优化 (T 轮)
        for t in range(T):
            B_mat = concat(B)           # [m×g, q]
            alpha = least_squares(B_mat, W_g)  # α = (B^T B)^{-1} B^T W
            for i in range(q):
                B[i] = binary_search(alpha, W_g)  # 重分配 B_i ∈ {-1,+1}
        
        store(B_1...B_q, alpha_1...alpha_q)
```

关键参数：AnyBCQ 使用 asymmetric BCQ（允许偏差项），g=128，T=20，lr=1e-4，10 MRE epochs。q-bit BCQ 可表达 2^q 个不同值（对应 B 的 2^q 种符号组合），在 α-space 的线性子空间内变化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BCQ 的开源实现：(1) LUT-GEMM (github.com/naver-aics/lut-gemm)：BCQ 格式下的 GPU 推理 kernel；(2) transformer_bcq (github.com/insoochung/transformer_bcq)：TensorFlow BCQ 教程；(3) AnyBCQ (github.com/naver-aics/anybcq)：多精度 BCQ。硬件加速器包括 iFPU（bit-plane 浮点-整数混合运算单元）和 FIGLUT（HPCA 2025，LUT-based FP-INT GEMM 加速器）。权衡：低比特下 BCQ 优于均匀量化但逊于 K-means 聚类；4-bit 时各方法差距小；BCQ 的硬件效率优势（直接比特平面操作，消除 centroid lookup 和 bit-transpose）使其更具实用性。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---
