## Optimal and Approximate Adaptive Stochastic Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出三个 ASQ (Adaptive Stochastic Quantization) 算法：(1) **QUIVER**——利用预处理数组（β/γ 累积和）使 C[k,j] 可 O(1) 计算，证明 C 满足 quadrangle inequality 从而使 DP 矩阵成为 totally monotone matrix，应用 SMAWK 算法以 O(d) 时间找到行最小值，将 ASQ 动态规划从 O(s·d²) 时间/O(d²) 空间优化到 O(s·d) 时间/O(s·d) 空间；(2) **Accelerated QUIVER**——推导 s=3 时中间量化值的闭式解（C²[k,j] 可通过 b*_{k,j} 公式 O(1) 计算），每次 SMAWK 调用跳过两个量化值而非一个，将 SMAWK 调用次数从 s-2 减半至 ⌊s/2⌋-1，速度提升最高 5.4×；(3) **Apx. QUIVER**——将量化值候选集离散化为 m+1 个均匀网格点，用 histogram-style 预处理计算 C_m[k,j]，运行复杂度 O(d + m·s)，提供保证 AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)。
  - 实验比较：(a) Exact 对比：Accelerated QUIVER vs ZipML 精确解，不同 d (2¹⁰ 到 2²³) 和 s (2 到 16) 下的 runtime 和 vNMSE；(b) Approximate 对比：Apx. QUIVER vs ZipML-CP Uniform、ZipML-CP Quantiles、ZipML 2-Apx、ALQ；(c) 分布泛化：LogNormal、Normal、Exponential、TruncNorm、Weibull 五种分布；(d) 加速比消融：Accelerated QUIVER vs QUIVER 在不同 s 和 d 下的加速比。

- 硬件平台是什么，配置是什么。
  - AWS g4dn.4xlarge EC2 实例，custom Intel Cascade Lake CPU，64 GB RAM，Ubuntu 22.04 OS。GPU 排序/量化的额外开销测量使用同实例上配备的 NVIDIA T4 GPU，PyTorch v2.1.2，CUDA toolkit v12.3。

- 模型是什么。数据集和bench分别是什么。
  - 本研究不涉及具体 ML 模型，而是以合成向量作为输入。向量条目为 i.i.d. 采样自五种分布：**LogNormal(0,σ²)**、**Normal(0,1)**、**Exponential(1)**、**Truncated Normal(μ=0,σ²=1,a=-1,b=1)**、**Weibull(1,1)**。这些分布被已有工作报道为能刻画 DNN gradients、模型权重和激活值的分布特征。向量维度 d 范围 2¹⁰ 到 2²⁴（约 16M）。评估指标为 **vNMSE**（vector normalized MSE = E[‖X-X̂‖²]/‖X‖²）和 **runtime**。每个实验 5 次随机种子取平均。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：C++ 实现 https://github.com/ranbenbasat/QUIVER。
  - QUIVER 算法核心流程（Algorithm 1）：
    ```
    输入: 已排序向量 X ∈ R^d, 量化值个数 s
    输出: 最优量化值集合 Q ⊆ X, |Q|=s
    
    1. Preprocess(X): 计算 β_j=Σ_{i=1}^j x_i, γ_j=Σ_{i=1}^j x_i², ∀j∈[1,d]
       // 使 C[k,j] = -x_j·x_k·(j-k) + (x_j+x_k)·(β_j-β_k) - (γ_j-γ_k) 可O(1)计算
    2. for j=2 to d: MSE[2,j] = C[1,j]  // 初始化 (i=2 行)
    3. for i=3 to s:
         // 隐式定义矩阵 A[k,j] = MSE[i-1,k] + C[k,j]
         // C 满足 quadrangle inequality → A 是 totally monotone
         K[i,·] = SMAWK(A)  // O(d) 时间找到每列的行最小值索引
         MSE[i,j] = MSE[i-1, K[i,j]] + C[K[i,j], j], ∀j∈[i..d]
    4. Q = {x₁, x_d}, j = d
    5. for i=s down to 3:
         j = K[i, j]; Q = Q ∪ {x_j}
    6. return Q
    ```
  - Accelerated QUIVER 关键加速（s=3 闭式解）：
    ```
    // C²[k,j] = C[k, b*_{k,j}] + C[b*_{k,j}, j]  在 O(1) 计算
    b*_{k,j} = ⌈(j·x_j - k·x_k - (β_j - β_k)) / (x_j - x_k)⌉
    
    // 推导：对区间 [x_k,x_j] 中间插入 q，Q(q) 为两段方差和
    // dQ/dq = Σ_{x∈[x_k,q]}(x-x_k) - Σ_{x∈(q,x_j]}(x_j-x)
    // 极小值在导数从负变正处 → b*_{k,j} 闭式解
    // 每次 SMAWK 调用跳过两个量化值，调用次数减半
    ```
  - Apx. QUIVER 关键流程：
    ```
    输入: X, s, m       // m 为离散网格划分数
    1. δ = (x_d - x_1)/m
    2. 收集直方图: A_ℓ = count(x in [s_ℓ, s_{ℓ+1}]), ∀ℓ∈[0,m-1]
    3. 计算累积量: α_ℓ, β_ℓ, γ_ℓ (O(m) 扫描)
    4. 用 C_m[k,j] = -s_j·s_k·(α_j-α_k) + (s_j+s_k)·(β_j-β_k) - (γ_j-γ_k)
       替代 C[k,j]，其余与 QUIVER 相同
    5. 复杂度: O(d + m·s)
    6. 近似保证: AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)
    ```
  - 使用示例（C++ 命令行）：
    ```bash
    # 编译
    g++ -O3 -std=c++17 quiver.cpp -o quiver
    # 运行：对 1M 维 LogNormal 向量计算最优 4-bit 量化值
    ./quiver --input vector_1M.txt --s 16 --algorithm accelerated
    # 近似量化：m=400 离散网格，6ms 完成
    ./quiver --input vector_1M.txt --s 16 --algorithm approximate --m 400
    ```
