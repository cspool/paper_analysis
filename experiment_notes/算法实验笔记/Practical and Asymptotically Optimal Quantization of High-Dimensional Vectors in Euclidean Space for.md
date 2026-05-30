## Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：扩展 RaBitQ 量化方法，使其支持中等压缩率（B bits/dim, e.g. 4-8 bits, 对应 4x-8x 压缩率）。核心设计：(1) **新码本构造**：从 B-bit 无符号整数网格 G = {-(2^B-1)/2 + u | u=0,...,2^B-1}^D 中取向量，归一化后用随机正交矩阵 P 旋转，得到码本 G_r = {P·y/||y|| | y∈G}。该码本由随机旋转的单位向量组成，继承了 RaBitQ 的无偏估计和渐近最优误差界；(2) **量化编码算法 (Algorithm 1)**：对每个数据向量 o，计算 o' = P^{-1}o，通过枚举至多 D·2^{B-1} 个 critical values（使用最小堆维护），每次 O(1) 更新时间，找到码本中最近向量 ȳ，编码为无符号整数向量 ȳ_u = ȳ + (2^B-1)/2 · 1_D，总复杂度 O(2^B·D log D)；(3) **距离估计**：使用无偏估计器 ⟨ō,q⟩/⟨ō,o⟩ 估计内积 ⟨o,q⟩，⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u, q'⟩ - (2^B-1)/2 · Σq'[i])，其中 q'=P^{-1}q；(4) **两阶段距离比较**：利用量化码的最高有效位（等价于原始 RaBitQ 的二进制码 ȳ₀）先通过 FastScan SIMD 批量估计粗略距离，若足以判定该候选非 NN 则剪枝，否则访问剩余位 ȳ_last 增量计算高精度距离 ⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩。
  - 实验比较（距离估计精度）：RaBitQ(ext) vs RaBitQ(pad)、SQ（均匀标量量化）、LVQ（逐向量标量量化）、PQ/OPQ（k=8），B=1~10 bits/dim，六个数据集上评估 average relative error 和 maximum relative error。
  - 实验比较（ANN 查询性能）：RaBitQ(ext) + IVF vs LVQ + IVF（最竞争 baseline），B=3,4,5,7,8,9，评估 Recall、Average Distance Ratio、QPS，K=100。还包含可扩展性验证（MSMARCO ~100M 向量）、无偏性验证（线性回归拟合 slope=1, intercept=0）、经验公式常数测量（c_ε=5.75）。

- 硬件平台是什么，配置是什么。
  - 两台 Intel Xeon Gold 6418H @4.0GHz（Sapphire Rapids 架构，48 cores/96 threads），1TB RAM。C++ 源码由 GCC 11.4.0 编译，-Ofast -march=native，Ubuntu 22.04 LTS。搜索性能单线程评估，索引时间多线程评估。所有方法均优化至 AVX512 SIMD 指令。

- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型——该方法用于压缩高维向量和加速 ANN 查询。数据集均为公开真实数据集：
    | 数据集 | 规模 | 维度 | 类型 |
    |---|---|---|---|
    | MSong | 992,272 | 420 | Audio |
    | Youtube | 999,000 | 1,024 | Video |
    | OpenAI-1536 | 999,000 | 1,536 | Text (text-embedding-3-large) |
    | OpenAI-3072 | 999,000 | 3,072 | Text (text-embedding-3-large) |
    | Word2Vec | 1,000,000 | 300 | Text |
    | GIST | 1,000,000 | 960 | Image |
    | MSMARCO | 113,520,750 | 1,024 | Text (Cohere embed-english-v3) |
  - 评价指标：Average Relative Error, Maximum Relative Error（距离估计）；Recall, Average Distance Ratio, QPS（ANN 查询）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/VectorDB-NTU/Extended-RaBitQ
  - Extended RaBitQ 量化 pipeline：
    ```
    # === Index Phase (压缩数据向量) ===
    # 输入：N 个 D 维数据向量 o_r，bit 数 B
    # 1. 计算全局质心 c = mean(o_r)
    # 2. 采样随机正交矩阵 P（用随机高斯矩阵 + QR 分解）
    # 3. 中心化并归一化: o = (o_r - c) / ||o_r - c||
    # 4. 变换: o' = P^{-1} @ o   (每个向量右乘 P^{-1})
    # 5. Algorithm 1 量化编码（对每个向量 o'）:
    #    t=0, v_max=0, t_max=0
    #    初始化 y_cur, <y_cur,o'>, ||y_cur|| with t=0
    #    while 存在未枚举的 critical value:
    #      t = 下一个最小 critical value (来自 minheap, O(log D))
    #      更新 y_cur（仅一个维度变化，O(1)）
    #      更新 <y_cur,o'> 和 ||y_cur||（O(1)）
    #      if <y_cur,o'>/||y_cur|| > v_max:
    #        v_max = <y_cur,o'>/||y_cur||, t_max = t
    #    ȳ = round(t_max · o')   (逐维度最近整数, 裁剪到 [-2^{B-1}+0.5, 2^{B-1}-0.5])
    #    ȳ_u = ȳ + (2^B-1)/2 · 1_D   (存储为 B-bit 无符号整数)
    #    存储: ||o_r-c||, 1/(||ȳ_u||·<ō,o>), ȳ_u

    # === Query Phase (估计距离) ===
    # 输入：查询向量 q_r
    # 1. 变换: q' = P^{-1} @ ((q_r-c)/||q_r-c||)
    # 2. 对每个候选数据向量:
    #    # 第一阶段：仅用最高有效位 ȳ₀（等价 RaBitQ 二进制码）
    #    # FastScan SIMD 批量计算 <ȳ₀, q'>
    #    dist_est_1 = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·(<ō,q>_est_1)
    #    # 其中 <ō,q>_est_1 = (2/√D)·<ȳ₀,q'> - (1/√D)·Σq'[i]，再除以 <ō,o>
    #    if dist_est_1 下界 > 当前最优距离:
    #      剪枝，跳过该候选
    #    else:
    #      # 第二阶段：访问剩余位 ȳ_last，增量计算
    #      <ȳ_u,q'> = 2^{B-1}·<ȳ₀,q'> + <ȳ_last,q'>
    #      <ō,q>_est_full = (1/||ȳ||)·(<ȳ_u,q'> - (2^B-1)/2 · Σq'[i])
    #      dist_est_full 用于最终 NN 选择
    ```
  - 关键张量计算（以 B=4, 8x 压缩为例）：
    - 原始 RaBitQ：ō₀ = P·(2/√D · x̄_b - 1/√D · 1_D)，其中 x̄_b ∈ {0,1}^D
    - Extended RaBitQ：ō = P·(ȳ/||ȳ||)，ȳ ∈ {-(2^B-1)/2 + u}^D，码本大小 2^{B·D}
    - B=1 时退化为原始 RaBitQ（码本 = 超立方体顶点 ±1/√D）
    - 误差经验公式：ε < 5.75 · 2^{-B} / √D（>99.9% 置信度）
  - IVF 集成：聚类数 = 4,096（百万级）/ 262,144（亿级 MSMARCO），每个聚类用本地质心中心化，扫描最近 nprobe 个聚类中的所有向量。
