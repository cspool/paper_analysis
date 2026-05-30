## RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：RaBitQ 是一种新的随机化量化方法，用于高维向量的 ANN 搜索距离估计。核心设计包括：(1) **码本构造**：先归一化数据向量到单位超球面，构造 2^D 个双值向量组成的确定码本 C = {±1/√D}^D（超立方体顶点），再采样随机正交矩阵 P 旋转码本得到 C_rand = {Px | x∈C}，消除对特定向量的偏好；(2) **量化编码**：对每个数据向量 o，计算 o'=P^{-1}o，取 o' 各维度的符号位构成 D-bit 字符串 x̄_b∈{0,1}^D 作为量化码，其中 1 对应 +1/√D、0 对应 -1/√D，O(D) 时间；(3) **无偏距离估计器**：⟨o,q⟩ ≈ ⟨ō,q⟩/⟨ō,o⟩，其中 ō=Px̄ 为量化向量，⟨ō,o⟩ 可预计算（期望值约 0.8），⟨ō,q⟩ = ⟨x̄,q'⟩，q'=P^{-1}q。证明该估计器无偏且具有严格的 O(1/√D) 概率误差界（渐近最优）；(4) **高效计算**：查询时对 q' 做随机化均匀标量量化（B_q=4 bit），⟨x̄_b,q̄_u⟩ 通过 bitwise-and + popcount 或 FastScan SIMD 批量实现。
  - 实验比较（距离估计精度与效率）：RaBitQ vs PQ、OPQ、LSQ，变长量化码（padding 0 或调整 M），六个数据集上评估 average relative error、maximum relative error 和 time per vector。
  - 实验比较（ANN 查询性能）：RaBitQ-batch + IVF vs OPQx4fs-batch + IVF vs HNSW，评估 Recall、Average Distance Ratio 和 QPS，K=100，对比 re-ranking 参数的影响（OPQ: rerank=500/1000/2500，RaBitQ: error-bound-based 无参数）。
  - 实验比较（参数验证）：ε₀（置信区间参数）从 0.0 到 4.0 的 recall 曲线；B_q（查询量化位数）从 1 到 8 的 average relative error 曲线；无偏性验证（10^7 样本对的线性回归）。
  - 实验比较（索引阶段时间）：RaBitQ (117s) vs PQ (105s) vs OPQ (291s) vs LSQ (>24h timeout)，GIST 数据集 32 线程。

- 硬件平台是什么，配置是什么。
  - AMD Threadripper PRO 3955WX @3.9GHz（Zen2 微架构，支持 AVX2 SIMD），64GB RAM。C++ 由 g++ 9.4.0 编译，-Ofast -march=core-avx2，Ubuntu 20.04 LTS。查询时间单线程评估，索引时间 32 线程评估。所有方法优化至 AVX2 SIMD 指令集。

- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型——用于压缩高维向量和加速 ANN 查询。六个公开真实数据集：
    | 数据集 | 规模 | D | Query Size | 类型 |
    |---|---|---|---|---|
    | MSong | 992,272 | 420 | 200 | Audio |
    | SIFT | 1,000,000 | 128 | 10,000 | Image |
    | DEEP | 1,000,000 | 256 | 1,000 | Image |
    | Word2Vec | 1,000,000 | 300 | 1,000 | Text |
    | GIST | 1,000,000 | 960 | 1,000 | Image |
    | Image | 2,340,373 | 150 | 200 | Image |
  - 评价指标：Average Relative Error、Maximum Relative Error、Time per Vector（距离估计）；Recall@100、Average Distance Ratio、QPS（ANN 查询）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/gaoj0017/RaBitQ
  - RaBitQ 量化 pipeline：
    ```
    # === Index Phase ===
    # 输入：N 个 D 维数据向量 o_r，聚类数 nlist=4096
    # 1. KMeans 聚类：{c_k}, 将向量分配到桶
    # 2. 采样随机正交矩阵 P（随机高斯矩阵 + QR 分解）
    # 3. 对每个聚类 k:
    #    o = (o_r - c_k) / ||o_r - c_k||          # 单位化
    #    o' = P^{-1} @ o                           # 逆变换
    #    x̄_b[i] = 1 if o'[i] > 0 else 0           # 逐维度符号 → D-bit 字符串
    #    precompute: ||o_r - c_k||, <ō,o> = (1/√D)·Σ|o'[i]|
    # 存储：量化码 x̄_b (D bits) + ||o_r-c_k|| + <ō,o>

    # === Query Phase (单向量) ===
    # 输入：查询向量 q_r
    # 1. 选最近 nprobe 个聚类质心
    # 2. 对每个选中聚类 k:
    #    q = (q_r - c_k) / ||q_r - c_k||
    #    q' = P^{-1} @ q
    #    q' 随机化均匀标量量化 → q̄_u (B_q=4-bit unsigned integers)
    #    对每个候选数据向量：
    #      <x̄_b, q̄_u> = Σ_j 2^j · popcount(x̄_b & q̄_u^{(j)})  # bitwise 分解
    #      <ō,q> = (2Δ/√D)·<x̄_b,q̄_u> + (2v_l/√D)·popcount(x̄_b) - (Δ/√D)·Σq̄_u[i] - √D·v_l
    #      estimator = <ō,q> / <ō,o>                         # 无偏估计 ⟨o,q⟩
    #      dist_est² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·estimator
    #      error_bound = √((1-<ō,o>²)/<ō,o>²) · (ε₀/√(D-1))
    #      if dist_est² - error_bound > best_exact_dist²: 距离下界 > 当前最优 → 剪枝
    ```

    # === Query Phase (batch, FastScan SIMD) ===
    # 对批处理 32 个量化码：
    #   D 位字符串拆成 D/4 个 4-bit 子段
    #   预计算 D/4 个 LUT，每个 LUT 含 2^4=16 个值
    #   LUT 加载到 AVX2 256-bit 寄存器（每寄存器 2 个 LUT）
    #   SIMD shuffle → 查表 + 累加，一周期处理 32 个向量
    ```
  - 关键张量计算：
    - 码本向量：x ∈ C = {±1/√D}^D，最终码本 C_rand = P·C（不显式物化，维护矩阵 P 即可）
    - 量化码：x̄_b ∈ {0,1}^D，关系 x̄[i] = (2·x̄_b[i] - 1)/√D
    - 查询量化：q̄ = Δ·q̄_u + v_l·1_D，q̄_u[i] = ⌊(q'[i]-v_l)/Δ + u_i⌋, u_i~Uniform(0,1)
    - 内积分解：⟨x̄,q̄⟩ = (2Δ/√D)·⟨x̄_b,q̄_u⟩ + (2v_l/√D)·Σx̄_b[i] - (Δ/√D)·Σq̄_u[i] - √D·v_l
    - 距离估计器：⟨o,q⟩ ≈ ⟨ō,q⟩/⟨ō,o⟩ (无偏), 误差界 O(1/√D) w.h.p.
    - bitwise 实现：⟨x̄_b,q̄_u⟩ = Σ_{j=0}^{B_q-1} 2^j · popcount(x̄_b AND q̄_u^{(j)})
  - IVF 集成：聚类数 4,096，每个聚类独立归一化和量化。查询时选最近 nprobe 个聚类，对聚类内所有向量基于 error bound 判定是否需要 re-rank（精确距离计算）。无 re-ranking 参数需手工调参。
  - 关键参数：ε₀=1.9（控制置信区间，固定无需调参），B_q=4（查询量化位宽，固定无需调参），默认量化码长度 = ceil(D/64)×64 bits。
