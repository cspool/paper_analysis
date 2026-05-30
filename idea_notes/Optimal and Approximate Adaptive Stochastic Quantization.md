## Optimal and Approximate Adaptive Stochastic Quantization

- baseline方法是什么？
  - **ZipML**：通过动态规划（DP）求解 ASQ 问题的最优量化值集合 Q ⊆ X。DP 状态 MSE[i,j] 表示用 i 个量化值量化前缀向量 X_j 的最优 MSE。转移：MSE[i,j] = min_{k} MSE[i-1,k] + C[k,j]，其中 C[k,j] 为区间内所有条目在连续量化值 {x_k, x_j} 下的方差和。时间复杂度 O(s·d²)，空间复杂度 O(d²)。当 d > 10⁵ 时内存和时间均不可行，无法用于"on the fly"量化场景。
  - **ZipML-CP (Candidate Points)**：从 X 中选取 m 个候选点（均匀网格或分位数），在候选点子集上运行 ZipML 精确解。时间复杂度 O(d + m²·s)，但未提供最优候选点选择策略，近似质量不可控。
  - **ZipML 2-Apx**：保证 MSE ≤ 2·opt_{X,⌊s/2⌋}（使用两倍量化值保证误差不超过最优解的两倍），时间复杂度 O(d log d + s³)。
  - **ALQ**：假设输入服从截断正态分布，拟合分布参数后通过迭代积分求解近似最优量化值。单次量化需 ≈10s 次积分计算，实际速度慢，且假设分布不总是成立。
  - 全栈执行例子（以 ZipML baseline 量化一个 LogNormal 梯度向量为例）：
    - **算法层**：加载 d=10⁵ 维 FP32 向量 → 排序 O(d log d) → 计算 C[k,j] 矩阵（对所有 k≤j 逐一求和，O(d³) 或优化到 O(d²)）→ 逐行填充 MSE[i,j] 表（i=2..s, j=i..d，每步 O(d) 枚举 k）→ 回溯构建 Q。对 d=10⁵, s=16 无法在 commodity PC 上完成（内存 > 80 GB，时间 > 10³ 秒）。
    - **系统框架层**：无特定框架依赖，纯算法计算。对于分布式/联邦学习场景，量化器运行在 CPU 上（gradient compression sender/receiver 端）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（baseline 为纯 CPU 算法）。
  - Baseline 核心缺陷：**时间复杂度 O(s·d²) 和空间复杂度 O(d²) 使得 ASQ 在大向量上不可行**。即使 d=10⁵ 级别也因内存溢出无法运行，而实际 ML 场景中梯度向量维度常达 10⁶-10⁷。已有近似方法要么假设特定分布（ALQ）、要么近似保证弱（ZipML 2-Apx 使用 2× 量化值只保证 2× 误差）、要么无理论保证（ZipML-CP）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **QUIVER** 通过三个创新将 ASQ 从不可行变为实用：
    1. **O(1) C[k,j] 计算**：预处理累积和数组 β_j=Σx_i, γ_j=Σx_i²，使方差和 C[k,j] = -x_j·x_k·(j-k) + (x_j+x_k)·(β_j-β_k) - (γ_j-γ_k) 可常数时间求值，无需预计算/存储 O(d²) 的 C 矩阵。
    2. **Quadrangle Inequality → SMAWK**：证明 C 满足 quadrangle inequality（∀a≤b≤c≤d: C[a,c]+C[b,d] ≤ C[a,d]+C[b,c]），从而 DP 矩阵 A[k,j]=MSE[i-1,k]+C[k,j] 是 totally monotone。对 totally monotone matrix，SMAWK 算法可在 O(d) 时间内找到每列的行最小值索引，替代原 DP 中每步 O(d²) 的枚举。
    3. **s=3 闭式解 → Accelerated QUIVER**：推导三个量化值时的中间值闭式解 b*_{k,j} = ⌈(j·x_j - k·x_k - (β_j-β_k))/(x_j-x_k)⌉，使得 C²[k,j] 亦可 O(1) 计算。每次 SMAWK 调用跳过两个量化值，调用次数减半，速度提升最高 5.4×。
    4. **离散化 → Apx. QUIVER**：将候选量化值限制在 m 个均匀网格点上，使用直方图预处理实现 O(d+m·s) 时间，并给出严格近似保证 AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)。
  - **如何解决 Baseline 缺陷**：
    - **时间从 O(s·d²) 到 O(s·d)**：通过预处理+隐式矩阵+SMAWK 的组合，将每步 DP 转移的枚举优化从 O(d) 降为 amortized O(1)，最终使 d=10⁶, s=16 的精确解在 1 秒内完成（vs ZipML 无法运行）。
    - **空间从 O(d²) 到 O(s·d)**：不再需要存储 C 矩阵（仅存 β/γ 两数组和当前 MSE 行），使 d=2²⁴（16M）可在一台 commodity PC 上运行。
    - **近似方案的严格保证**：Apx. QUIVER 提供 additive error bound（而非 ZipML 2-Apx 的 multiplicative 2× bound），且通过参数 m 提供可控的 accuracy-speed tradeoff（m=400 时 vNMSE 接近最优，6ms 完成 1M 向量）。
    - **无需分布假设**：与 ALQ（需截断正态假设）不同，QUIVER 是精确算法（Q ⊆ X），不受 input distribution 限制。
  - 论文方法全栈执行例子（以 Accelerated QUIVER 量化 d=10⁶ LogNormal 向量为例）：
    - **算法层**：
      1. Preprocess: 排序（若非已排序则 O(d log d)），一趟扫描计算 β_j=Σx_i, γ_j=Σx_i²（O(d) 空间）。
      2. 初始化 MSE 表：若 s 为偶数则 MSE[2,j]=C[1,j]；若 s 为奇数则 MSE[3,j]=C²[1,j]（利用 b* 闭式解 O(1)）。
      3. SMAWK 迭代（⌊s/2⌋-1 轮）：每轮在隐式 totally monotone matrix B[k,j]=MSE[prev,k]+C²[k,j] 上运行 SMAWK，O(d) 时间求得当前行 MSE 和 argmin K。
      4. 回溯重建：从 x_d 开始，沿 K 逆向跳转，并在每个区间中用 b* 恢复跳过的中间量化值。
      5. 结果：Q={q₁,...,q₁₆} ⊆ X，信源量化：对每个 x∈X，找到包围它的两连续量化值 q_l≤x≤q_r，以概率 p_l=(q_r-x)/(q_r-q_l) 输出 q_l，否则输出 q_r（保证 E[x̂]=x）。
    - **系统框架层**：C++17 单线程 CPU 实现，compiled with -O3。用于分布式学习的 gradient compression 时，sender 端先排序→QUIVER→量化→发送 Q+量化比特；receiver 端根据 Q 和比特解码。实际场景中排序可由 GPU 完成（T4 GPU 上 1M 向量排序仅 4ms）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（纯 CPU 算法，不涉及 GPU kernel 或专用硬件）。论文明确指出现有 QUIVER "not GPU friendly"，将 GPU-friendly ASQ 设计列为 future work。
  - 关键数学洞察：**C 的 quadrangle inequality 证明**是算法的理论基石——它将看似 ad-hoc 的方差和函数与 totally monotone matrix 的经典理论（SMAWK, 1986）联系起来，实现了从 O(d²) 到 O(d) 的渐进改进。这一观察不仅适用于 ASQ，也可能适用于其他满足该性质的序列分割 DP 问题。
