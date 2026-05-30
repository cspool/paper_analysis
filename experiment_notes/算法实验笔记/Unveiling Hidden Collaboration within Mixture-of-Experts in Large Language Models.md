## Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：(1) **HSDL (Hierarchical Sparse Dictionary Learning)**——从 MoE LLM 的 expert activation matrix 出发，通过层级稀疏字典学习递归分解字典矩阵 $D_k \approx D_{k+1} \cdot R_{k+1}$，从粗到细地揭示专家之间跨层的协作模式（collaboration patterns）。引入三个约束：稀疏性约束 $L_{\text{sparse}} = ||R_{k,i,:}||_{\infty}$、层间一致性约束 $L_{\text{hier}}$、重构误差项 $L_{\text{rec}}$，总损失 $L_{\text{total}} = L_{\text{sparse}} + \lambda_1 L_{\text{hier}} + \lambda_2 L_{\text{rec}}$。(2) **CAEP (Contribution-Aware Expert Pruning)**——基于 HSDL 发现协作模式后，利用稀疏表示矩阵 R 和字典矩阵 D 计算每个专家的贡献分数 $\mathbf{e} = \sum_{i=1}^{N_p} \mathbf{D}_{\text{sum},i}$，通过初始阈值 mask + 迭代移除最少使用的 pattern 来逐步剪枝低贡献专家，直到达到目标剪枝比例。
  - 实验比较：(a) HSDL 发现的协作模式与穷举搜索（exhaustive search）的 pair/triplet 高频组合的覆盖度对比（Top-k% Coverage）；(b) 不同领域（数学、计算机科学、物理、法律、心理学）的专家激活频率分布和 cosine similarity 混淆矩阵；(c) CAEP vs Random/SEER-MoE/GEM 剪枝方法在 25% 专家删除后的 benchmark 性能（AVG/OBQA/ARC-C/HellaSwag/WinoGrande/RTE/PIQA）；(d) CAEP 在不同剪枝比例（25%/50%）下的性能退化曲线；(e) 按特定领域剪枝 50% 专家后在各领域的准确率退化热力图。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明硬件平台和 GPU 配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型：phi-moe（协作模式挖掘实验，Section 4.3.1）；DeepSeek-MoE-16B（剪枝实验，Section 5 及 Appendix C，仅剪枝 normal experts，保留 shared experts）。
  - 数据集：(a) 协作模式挖掘——MMLU-pro 数据集 2,812 个样本，覆盖数学、计算机科学、物理、法律、心理学 5 个领域；(b) 剪枝实验——MMLU 数据集 128 样本，输入序列长度 2,048 tokens（遵循 He et al., 2024 的设置）。
  - Benchmark：使用 EleutherAI LM Harness 框架评估，包含 ARC-C、BoolQ、HellaSwag、MMLU、OBQA、PIQA、RTE、WinoGrande，报告 normalized zero-shot accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明代码开源链接。
  - 算法 pipeline（从 expert activation 提取到剪枝的全流程）：
    ```
    # === 阶段一：Expert Activation Data Collection ===
    # 输入：MoE LLM（m 层，n 专家），数据集 S（N_s 样本）
    # 对每个样本 i 的第 t 个 token，记录 router 分配 α(i)_{t,j,k}
    # 句子级激活值聚合：
    v_{i,j,k} = Σ_{t=1}^{T} α(i)_{t,j,k}      # 式(1)
    # 构造 expert activation matrix：
    X ∈ R^{N_e × N_s}                           # 式(2) N_e = m × n

    # === 阶段二：HSDL 层级稀疏字典学习 ===
    # Layer 1: 对 X 做稀疏字典学习
    X ≈ D_1 · R_1                               # D_1 ∈ R^{N_e × N_p}, R_1 ∈ R^{N_p × N_s}
    # Layer k+1: 递归分解上一层字典
    D_k ≈ D_{k+1} · R_{k+1}                     # 式(3)
    # 损失函数（式(4)-(7)）：
    L_total = L_sparse + λ_1 * L_hier + λ_2 * L_rec
    # 输出：多层字典 {D_k} 和稀疏编码 {R_k}
    # D_k 的每个 atom 代表一组专家协作模式

    # === 阶段三：CAEP 剪枝（Algorithm 1）===
    # 输入：字典矩阵 D，稀疏表示矩阵 R，阈值比 k_1，目标剪枝比 k_2
    R_sum = Σ_{j=1}^{N_s} R_{:,j}               # 对样本维度求和
    D_sum = D · R_sum^T                          # 专家-模式贡献矩阵
    e = Σ_{i=1}^{N_p} D_sum[:,i]                 # 每个专家的总贡献分数
    e_sorted = sort_descending(e)
    threshold = e_sorted[ceil(k_1 * N_e)]         # k_1-分位数阈值
    m = 1_{e ≥ threshold}                        # 初始二值 mask
    while ||m||_0 > (1 - k_2) * N_e:             # 未达到目标剪枝比
        i* = argmin_i R_sum[i]                   # 找到最少使用的 pattern
        remove column i* from D, row i* from R
        recompute R_sum, D_sum, e
        m = 1_{e > threshold}                    # 更新 mask
    return m                                     # 保留=1，丢弃=0
    ```
  - 核心思想：字典每个 atom 编码一组跨层专家（如图 2 中 Layer 5 Expert 21 和 Layer 6 Expert 3 的共激活模式），稀疏编码 R 控制各 pattern 在不同样本上的参与度。剪枝时优先移除贡献分数低于阈值的专家，同时在迭代中移除最少被使用的协作模式，确保保留高贡献的专家组合。
