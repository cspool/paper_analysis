## EGNN (E(n) Equivariant Graph Neural Network)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EGNN（E(n) Equivariant Graph Neural Network，Satorras et al. 2022）是一种对 E(n) 群（n 维欧几里得空间的旋转、平移、反射变换）保持等变性的图神经网络。与标准 GNN 仅学习 node representation 不同，EGNN 同时更新 node embedding hᵢ 和 3D 坐标 xᵢ，且坐标更新满足等变性：对输入坐标施加任何 E(n) 变换（旋转/平移/反射），输出坐标会自动施加相同的变换。核心机制是每层中：(1) 边消息 eᵢⱼ = ϕ(hⱼ, xⱼ, hᵢ, xᵢ) —— 输入包含坐标的 L2 距离差，ϕ 为 MLP；(2) node 表示更新 hᵢᐟ = COM^H(hᵢ, AGG({eᵢⱼ}))；(3) 坐标更新 xᵢᐟ = COM^X(xᵢ, AGG({eᵢⱼ})) —— 等变性的关键在于坐标更新仅使用边消息聚合（不直接操作坐标），因此自然保持 E(n) 等变性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: X⁽⁰⁾ ∈ R^{N×3} (初始坐标), H⁰ = MLP(X⁽⁰⁾) (node embedding), 边集 E
for l in 1..L:                            // L 层 EGNN
    for each edge (i,j) in E:
        // 计算相对距离平方（等变特征）
        d_ij² = ||x_i^{l-1} - x_j^{l-1}||²
        // 边消息网络 ϕ（MLP），输入: h_i, h_j, d_ij²
        e_ij^l = ϕ(h_j^{l-1}, h_i^{l-1}, d_ij²)
    
    for each node i:
        // 聚合邻居边消息
        m_i = Σ_{j∈N(i)} e_ij^l / |N(i)|    // mean aggregation
        // 更新 node embedding
        h_i^l = h_i^{l-1} + MLP_h(m_i)     // 残差更新
        // 更新坐标（等变）
        Δx_i = Σ_{j∈N(i)} (x_i^{l-1} - x_j^{l-1}) · MLP_x(e_ij^l)
        x_i^l = x_i^{l-1} + Δx_i           // 残差更新

输出: H^L, X^L → Decoder → X̂⁽ᵗ⁾
```
坐标更新公式 Δxᵢ = Σ(xᵢ−xⱼ)·MLP_x(eᵢⱼ) 保证等变性：坐标差 (xᵢ−xⱼ) 本身是等变的（平移不变、旋转变换一致），乘以标量 MLP 输出后仍保持等变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/vgsatorras/egnn（官方 PyTorch 实现）
- 实现要点：ϕ 网络输出可以是 scalar（用于坐标更新权重），node/coordinate 的 COM 操作支持 sum、mean、concat 等多种聚合方式
- 数据集：广泛用于 N-body 物理模拟（Spring, Charged）、分子动力学（MD17, QM9）、人体运动捕捉（Motion Capture）
- 在 LEGO 中的使用：EGNN 作为 Graph MoE 框架的基础 expert 模型。多个 EGNN 专家（同架构不同参数 θ¹...θᴷ）并行预测，由 LLM Judge 选最优。当基础 expert 改为 EGNO 或 Radial Field 时，LEGO 框架保持不变
- 局限：EGNN 仅保证 E(n) 等变性（非 SE(3)），对包含手性的分子场景可能不够严格；坐标更新仅依赖 L2 距离会丢失角度信息
- 相关模型：EGNO（Fourier 神经算子扩展）、Radial Field（仅操作坐标的 E(n) 模型）、SE(3)-Transformers（更严格的 3D 等变）、TFN（Tensor Field Networks）

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
