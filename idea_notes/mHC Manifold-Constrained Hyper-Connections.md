## mHC Manifold-Constrained Hyper-Connections

- baseline方法是什么？
  - Baseline 方法包括：
    - **标准残差连接（Residual Connection）**：$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$，identity mapping 保证 shallower layer 信号直接映射到 deeper layer，训练稳定。核心缺陷：残差流宽度固定为 C，信息容量受限于层输入维度，无法在不增加 FLOPs 的情况下扩展残差流的表达能力。
    - **Hyper-Connections (HC, Zhu et al. 2024)**：将残差流宽度扩展 n 倍（$\mathbf{x}_l \in \mathbb{R}^{n \times C}$），引入三个可学习映射——$\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$ 聚合 n-stream 为 C 维层输入，$\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$ 将层输出映射回 n-stream，$\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ 混合残差流内特征。不增加 FLOPs 前提下提升了模型性能，但 **核心缺陷**：(1) $\mathcal{H}_l^{\text{res}}$ 无约束导致复合映射 $\prod \mathcal{H}_l^{\text{res}}$ 偏离 identity mapping，信号可能爆炸或消失（Amax Gain Magnitude 可达 ~3000）；(2) 显存 I/O 开销约为标准残差连接的 $(5n+1)C$ 倍读和 $(3n+1)C$ 倍写；(3) pipeline parallelism 通信量增加 n 倍。
  - 全栈执行例子（以 HC 训练 27B MoE 模型为例）：
    - **算法层**：输入 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ → RMSNorm → 线性投影计算动态映射 $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}, \mathcal{H}^{\text{res}}$（含 tanh 激活）→ 加上学习 bias 静态映射 → $\mathcal{H}^{\text{pre}} \mathbf{x}_l$ 聚合为 C-dim → 标准 Attention/FFN 计算 → $\mathbf{x}_{l+1} = \mathcal{H}^{\text{res}} \mathbf{x}_l + \mathcal{H}^{\text{post}^\top} \mathcal{F}$ 更新 n-stream。因 $\mathcal{H}^{\text{res}}$ 无约束，跨 30 层（60 个 sublayer）后的复合映射行列和可达 3000×，梯度爆炸导致 12k step 处训练崩溃。
    - **系统框架层**：DeepSeek-V3 训练框架，使用 DualPipe pipeline parallelism。n-stream 残差在 stage 边界需传输 n×C 元素（vs 标准残差连接的 C），通信量增加 n 倍。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准矩阵乘法和逐元素操作，无融合。每个 $\mathcal{H}$ 的计算涉及独立 kernel launch（matmul + bias add + tanh + 应用映射），读/写量 = $(5n+1)C + n^2 + 2n$ / $(3n+1)C + n^2 + 2n$。中间激活需完整保存用于反向（nC + n×C 的 stream + 小系数）。
    - **硬件架构层**：在 NVIDIA GPU 集群上训练，具体型号论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **mHC 的核心设计**：将 HC 的 $\mathcal{H}_l^{\text{res}}$ 通过 Sinkhorn-Knopp 算法投影到双随机矩阵流形（Birkhoff polytope），同时约束 $\mathcal{H}_l^{\text{pre}}/\mathcal{H}_l^{\text{post}}$ 为非负（Sigmoid），恢复 identity mapping 的稳定性，并通过基础设施优化解决 I/O 瓶颈。
  - **解决 HC 的三个缺陷**：
    1. **训练不稳定性 → 双随机约束**：$\mathcal{H}_l^{\text{res}}$ 限制为双随机矩阵（行和=列和=1，元素≥0），(a) 谱范数 ≤ 1 防止梯度爆炸；(b) 双随机矩阵乘法封闭性保证跨任意深度的复合映射仍为双随机，Amax Gain Magnitude 从 ~3000 降至 ~1.6（降低 3 个数量级）；(c) Birkhoff polytope = 置换矩阵凸包，残差映射解释为"凸组合置换"，一致地单调增加特征混合而非发散。$\mathcal{H}_l^{\text{pre}}/\mathcal{H}_l^{\text{post}}$ 施加 Sigmoid 非负约束，防止正负系数抵消导致的信号衰减。
    2. **显存 I/O 开销 → Kernel Fusion + Recomputing**：5 个融合 kernel 消除冗余内存访问；RMSNorm 重排序优化；Sinkhorn-Knopp 单 kernel 实现（含定制反向）；Post+Res 映射应用与 residual merge 融合（读取从 $(3n+1)C$ 降至 $(n+1)C$）；选择性重计算策略（$L_r^*$ 最优块大小与 pipeline stage 对齐）使 n=4 时总开销仅 6.7%。
    3. **Pipeline 通信膨胀 → DualPipe 通信重叠**：扩展 DualPipe schedule，MLP 层 $\mathcal{F}_{post,res}$ kernel 在专用高优先级 compute stream 上运行以允许被通信抢占；attention 层避免 persistent kernel；重计算与通信解耦（首层激活已在本地缓存）。
  - 全栈执行例子（以 mHC 训练 27B MoE 模型为例）：
    - **算法层**：输入 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ → **Kernel 1**：flatten + RMSNorm + 线性投影获得 $\tilde{\mathcal{H}}^{\text{pre}}, \tilde{\mathcal{H}}^{\text{post}}, \tilde{\mathcal{H}}^{\text{res}}$ → **Kernel 2**：乘以 gating factor α、加 bias、RMSNorm 归一化 → **Kernel 3**：$\tilde{\mathcal{H}}^{\text{pre}}$ 经 $\sigma(\cdot)$、$\tilde{\mathcal{H}}^{\text{post}}$ 经 $2\sigma(\cdot)$（非负约束）、$\tilde{\mathcal{H}}^{\text{res}}$ 经 Sinkhorn-Knopp 20 次迭代（双随机约束）→ **Kernel 4**：$\mathcal{H}^{\text{pre}} \mathbf{x}_l$ 聚合为 C-dim → 标准 Attention/FFN → **Kernel 5**：$\mathcal{H}^{\text{res}} \mathbf{x}_l + \mathcal{H}^{\text{post}^\top} \mathcal{F}$ 更新 n-stream（融合 residual merge）。信号经 60 个 sublayer 后 Amax Gain Magnitude 仅 ~1.6，梯度稳定，无 loss spike。
    - **系统框架层**：DeepSeek-V3 + DualPipe，pipeline stage 边界通信与 mHC kernel 重计算被重叠。重计算块边界与 pipeline stage 对齐（$L_r^* \approx \sqrt{nL/(n+2)}$）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：5 个定制融合 kernel + 选择性重计算。前向仅持久化 $\mathbf{x}_{l_0}$ 和 $\mathcal{F}$ 的输出（每层），中间 stream 和映射系数均在反向重计算。TileLang 用于实现复杂 kernel，混合精度（bfloat16 输入 → tfloat32 权重 → float32 计算）。
    - **硬件架构层**：在 NVIDIA GPU 集群上训练，具体型号论文未明确说明。mHC 在 n=4 时额外时间开销仅 6.7%，意味着相对于 Baseline 的吞吐量损失非常小。
