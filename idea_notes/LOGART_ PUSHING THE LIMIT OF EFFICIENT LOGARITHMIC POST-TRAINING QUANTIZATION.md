## LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

- baseline方法是什么？
  传统对数 PTQ（Log2/Log√2/DLog）使用 RTN（rounding-to-nearest）舍入 + 对称量化网格，存在三个根本缺陷：
  1. **对称量化网格**：所有现有对数 PTQ 先取 weight 绝对值再做对称量化，无法匹配 LLM 中常见的非对称 weight 分布（正负比例不均衡）。线性 PTQ 可通过 zero-point 偏移解决，但对数域因零附近的非线性间距无法简单实现。
  2. **对 outlier 高度敏感**：使用 max(|W|) 决定量化范围，单个 outlier 会撑大量化台阶，导致大量正常值被压缩到粗粒度的码字中。
  3. **RTN 舍入次优**：RTN 仅按数值最近原则分配码字，完全不考虑最终任务损失（如激活重建误差）。线性 PTQ 已有 AdaRound/BRECQ 证明了可学习舍入的显著优势，但对数域因 (a) 对数映射非线性、(b) 舍入操作不可微、(c) 混合基离散性，直接迁移可学习舍入不可行。
  - **全栈执行例子（baseline SLogII/DLog + RTN）**：
    - 算法层：取 max(|W|) 确定 s，对 |W| 做 DLog 量化（base-2 或 base-√2），RTN 舍入，固定量化网格
    - 系统框架：论文未明确说明
    - 编译框架：论文未明确说明
    - kernel调度：RTN 舍入无需反向传播，算子直接将 FP16 weight 映射到对数域整数码字
    - 硬件架构：base-√2 乘法需 LUT/multiplier + shifter（AdaLog AE）或 shift-add（Log√2 AE）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LogART 通过 LLR + OHS + HAF 三层创新逐一解决 baseline 缺陷：
  1. **LLR 解决 RTN 次优**：首次将对数域舍入建模为可学习变量 R，用 sigmoid σ(R) 软化为 0~1 之间的选择（floor or ceil）。梯度链：∂L/∂R = 2s·ln2 · M_c ⊙ 2^{-Q_W} ⊙ sign(W) ⊙ [(WX - W̃X)X^T] ⊙ σ'(R) + λ·∂f_reg/∂R。与线性可学习舍入的关键区别是对数域梯度包含指数项 2^{-Q_W}，对小幅值 weight 梯度较小、大幅值 weight 梯度较大——这与对数分布的密度结构一致。
  2. **OHS 逐一解决对称性和 outlier 问题**：(a) ABS 通过自适应边界 l_a 为非对称 weight 分配不同数量的正/负码字——纯 tensor-wise 计算无需校准；(b) SFS 通过块级重建误差搜索缩放因子 s_of 替代 max(|W|)，实现 outlier 自适应裁剪；(c) DBS 自适应分配 base-2:base-√2 码字比例，在硬件效率（base-2 纯移位）和精度（base-√2 细粒度）间分布感知权衡。
  3. **HAF 解决硬件效率与精度矛盾**：用 K-term SDE 展开（如 √2 ≈ 2⁰+2⁻¹）将乘 √2 替换为 shift-add。关键是 HAF 嵌入 LLR 前向传播中，近似误差被梯度下降作为噪声吸收——而非后处理修正。
  - OHS 与 LLR 的协同效应（核心 insight）：论文用三角不等式在 Hessian 加权度量下分解量化误差：||ΔW·H^{1/2}||² ≤ (E₁(OHS) + E₂(LLR))²。E₁ 是量化网格的固有离散化误差（OHS 通过搜索最优 θ*={s_of, n₁, l_a} 最小化），E₂ 是理想投影与 LLR 学习结果的残差（LLR 在 OHS 建立的优质网格上收敛更快更优）。实验证实：OHS+LLR 联合 500 次迭代比纯 LLR 2000 次迭代得到更低的 PPL（31.15 vs 36.27）和更短的总耗时（1.25 min vs 4.00 min）。
  - **全栈执行例子（LogART）**：
    - 算法层：OHS 先搜索 {l_a, s_of, n₁:n₂} 建立最优对数量化网格 → LLR 用 Adam 优化 R 最小化 ||ΔW·X||² + λ·f_reg → HAF 在 forward pass 注入硬件近似噪声 → 收敛后 hard round σ(R)
    - 系统框架：论文未明确说明
    - 编译框架：论文未明确说明
    - kernel调度：在 NVIDIA RTX 5090D GPU 上完成 PTQ 量化（一次离线过程），量化后的 weight 为 INT N-bit 码字，推理时 kernel 使用 LogART AE 设计做 shift-add 而非乘加
    - 硬件架构：LogART AE（Figure 4(e)）——Decoder（组合逻辑）+ Approx 模块（SDE shift-add）+ Shift 模块 + Adder Tree，纯 shift-add 实现，无乘法器。28nm UMC 工艺下面积 53.2 µm²、功耗 3.45 µW（比 BRECQ AE 减少 ~44% 面积和 ~45% 功耗）
