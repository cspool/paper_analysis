## PARQ Piecewise-Affine Regularized Quantization

- baseline方法是什么？
  - **STE/BinaryConnect（Courbariaux et al. 2015）**：QAT 的标准 baseline，使用硬量化映射 Q(·)（图 1 的阶梯函数）在训练全程对权重进行硬量化。更新规则：u^{t+1}=u^t-η_t∇f(Q(u^t), z^t)，w^{t+1}=Q(u^{t+1})。其中 u^t 作为全精度隐变量累积在量化点 w^t=Q(u^t) 处的梯度。STE 在反向传播中将 dQ/du=0 替换为 dQ/du=1，本质是一个启发式近似，缺乏严格收敛理论支撑。缺陷：(1) 全程硬量化导致训练动态不稳定（如图 12 中的 sudden accuracy drops），在小模型/极低位宽（1-bit/ternary）下尤其明显；(2) 仅在特殊情形下有弱收敛结果（如期望收敛而非最后迭代收敛），理论保证不足；(3) 硬量化图（阶梯函数）对应非凸 indicator 函数 δ_Q 的 proximal map，无法享有凸优化的收敛性质。
  - **BinaryRelax（Yin et al. 2018）**：用 W 形非凸正则化的 proximal map（图 9b）替代硬量化映射。slanted segment 斜率逐步减小至 0，通过放松量化约束来稳定训练。缺陷：(1) 使用的正则化是非凸的（W 形），梯度方法容易在初始权重落入"错误山谷"时被困于局部最优；(2) 同样缺乏最后迭代收敛保证，仅提供平均迭代的收敛结果——而平均迭代通常不满足量化结构；(3) 非凸正则化无法享受凸优化的全局收敛性质。
  - 全栈执行例子（以 STE/BinaryConnect 训练 1-bit ResNet-20 为例）：
    - **算法层**：加载 FP32 ResNet-20 权重 → Q(u^t) 将每个权重二值化投影到 {±q}（q=|u|₁/d）→ 在二值化权重 w^t 处计算交叉熵损失梯度 → STE 直接将梯度传递给隐变量 u^{t+1}=u^t-η_t∇f(w^t)→ 下一次迭代再次硬量化。全程在二值权重处计算梯度，没有从软到硬的渐进过程。
    - **系统框架层**：PyTorch，标准 SGD optimizer，GPU 训练（200 epochs on CIFAR-10）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（baseline 为纯 PyTorch 训练）。
  - Baseline 核心缺陷：**硬量化全程使用 → 训练初期不稳定**；**缺乏凸性 → 无全局收敛保证**；**弱收敛理论（仅平均迭代收敛）→ 量化结构在理论分析中无法得到保证**；**非凸正则化的 W 形 valley → 初始点不良时可能陷入局部最优**。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PARQ 通过以下设计解决 baseline 缺陷：
    1. **凸 PAR 替代非凸/无正则化**：构建凸分段仿射正则化函数 PAR(w)=max_k{a_k(|w|-q_k)+b_k}，其中斜率 0≤a_0<a_1<...<a_m=+∞ 严格递增。该函数是凸的（有限个线性函数的最大值），但非光滑点（±q_k）的自然聚类效应使其能有效诱导量化。相比之下：(a) STE 无显式正则化，对应非凸 indicator δ_Q；(b) BinaryRelax 使用的 W 形正则化是非凸的。凸性确保了全局收敛性质和对初始点不敏感。
    2. **AProx 聚集 proximal 算法替代 Prox-SGD**：AProx 的关键在于用累积步长 γ_t=Ση_s 缩放 proximal map（而非 Prox-SGD 的单步步长 η_t）。由于 γ_t→∞，prox 中的 flat segments（长度 γ_t λ(a_k-a_{k-1})）不断增大，sloped segments 相对缩小，proximal map 从软量化渐近到硬量化（图 7→图 8）。这解决了 Prox-SGD 因 η_t→0 导致正则化消失的问题。
    3. **最后迭代收敛理论保证**：证明了 AProx 的最后迭代（last-iterate）收敛率 O(ln(t)/√t)，与平均迭代收敛率匹配。这比 BinaryRelax/Dockhorn et al. 仅证明平均迭代收敛更强——因为平均迭代通常不被量化，而最后迭代可以在渐近阶段被保证量化。
    4. **LSBQ 在线估 Q + 独立斜率 schedule**：避免预设量化值和正则化强度的难题。LSBQ 从隐变量 u^t 中在线估计目标量化值 {q_k}；独立逆斜率 schedule ρ_t^{-1}（cosine decay）从 1→0，使 proximal map 从近似 identity（训练早期，平滑过渡）→ 硬量化（训练末期）。这使训练初期 PARQ 接近全精度训练（loss 曲线靠近 FP），后期自然过渡到量化状态。
  - **如何解决 Baseline 缺陷**：
    - **训练稳定性**：PARQ 的渐进软→硬量化（而非全程硬量化）使训练更稳定。图 12 显示 PARQ 无 STE 的 sudden accuracy drops，训练 loss 曲线更平滑。
    - **凸优化保证**：凸 PAR 确保全局收敛性质，不会像 BinaryRelax 的 W 形非凸正则化那样陷入局部最优。
    - **更强的理论保证**：最后迭代收敛结果保证最终模型权重被量化（而非平均迭代），理论更贴近实际需求。
    - **实用自适应性**：无需为不同模型/数据集调优 q_k、a_k、λ，LSBQ+schedule 自动适应。
  - 论文方法全栈执行例子（以 PARQ 训练 2-bit DeiT-Ti 为例）：
    - **算法层**：
      1. 初始化：u¹=w¹（随机初始化的 FP32 权重）。
      2. 每轮迭代：(a) 在 w^t 处计算 mini-batch 梯度 g^t；(b) u^{t+1}=u^t-η_t g^t（累积纯梯度）；(c) LSBQ(u^{t+1}, n=2) 估计 Q^{t+1}={±q₁,±q₂}（q₁≈v₁-v₂, q₂≈v₁+v₂），值从随机初始化时的小量快速膨胀→缓慢收缩（图 13）；(d) w^{t+1}=prox_PARQ(u^{t+1}, Q^{t+1}, ρ_t)，其中 ρ_t^{-1} 从 1→0：(早期) slope≈1，prox 接近 identity → 权重几乎未被量化，训练接近 FP；(中期) slope 增大，prox 呈现 slanted+flat 混合结构（图 11 中），soft quantization → 权重开始向 Q 中离散值聚类；(晚期) slope→∞，prox 收敛为 hard quantization（图 11 右），权重完全量化。
      3. 最后 20 epochs：lr 固定在 1e−8，所有方法均在硬量化模式下微调。
    - **系统框架层**：PyTorch 实现，SGD (ResNet) / AdamW (DeiT)，标准数据增强 pipeline。开源代码 `parq` 包可直接 pip install 使用。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PARQ 是纯训练算法，不涉及推理时的编译或 kernel 修改。量化后的模型可用标准 PyTorch runtime 推理，或进一步结合 TensorRT/其他量化推理框架加速。
  - 关键理论洞察：**凸性 + 非光滑性 → 量化诱导**。最优条件分析揭示：在 PAR 正则化的最优解处，与非量化区间 (q_{k-1}, q_k) 对应的梯度值只能是 2m 个离散值 {±λ a_k} 之一，而几乎所有其他梯度值可通过将权重置于 Q 的 2m+1 个离散值上来平衡。这意味着最优解处的权重"大概率"聚合在离散量化值上——这是 PARQ 的数学基础。此外，AProx 可被解读为 STE 的渐近形式：当 γ_t→∞ 时，prox_PAR 收敛到硬量化映射 Q(·)，此时 AProx 退化为 BinaryConnect/STE。
