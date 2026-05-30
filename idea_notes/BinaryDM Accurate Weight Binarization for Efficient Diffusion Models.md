## BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

- baseline方法是什么？
  Baseline使用基础XNOR-Net风格权重二值化：w^bi = σ * sign(w)，其中σ = ||w||/n初始化为逐通道可学习浮点标量（Rastegari et al., 2016）。激活使用LSQ逐层量化器（Esser et al., 2019）。训练损失为标准简化变分下界 L_simple = E_t,x_0,ε[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]。全栈执行例子：全精度DDIM/LDM权重 → 逐通道sign(w)二值化为{-1,+1} → σ可学习缩放因子调整幅度 → LSQ逐层激活量化为低bit整数 → 前向卷积 o = σ * (a ⊗ sign(w))（⊗仅含整数加法，无乘法） → L_simple损失反向传播 → STE近似sign梯度 ∂L/∂w ≈ ∂L/∂w^bi * 1_{|w|≤1} → 迭代优化。
  Baseline存在两个核心缺陷：(1) **表征能力坍塌**：权重从2^32候选值骤降到2^1，信息熵急剧下降，对生成模型关键表征造成灾难性损害；(2) **优化方向模糊**：离散化sign函数引入显著前向参数误差和反向梯度近似误差，QAT中的精细特征学习受扰，收敛不稳定甚至不可达。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BinaryDM通过EBB（表征角度）和LRM（优化角度）两个技术对应解决baseline两大缺陷：

  **(1) EBB应对表征能力坍塌**：训练第一阶段使用双基二值化 w_EBB^bi = σ_I*sign(w) + σ_II*sign(w - σ_I*sign(w))，通过残差结构将权重候选值从2个扩展到更多组合，显著提升信息熵和表征空间。正则化损失 L_EBB = τ/N * Σ σ_II 驱动高阶基σ_II→0，平滑演化到第二阶段的单基全二值化 w^bi = σ_I*sign(w)。仅应用于首尾各6层（约15%参数），中间层保持vanilla binarizer，减少过渡不稳定性。这使二值化DM从更高信息容量的初始状态开始优化，避免了表征骤然坍塌。

  **(2) LRM应对优化方向模糊**：对全精度DM中间表征通过PCA计算协方差矩阵 C_i = (hw)⁻² * ε̂^FP * (ε̂^FP)^T，特征分解后取前⌈c/K⌉列特征向量E_i作为低秩投影矩阵（K默认为4），将全精度和二值化DM的中间表征同时投影到低秩空间：R_i^FP = ε̂^FP * E_i^(⌈c/K⌉)，R_i^bi = ε̂^bi * E_i^(⌈c/K⌉)。MSE损失 ||R_i^FP - R_i^bi|| 在低秩空间中驱动二值化DM沿主成分方向学习全精度表征，避免高维空间直接对齐导致的优化方向模糊。投影矩阵在首batch计算后固定不变，保证优化方向稳定性。

  全栈执行例子：预训练全精度DDIM/LDM → EBB双基二值化初始化(σ_I, σ_II) → 第一阶段多基卷积 o = σ_I*(a⊗sign(w)) + σ_II*(a⊗sign(w - σ_I*sign(w))) → LRM在每组timestep embedding模块后计算低秩投影对齐：PCA(ε̂_θi^FP) → R^FP, R^bi → MSE loss → 总损失 L_total = L_simple + τ*Σσ_II/N + λ*Σ||R^FP - R^bi||/M → 第二阶段σ_II→0后转换为单基 w^bi = σ_I*sign(w) → W1A4推理（4-bit激活分解为4个1-bit激活+偏置） → 15.2×OPs节省、29.2×存储节省 → Qualcomm Snapdragon 855 Plus实测4.62×加速。
