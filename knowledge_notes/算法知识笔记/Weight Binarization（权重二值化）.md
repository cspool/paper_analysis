## Weight Binarization（权重二值化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Binarization 将浮点权重压缩为仅 1 bit（±1 值），是最极端的量化形式。标准一阶二值化：对于权重 W ∈ R^{n×m}，先计算行均值 μ = mean(W)，做中心化 W̃ = W - μ 使分布对齐零均值，然后取 α = mean(|W̃|, axis=1) 作为行缩放因子，B = sign(W̃) 为二值矩阵，重构 Ŵ = αB + μ。量化误差 L₁ = ||W - Ŵ||²。二阶二值化（Ŵ = α₁B₁ + α₂B₂ + μ）用两个二值矩阵和两个缩放因子更好地逼近原始权重，通过枚举 {±α₁±α₂} 四种组合选最近邻确定 B₁,B₂ 元素。二值化不改变推理时的计算范式（仍以 FP16 GEMM 执行），但因权重以 1-bit 存储，压缩比可达 ~16×（vs FP16）。二值化 PTQ 的关键挑战是分布偏移（二值化后均值不对齐）和列间偏差（某些列值远大于其他列）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B 某 Linear 层权重 W ∈ R^{4096×4096} 的一阶二值化为例：
```
μ = mean(W, dim=1)             # 行均值, shape (n,)
W_centered = W - μ              # 零均值化
α = mean(|W_centered|, dim=1)   # 行缩放因子, shape (n,)
B = sign(W_centered)            # 二值矩阵 ±1, shape (n,m)
Ŵ = α.unsqueeze(1) * B + μ.unsqueeze(1)   # 重构 W
```
二阶二值化：候选向量 V = {-α₁-α₂, -α₁+α₂, +α₁-α₂, +α₁+α₂}，对每个元素选最接近 (W-μ)[i,j] 的候选值，确定 B₁,B₂ 对应元素。推理时：`W_deq[i,j] = α₁[i]·B₁[i,j] + α₂[i]·B₂[i,j] + μ[i]`，标准 FP16 GEMM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch PTQ 实现：基于 Hessian 选出 salient columns → 二阶二值化 salient → 一阶二值化 non-salient → block-wise OBC 补偿。BiLLM（ICML 2024）是首个在 LLM 上实现 ~1.1-bit 的 PTQ 方法，ARB-LLM（ICLR 2025）进一步引入交替精炼（T=15 迭代）。推理时权重以 packed 1-bit 格式存储，通过 dequant 后 FP16 GEMM 执行。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- Binarized Diffusion Model for Image Super-Resolution
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models
- PB-LLM Partially Binarized Large Language Models

PB-LLM 提出**部分二值化**范式：不是全部二值化，而是检测保留 5%-30% salient weights 为 INT8，仅二值化剩余权重。关键发现：已有全部二值化方法（BNN, XNOR, Bi-Real, ReCU, FDA）直接应用于 LLM 导致完全崩溃（< random guess）。PB-LLM 使用 column-wise α* = ||w_F||₁/n 作为最优缩放因子（与 XNOR-Net 一致），结合 salient weights frozen 进行 QAT。

在 BinaryDM 中，权重二值化采用 Evolvable-Basis Binarizer (EBB)：通过双基残差结构 w_EBB^bi = σ_I*sign(w) + σ_II*sign(w - σ_I*sign(w)) 在训练初期增强表征，再通过正则化驱动 σ_II→0 平滑过渡到标准单基二值化 w^bi = σ_I*sign(w)。BinaryDM 是首个将 DM 权重推至真正 1-bit 的方法，W1A4 下实现 15.2× OPs 和 29.2× 存储节省。

扩散模型二值化的独特挑战：BI-DiffSR 首次将二值化应用于扩散模型 SR。与 LLM 二值化不同，扩散模型面临 (1) UNet 结构维度变化导致 identity shortcut 无法传递 FP 信息，(2) skip connection 中 encoder/decoder 特征值域差异巨大导致融合困难，(3) 多步迭代中激活分布随 timestep 剧烈变化。BI-DiffSR 通过 CP-Down/CP-Up（保持 ResBlock 维度一致以允许 shortcut）、CS-Fusion（channel shuffle 平衡值域后融合）、TaR/TaA（MoE 风格分组 bias+RPReLU 适配 timestep 变化）解决上述问题。二值化计算通过 XNOR + bit-count 替代浮点 MAC，理论节省 32x 内存和 64x 计算。

---
