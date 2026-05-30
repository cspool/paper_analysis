## AffineQuant Affine Transformation Quantization for Large Language Models

- baseline方法是什么？
  Baseline 是 **OmniQuant**（Shao et al., 2023），当时 LLM PTQ 中等价变换的 SOTA 方法。OmniQuant 的核心流程：(1) 逐 transformer block 优化，引入 learnable scale（对角矩阵）和 learnable shift（平移向量）两种等价变换；(2) 优化目标为 block 输出在量化前后的 MSE；(3) scale 和 shift 通过梯度下降联合优化，使用 Hessian-guided 学习率；(4) 变换合并入相邻层以保证推理无额外开销。OmniQuant 还比较了 AWQ（仅缩放变换，per-channel scale 由激活统计量确定）、SmoothQuant（手动设计的 scale，将激活量化难度迁移到权重）、RPTQ（per-cluster 重排等价于置换矩阵变换）。

  OmniQuant 等方法的**核心缺陷**是：优化空间仅限于对角线缩放（scale）和平移（shift），即权重矩阵 W 的每个 output channel 只能被统一缩放和平移，不能改变 channel 内部各维度的相对关系。这导致在低比特或小模型场景下，量化的固定点（2ⁿ-1 个量化级别）与权重分布不匹配，大量信息因无法重分布而丢失，量化误差显著增大。直观上如图 1 所示：scaling 仅能做统一的线性拉伸/压缩，translation 仅能做整体平移，两者都无法将二维向量的各维度分别对齐到量化固定点；而 affine 变换则可以实现任意维度的重新分布。

  Baseline（OmniQuant）全栈执行例子（LLaMA2-7B w4a4 量化）：
  - 算法pipeline：加载 FP16 LLaMA2-7B → 逐 transformer block：初始化 scale s=1（对角矩阵）、shift δ=0 → 校准集前向传播缓存输入 → block 输出 MSE loss 计算 → 梯度下降更新 s 和 δ → 量化权重 Q(sW) → 合并 scale 入 LayerNorm → 逐 block 重复至全模型量化完毕。C4 PPL=18.02，WikiText2 PPL=14.26。
  - 系统框架：PyTorch + HuggingFace Transformers，单卡 Nvidia A100 GPU。校准集：WikiText2 训练集 128 segments × 2048 tokens。
  - 编译框架：论文未明确说明（使用标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（量化后权重以 INT4 格式存储，推理时通过标准 dequantize→FP16 GEMM 执行，使用 PyTorch 原生算子）。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AffineQuant**，用一个完整的**仿射变换矩阵 A**（不再是 restricted 的对角矩阵）替代原有等价变换方法中受限的 scale 向量，极大扩展优化空间。具体设计包括三个方面：

  **(1) 仿射变换矩阵替代对角缩放：扩大优化空间。** 
  OmniQuant 的优化空间是 d 维对角 scale + d 维 shift（共 2d 自由参数），而 AffineQuant 的优化空间是 d×d 维矩阵 A + d 维 shift（共 d²+d 自由参数）。这意味着权重 W 的每一行（output channel）可以在行空间内实现任意线性重组——本质上是对每个 output channel 执行旋转+缩放，使权重向量更好地对齐到量化的 2ⁿ-1 个固定点上。图 1 直观展示了差异：scaling 仅能统一缩放（不能改变方向），translation 仅能平移（不能改变各维度的相对位置），而 affine 变换可以任意旋转和缩放各通道以贴合固定点网格。这直接解决了 baseline 在低比特下优化空间不足的根本问题。

  **(2) Gradual Mask 保证仿射矩阵可逆：解决高维矩阵不稳定问题。**
  仿射变换要求计算 A⁻¹，但 d×d 矩阵（d 可达 4096 以上）在有限校准数据下的自由优化极易退化为奇异矩阵（不可逆）。论文基于 Levy-Desplanques 定理（严格对角占优矩阵必可逆）提出 Gradual Mask（GM）方法：将 A 初始化为对角矩阵（严格对角占优平凡满足），在训练早期冻结所有非对角线元素为零，随着 epoch 推进逐步释放靠近对角线的元素参与优化。具体地，在第 e 个 epoch，只允许 |i-j| ≤ (e/t)·hidden_size 的非对角线元素更新，且更新幅度由稳定性因子 α（<1）抑制。GM 在前向通过 Hadamard 积缩小非对角线元素幅度保证 A* 可逆，在反向作为学习率调节器抑制非对角线参数更新速率。OTA-125M w3a16 无 GM 时 WikiText2 PPL 达 53.52（vs 有 GM 30.17），LLaMA-7B w2a16 无 GM 直接 NaN（训练崩溃）。

  **(3) 仿射变换与平移正交互补。**
  平移变换 b 是全局的：v → v + b。仿射变换 A 是旋转+缩放：v → Av。两者数学正交，可以同时施加而不互相干扰：v → Av + b。AffineQuant 同时学习 A 和 δ（shift），在 transformer block 级别优化 argmin_{A,δ} ||f_i(X,W) - f_i((X-δ)A⁻¹, Q(AW), b+δW)||²。

  论文方法全栈执行例子（LLaMA2-7B w4a4 量化）：
  - 算法pipeline：加载 FP16 LLaMA2-7B → 逐 transformer block：① 对每个线性层初始化 A 为对角矩阵（对角线=SmoothQuant scale）、δ=0；② 每 epoch：计算 GM（从中心对角线逐步向外释放），A* = A∘GM，A_inv = inv(A*)，X_t = (X-δ)A⁻¹，W_t = Q(A*W)，block 前向计算 MSE loss，梯度下降更新 A 和 δ（GM 抑制非对角线更新）；③ 多 epoch 后合并：W_final = Q(AW)，bias_final = b+δW，对角 A（LayerNorm 后）合并入 LN。C4 PPL=15.76（OmniQuant=18.02，↓2.26），WikiText2 PPL=12.69（OmniQuant=14.26，↓1.57）。LLaMA-30B w4a4 6-task zero-shot avg=58.61%（OmniQuant=56.63%，↑1.98%）。
  - 系统框架：基于 OmniQuant 代码修改 → PyTorch + HuggingFace Transformers → 单卡 Nvidia A100 GPU → 矩阵求逆使用 PyTorch linalg.inv（float/double 精度）。校准集：WikiText2 128 segments × 2048 tokens。优化参数（lr、epoch、clipping）对齐 OmniQuant。
  - 编译框架：论文未明确说明（使用标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明。推理时将 A 和 δ 合并入权重和 LayerNorm 参数，最终仅使用标准 INT4 group-wise 量化权重和 FP16 激活进行 GEMM 推理，无额外 kernel 需求。
  - 硬件架构：论文未明确说明。

  关键设计动机映射：
  - OmniQuant 仅对角 scale 优化空间有限（d 自由参数）→ AffineQuant 用完整仿射矩阵 A（d² 自由参数）极大扩展优化空间
  - 高维矩阵优化不稳定（易奇异）→ Gradual Mask + Levy-Desplanques 定理确保矩阵始终保持严格对角占优
  - Low-bit quantization 下固定点与权重分布不匹配 → 仿射变换旋转权重通道以重新分布，使所有维度都贴合量化固定点
