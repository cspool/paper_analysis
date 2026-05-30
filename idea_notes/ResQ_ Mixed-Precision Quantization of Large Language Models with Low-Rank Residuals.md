## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

- baseline方法是什么？
  Baseline 方法为 SpinQuant、QuaRot、QUIK 等 4-bit 后训练量化方案。以 SpinQuant 为例的全栈执行：
  - **算法层**：使用可学习的旋转矩阵（通过 Cayley 优化）对权重和激活进行旋转，使旋转后的张量对均匀 4-bit 量化更友好，所有通道统一 4-bit。
  - **系统框架层**：基于 HuggingFace Transformers，旋转矩阵可融入相邻权重，避免运行时显式投影开销。使用 GPTQ 做权重优化。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：调用标准 INT4 GEMM kernel（如 CUTLASS）。因所有操作均为统一精度，kernel 选择简单。
  - **硬件架构层**：运行在 NVIDIA A100/RTX 3090 GPU 上，使用 TensorCore。
  Baseline 的核心缺陷：统一 4-bit 量化的误差上界由激活的最大 outliers 决定。旋转虽能部分抑制 outliers，但无法从根本上消除极端值的影响。SpinQuant 在 Meta-Llama-3-8B 上相比 16-bit baseline 仍有约 20% 困惑度退化。QUIK 的 mixed-precision 方案（按 l_∞-norm 选高精度通道）缺乏理论最优性保证。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ResQ 通过 **PCA 驱动的低秩残差混合精度量化** 解决 baseline 缺陷：
  - **算法层**（核心创新）：不依赖启发式 outlier 检测，而是通过 PCA 从理论上找到最小化量化误差的投影基。具体地：对校准激活做 PCA，将特征向量按特征值排列，后 r（=d/8）列对应方差最大的低秩子空间保留 8-bit，前 d-r 列对应低方差的互补子空间量化到 4-bit。Theorem 4.2 从理论上证明 PCA 基选择是最优的——最小化量化误差上界。同时在每个子空间内应用随机旋转（Lemma 4.1 保证旋转后分布近似高斯，进一步降低量化误差）。
  - **系统框架层**：四种投影矩阵（U_A/U_B/U_C/U_D）分别处理不同位置的激活投影：U_A 融入跨 block 边界的权重（后乘 o_proj/down_proj 权重，无运行时开销）；U_B 处理 value 投影并融入 o_proj；U_C 为 key/query 投影，因 RoPE 存在需运行时计算但量化为 8-bit；U_D 用于 FFN 内部激活投影，用 Hadamard 矩阵实现快速变换。权重量化可结合 GPTQ 进一步优化。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：实现 CUTLASS INT4 + INT8 混合精度 GEMM kernel，分别计算低精度和高精度分量的矩阵乘法，结果在 INT32 累加器求和。相比纯 INT4 kernel 仅增加约 14% 延迟。运行时 U_C 投影和 U_D Hadamard 变换均有高效 CUDA kernel 实现。
  - **硬件架构层**：运行在 NVIDIA A100/RTX 3090 GPU 上，利用 TensorCore 加速 INT4/INT8 GEMM。

  对比 baseline 的全栈改进：从一个请求（token 序列）出发，激活 X 进入 decoder block 后，首先经 U_A（已融合到前一层权重中）自动完成投影 → 在注意力块内 query/key 经 U_C 运行时 8-bit 量化投影计算 attention → value 经 U_B 投影后写入 KV cache（4/8-bit 混合精度）→ FFN 内激活经 U_D Hadamard 变换 → 各线性层执行 INT4+INT8 混合精度 GEMM。整个过程将量化误差理论上界最小化，实际达到相比 SpinQuant 4-33% 的 Wikitext 困惑度降低，无需梯度优化（SpinQuant 需 Cayley 优化训练旋转矩阵），且提供通过调节 rank r 实现的 Pareto 最优精度-效率权衡。
