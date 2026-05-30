## Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression

- baseline方法是什么？
  Baseline 是 **SVD-LLM**（Wang et al., 2024b），SVD-based per-layer weight compression 的 SOTA 方法。SVD-LLM 流程：(1) 对每层的每个权重矩阵独立处理；(2) 引入 whitening matrix 捕获激活中的 outlier 信息来调整权重矩阵，即 S(S^T) = cholesky(X^T X)，用 S 缩放权重后做 SVD；(3) 截断小奇异值实现压缩；(4) 压缩比 k 由目标压缩率决定。

  Baseline 的核心缺陷：**仅对单层内权重矩阵做独立压缩，完全忽略了跨层权重之间的相似性**。LLaMA/LLaMA2 等 decoder-only transformer 的不同层中，同类型权重矩阵（如 W_K, W_Q, W_V）可能具有相似的参数分布，独立 SVD 无法利用这种跨层冗余实现进一步压缩。在相同压缩比下，跨层共享可降低总体的 Frobenius loss（实验证实在 W_K 上 9-10 层共享后 loss 从 66682.9 降至 61817.3）。

  Baseline（SVD-LLM）全栈执行例子（LLaMA-7B, 20% 压缩）：
  - 算法pipeline：加载 FP16 LLaMA-7B → 逐层逐矩阵：评估 S（256 WikiText-2 样本, FP64）→ SVD(S·W) → 截断 k 个奇异值 → Ŵ = S^{-1}U_kΣ_kV_k^T → 推理时 Ŵ @ X。w/o basis sharing: PPL=7.94（WikiText-2）。
  - 系统框架：HuggingFace Transformers + PyTorch，两块 A100 GPU。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（标准 PyTorch FP16/BF16 GEMM）。
  - 硬件架构：论文未明确说明（NVIDIA A100，无自定义硬件）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Basis Sharing**，核心设计：将跨层同类型权重矩阵水平拼接为一个合并矩阵，对合并矩阵做一次性 SVD，分解为**共享基向量 B''**（所有层共用）+ **每层独有系数矩阵 C^(i)**。通过三个关键设计解决基线缺陷：

  **(1) 跨层拼接+SVD共享基向量（解决跨层冗余未被利用的问题）**
  SVD-LLM 对单层 W 分解得到 W ≈ U_kΣ_kV_k^T，U_kΣ_k（基矩阵）和 V_k^T（系数矩阵）都只为单层服务。Basis Sharing 将 n 层拼接为 W_cat ∈ R^{d1 × n·d2}，SVD 后得到共享基矩阵 B'' = S^{-1}U_k'Σ_k' 和系数 C（前 d2 列属第 1 层，后 d2 列属第 2 层，...）。共享基向量意味着所有层共享相同的"参数原型"，不同层通过不同的系数组合来表达各自的权重功能，区别仅在于系数。这实现了比 independent SVD 更低的 Frobenius loss（共享后 loss 可能小于独立压缩之和），从而在相同压缩比下获得更好的 model quality。

  **(2) 矩阵类型筛选（避免在无关矩阵上做有害共享）**
  并非所有矩阵类型都适合跨层共享。论文通过 Frobenius loss 热力图分析发现：W_K, W_Q, W_V, W_Up, W_Gate 共享后 Frobenius loss ≤ 独立 SVD 之和（对角块外颜色 ≤ 对角块），适合共享；W_Down（高维→低维投影，拼接后 rank 增大导致截断损失更大）和 W_O 共享后 loss 反而增大，不适合共享。这个设计避免了在错误的矩阵类型上强制共享导致的性能退化。

  **(3) 相邻层分组策略（最小化 group 内 Frobenius loss）**
  层分组不是任意的：相邻层共享基矩阵产生的 Frobenius loss 最小，因为相邻层在 transformer 中通常处理相似特征层次的特征。默认按 2 层一组顺序分组（1-2, 3-4, ...）。消融实验验证：2 层分组在 ≥30% 压缩比下优化，4-5 层在 ≤30% 下较优；LoRA 微调后更多层（甚至 32 层全共享）也在可接受范围内。

  论文方法全栈执行例子（LLaMA-7B, Basis Sharing, 20% 压缩, 2 层分组）：
  - 算法pipeline：加载 FP16 LLaMA-7B → 逐类型矩阵（W_K/Q/V/Up/Gate 共享；W_Down/W_O 独立 SVD-LLM）：① 垂直拼接相邻 2 层输入 X → ② 计算 S = cholesky(X^T X)^{1/2} → ③ 水平拼接 2 层权重 W_cat → ④ SVD(S·W_cat) → ⑤ 截断 k = (d1·d2·2·0.8)/(d1+2·d2) → ⑥ B'' = S^{-1}U_kΣ_k（共享基）, C = V_k^T（每层各 d2 列系数）→ 推理：Y_i = X_i·B''·C^(i)。WikiText-2 PPL=7.74（SVD-LLM=7.94）。50% 压缩比下 PPL=19.99（SVD-LLM=23.97, ↓17%）。
  - 系统框架：HuggingFace Transformers + PyTorch，两块 A100 GPU。压缩时间：GPT2 仅需 26.47s（Dynamic Tying 需 13.75h 训练）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（压缩后仍为 FP16 矩阵乘法：先 X_i·B'' 再乘 C^(i)，两次小矩阵乘代替一次大矩阵乘）。
  - 硬件架构：A100 GPU。50% 压缩比下 throughput=1.57× dense 模型（batch=512, seq=32）。

  关键设计动机映射：
  - SVD-LLM 不利用跨层相似性 → Basis Sharing 拼接多层的同类型矩阵，SVD 分解后共享基向量
  - 盲目共享所有类型矩阵会导致退化 → Frobenius loss 热力图筛选适合共享的矩阵类型（W_K/Q/V/Up/Gate vs W_Down/W_O）
  - 任意层分组可能导致高 loss → 相邻层分组策略最小化 Frobenius loss
  - 高压缩比（≥40%）下后续层输入偏差累积 → 更新后续层输入以补偿偏差（与 SVD-LLM 相同的补偿策略）
