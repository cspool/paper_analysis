## Scaling Law for Quantization-Aware Training

- baseline方法是什么？
  Baseline 是现有 QAT 缩放定律 [Frantar et al. 2025, Kumar et al. 2024]，其核心建模方式为在 Chinchilla 缩放定律 L(N,D) = A/N^α + B/D^β + E 中引入 Effective Parameter Multiplier（EPM）eff(C) 乘以参数项 N：L(N,D) = A/(N·eff(C))^α + B/D^β + E。由此可推导出量化误差 δ_p(N) = A/(N·eff(C))^α − A/N^α，仅依赖模型参数量 N。

  Baseline 全栈执行例子（Kumar 2024 scaling law, W4A4 QAT per-tensor granularity, Llama-style model）：
  - 算法pipeline：用 per-tensor 量化粒度（activation 全层一个 scale）训练 W4A4 QAT 模型 → 统计不同 N 的最终训练 loss → 拟合 eff(C) 作为 N 无关常数 → 预测其他 N 的量化误差。eff(C) 仅依赖模型架构和压缩类型，不随 N、D、G 变化。然而实际实验中，当 D 从 10B→100B tokens 增长时，W4A4 量化误差平均增加 22%（论文 Figure 4b），baseline 无法捕捉这一趋势。
  - 系统框架：基于 PyTorch + HuggingFace Transformers 的标准 QAT 训练。使用 STE 模拟量化前向，FP32 权重 + fake-quantize。
  - 编译框架：论文未明确说明（标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（标准 CUDA kernel 模拟量化推理）。
  - 硬件架构：NVIDIA A100 GPU。量化格式为 INT4/FP4。

  **Baseline 的核心缺陷：**
  1. **忽略训练数据量 D 对量化误差的影响**：现有缩放定律假定 δ_p 与 D 无关（式 3 中 D 被消除）。但论文实验（Figure 4b）证明 δ_{W4A4} 随 D 增加显著上升（10B→100B 平均 +22%），原因是 QAT 训练中模型参数会"适应"量化误差——更多训练数据意味着更充分的全精度训练收敛，从而放大量化带来的差距。
  2. **忽略量化粒度 G 的影响**：Baseline 未能建模 group-wise 量化粒度对误差的影响，通常使用 per-tensor 或单一固定 G。论文实验显示 finest→coarsest G 的 δ 差异达 0.037（占粗粒度误差的近半数）。不同 G 需分别拟合独立曲线（baseline 需 5 条曲线覆盖 5 种 G），无法统一建模。
  3. **未区分权重/激活误差来源**：Baseline 将量化视为单一压缩比参数，未揭示 W4A4 中权重与激活量化误差的不同行为——激活误差主导（ratio R>1）、但权重误差对 D 更敏感（γ_D: 0.1610 vs 0.0331）。
  4. **未识别激活瓶颈层**：Baseline 对激活量化误差的来源缺乏逐层分析，无法定位 FC2 Proj 输入（kurtosis=89）为根本瓶颈——这是 SwiGLU 输出的系统性 outlier 导致的特异性问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**统一的 QAT 缩放定律框架**，通过三个层次的递进分析解决 baseline 缺陷：

  **(1) 三维统一建模替代单参数建模（解决缺陷 1 和 2）**
  不再修改 N 的有效参数值，而是直接建模量化误差项 δ_p(N, D, G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}，作为 Chinchilla loss 的独立加项。log₂(G) 满足 G=1 时 δ_p=0（无量化）的边界条件。拟合 80 次 W4A4 QAT 实验数据，单条曲线即可覆盖所有 G（vs baseline 需 5 条曲线），且 W4A16/W4A4 预测相对误差分别从 19.3%/8.5% 降至 5.2%/4.7%。

  **(2) 误差解耦：权重 vs 激活独立分析（解决缺陷 3）**
  通过训练 W4A16（仅权重量化）和 W16A4（仅激活量化）两种额外配置解耦误差源，发现 δ_{W4A4} ≈ 0.906 · (δ_{W4A16} + δ_{W16A4})（强相关性）。通过分别拟合 δ_{W4A16} 和 δ_{W16A4} 的缩放定律参数，揭示：
  - 权重量化误差对 D 更敏感（γ_D=0.1610），更多训练数据时需重点优化权重
  - 激活量化误差对 G 更敏感（γ_G=0.9812），粗粒度下需重点优化激活
  - 激活量化误差始终大于权重（ratio R>1），但随 D/N 增大差距缩小

  **(3) FC2 瓶颈识别与混合精度方案（解决缺陷 4）**
  逐层分析 kurtosis（峰度）揭示 FC2 Proj 输入层的 kurtosis=89，远高于 QKV Proj、O Proj、FC1 Proj 等其他层（均 <10）。根源：FC2 输入来自 SwiGLU 非线性输出（gating + SiLU + element-wise multiply），复合非线性运算产生系统性 outlier，即使 QAT 正则化也无法完全消除。方案：FC2 Proj 输入保持 8-bit 量化，其余保持 4-bit。效果：量化误差降 20.5%（G=32）至 42.9%（G=256）；激活误差对 G 的敏感度 γ_G 从 0.9812 降至 0.4471；δ_{W16A4} 与 δ_{W4A16} 的 ratio R 降至 0.85-1.10，两者贡献趋于均衡。

  论文方法全栈执行例子（W4A4 QAT, 595M Llama3-style model, 100B tokens, G=128）：
  - 算法pipeline：
    1. BF16 基线训练：用 OLMo2-Mix-1124 全精度训练 → 记录 L_bf16
    2. W4A4 QAT 训练：插入 AbsMax per-group 量化器（weight: AbsMax, activation: AbsMax for G<256 or LAC for G≥256）→ STE 前向 + 反向 → 记录 L_W4A4
    3. 误差分解训练：W4A16 (weight-only) 和 W16A4 (activation-only) 分别训练
    4. 缩放定律拟合：用 80 次实验数据通过 Huber loss + L-BFGS 拟合式 5 的参数 k, γ_N, γ_D, γ_G
    5. FC2 Proj 分析：统计每层 kurtosis → 识别 FC2 Proj input → 实施 8-bit FC2 Proj + 4-bit others 混合精度 QAT → 重新拟合缩放定律
    6. 外推验证：973M 模型 100B/200B tokens 预测 vs 实际误差
  - 系统框架：PyTorch + HuggingFace Transformers，基于 OLMo2 训练 pipeline。LR 实验证明 4-bit QAT 无需高于 FP 训练的 LR（量化误差在 [0.60, 0.65] 内几乎恒定），可直接复用全精度训练超参数。
  - 编译框架：论文未明确说明（标准 PyTorch eager mode，fake-quantization 推理模拟）。
  - kernel调度：论文未明确说明（无自定义 kernel，使用标准 CUDA fake-quantize 模拟 INT4 GEMM）。
  - 硬件架构：NVIDIA A100 GPU。总计 268 次实验消耗 276K GPU-hours。

  全栈执行例子对比基线改进：
  - 量化误差预测：从 5 条独立曲线（每种 G 一条）→ 1 条统一曲线涵盖所有 G
  - 误差分解：从 unknowing（无法区分权重 vs 激活贡献）→ 明确 δ_{W16A4} > δ_{W4A16}（R>1），且提供 D/N-G 二维 heatmap 指导优化方向
  - FC2 瓶颈：从无法定位 → 明确 FC2 Proj input（SwiGLU 输出）为根本瓶颈，8-bit 处理可降误差 20-43%
  - EPM 量化：从常数值 → EPM(N, D, G) 动态值（式 13），W4A4 EPM 始终 >0.5（4-bit QAT trade-off 优于 8-bit QAT），FC2 8-bit 后提升 0.06-0.14

  关键设计动机映射：
  - 现有 QAT 缩放定律仅依赖 N → 直接建模 δ_p(N,D,G) 独立相加项，纳入 D 和 G
  - 均匀粒度建模（per-tensor）→ Group-wise 量化引入 log₂(G) 参数，统一拟合
  - 量化误差来源未知 → W4A16/W16A4 解耦训练 + 独立缩放定律拟合 + ratio R 分析
  - 无法定位激活瓶颈 → 逐层 kurtosis 分析 → FC2 Proj input（SwiGLU 输出 outlier）→ 混合精度
  - EPM 与 D/G 无关（baseline）→ 式 13 量化 EPM 随 N, D, G 演化，指导实际部署决策
