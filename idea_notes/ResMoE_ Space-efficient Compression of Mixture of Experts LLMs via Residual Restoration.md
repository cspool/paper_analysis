## ResMoE: Space-efficient Compression of Mixture of Experts LLMs via Residual Restoration

- baseline方法是什么？
  - Baseline 有三类 MoE 压缩方法：(1) **Expert Merging**：M-SMoE、MEO、OneS 等将多个 expert 合并为更少的 expert（如 8→2），直接减少 expert 数量。Git Re-Basin、OT Fusion 通过 permutation + optimal transport 对齐权重后合并。核心缺陷：直接减少 expert 数量导致各 expert 的专业知识大量丢失，且通过 OSE（Oblivious Subspace Embedding）框架理论分析证明，压缩维度 d < O(p log p/ε²) 时合并误差不可忽略；(2) **Expert Pruning**：基于重要性评分移除整组 expert 权重，依赖 calibration data 的 i.i.d. 假设；(3) **Direct Compression**：unstructured pruning、structured pruning、Wanda、truncated SVD 直接对每个 expert 独立压缩，未利用 expert 间的共同模式。
  - 全栈执行例子（Baseline: M-SMoE, Mixtral 8×7B, 压缩率 75%, 4×V100 32GB）：
    - 算法层：M-SMoE 利用 router gating score 分布将 8 个 expert 合并为 2 个——每个合并 expert 是原 expert 权重的加权平均。参数量降至原始的 25%，但丢失 6 个 expert 的专有知识。WikiText PPL 从 3.87 升至 10.45。
    - 系统框架层：HuggingFace transformers + PyTorch，合并后直接加载新 checkpoint 推理。
    - 编译框架层：论文未明确说明。
    - kernel调度层：标准 cuBLAS GEMM，合并后仅 2 个 expert 参与计算。
    - 硬件架构层：4× Tesla V100 32GB。内存从 87GB 降至约 22GB，但精度损失不可接受。核心矛盾：减少内存必须丢 expert，丢 expert 就丢精度——在 baseline 中互斥。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：ResMoE 核心创新是**保留所有 expert，但用 barycenter + compressed residual 存储**——不减少 expert 数量，而减少每个 expert 的表示开销。通过 Wasserstein barycenter 提取所有 expert 的共同模式（barycenter expert），仅压缩残差 Δ_k = T_k W_k - W_ω。
  - 全栈执行例子（ResMoE (UP), Mixtral 8×7B, 75% 压缩率, 4×V100 32GB）：
    - 算法层（解决"合并丢专业知识"和"直接压缩未利用共同模式"的缺陷）：
      - **Barycenter Expert Extraction**：利用 MLP = Σ bottleneck-1 sub-MLP 的视角（E_k(x) = Σ_i W_{k,·,i}^{(2)} · σ(⟨W_{k,i,·}^{(1)}, x⟩ + b_{k,i}^{(1)}) + b_k^{(2)}），将每个 MLP 的每一行作为一个"粒子"构造均匀分布 μ_k 在 W_k 的行上，求解 Wasserstein barycenter: μ_ω = argmin (1/N) Σ W_2^2(μ_k, μ_ω)。Proposition 4.1 证明 W_ω + T_k = p_I · OT(μ_k, μ_ω) 是优化问题 min (1/N) Σ [||T_k W_k - W_ω||_F^2] 的最优解。
      - **对比 baseline**：Baseline 合并 8→2 个 expert 直接丢专家；ResMoE 保留 8 个 expert，以 barycenter + residual 表示。barycenter 捕获共同模式，残差 Δ_k 独立编码各 expert 的差异化信息。残差矩阵 Δ_k 的权重幅值远小于原始 W_k（大部分值接近 0），同样 75% sparsity 下信息损失远小于直接压缩 W_k。
      - 结果：Mixtral PPL 3.87 → ResMoE UP 5.38（vs M-SMoE 10.45, vanilla UP 13.03）。LAMBADA ACC 74.05 → ResMoE UP 69.44（vs M-SMoE 58.57, vanilla UP 36.10）。
    - 系统框架层：PyTorch + HuggingFace transformers。压缩离线完成（one-shot, <1 day for Mixtral vs OT Fusion >4 days），推理时动态恢复 expert。标准推理框架无需修改。
    - 编译框架层：论文未明确说明。
    - kernel调度层：标准 cuBLAS GEMM。ResMoE 的 runtime（38.85s/39.44s Mixtral Winogrande）几乎与原始模型相同。ResMoE (SVD) 的 FLOPs 从 3.26 降至 2.73 TFLOPs。
    - 硬件架构层：同 baseline V100/A100。ResMoE (SVD) 将 Mixtral 单层 MoE 内存从 5,376MB 降至 2,016MB（Table 10），DeepSeekMoE 单层从 2,112MB 降至 561MB——overhead 随 expert 数增加而摊薄。
  - 解决 Baseline 缺陷的方式：
    1. **针对"合并丢专业知识"**：保留所有 expert，barycenter + residual 各自独立编码。WikiText PPL: 3.87 → ResMoE UP 5.38 vs M-SMoE 10.45（~2× lower PPL）。
    2. **针对"直接压缩未利用共同模式"**：barycenter 使残差 Δ_k 接近 0，压缩残差几乎不丢信息。vanilla UP PPL 13.03 vs ResMoE UP PPL 5.38（同 75% sparsity）。
    3. **针对"逐层对齐计算开销"**：ResMoE 基于 MLP = sub-MLP ensemble 的分布视角一次性提取 barycenter（同时对齐 W^{(1)} 和 W^{(2)}），避免 layer-by-layer 策略的多次 permutation。Mixtral: <1 天 vs OT Fusion >4 天。
    4. **通用性**：encoder-decoder (Switch)、decoder-only (Mixtral)、fine-grained MoE (DeepSeekMoE 64 experts) 均有效。One-shot、data-agnostic、无需 retraining。
