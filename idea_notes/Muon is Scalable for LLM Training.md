## Muon is Scalable for LLM Training

- baseline方法是什么？
  Baseline 是 AdamW 优化器，作为当前大规模 LLM 训练的事实标准。AdamW 的核心机制：(1) 维护两个动量 buffer（一阶动量 m_t 和二阶动量 v_t）；(2) 通过自适应学习率 η_t / (√(v_t) + ε) 进行逐元素更新；(3) 从 steepest descent 视角看，AdamW 是 Max-of-Max norm 约束下的最陡下降，其 norm constraint 动态变化。
  
  Baseline 缺陷：
  (1) **计算效率不足**：AdamW 的逐元素自适应更新虽然稳定，但计算效率受限于二阶矩估计和 element-wise 操作。相比矩阵级正交化更新，AdamW 在相同计算预算下达到的 loss 更高——scaling law 拟合显示 AdamW 的 loss-C 曲线为 2.608 × C^(-0.054)，高于 Muon 的 2.506 × C^(-0.052)。
  (2) **优化方向多样性不足**：AdamW 的逐元素更新缺乏对矩阵整体结构的考虑。从 steepest descent 角度看，AdamW 使用的 norm constraint 是动态变化的 Max-of-Max norm，而非更合理的 operator norm（spectral norm），导致权重矩阵在低维主导方向上过拟合。
  (3) **对非矩阵参数无特别优势**：对 RMSNorm、embedding 等非矩阵参数，AdamW 的逐元素更新是合理的。但对于矩阵参数（attention 投影、FFN），缺乏矩阵级结构利用。

  全栈执行例子（AdamW 训练 Llama 密集模型）：token → embedding → 逐层 attention（QKV 投影 [H,H] 矩阵 × 输入 → attention → output 投影 [H,H]）→ FFN（[H, 2.6H] up/gate → SwiGLU → [2.6H, H] down）→ RMSNorm → LM head → cross-entropy loss → 反向传播得到各矩阵梯度 G → AdamW：m_t = β₁ m_{t-1} + (1-β₁) G, v_t = β₂ v_{t-1} + (1-β₂) G² → 更新 ΔW = -η m̂_t / (√(v̂_t) + ε) + λW → 每个矩阵元素独立更新，无矩阵级正交性约束 → 权重矩阵的奇异值分布逐渐集中在少数主导方向 → SVD entropy 偏低 → 模型容量利用不充分。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出扩展 Muon 优化器进行大规模 LLM 训练，核心理念：**对矩阵参数使用矩阵正交化更新替代逐元素自适应更新**。Muon 将 momentum 矩阵通过 Newton-Schulz 迭代进行近似正交化（≈ (M M^T)^(-1/2) M = U V^T），使得更新矩阵的奇异值全部为 1，确保在所有方向上均匀更新。从 steepest descent 角度，Muon 提供的是 spectral norm（operator norm）约束，对矩阵参数而言比 AdamW 的动态 Max-of-Max norm 更合理。三项关键技术保证可扩展性：

  **对应缺陷 1（计算效率不足）→ 矩阵正交化 + Consistent Update RMS**
  - Muon 的正交化更新迫使参数在所有奇异向量方向上等强度学习，避免在少数主导方向过拟合，使相同 FLOPs 下的有效学习更充分。Scaling law 显示 Muon 仅需 ~52% FLOPs 即可匹配 AdamW 性能。
  - Consistent Update RMS：Lemma 1 证明 shape [A,B] 矩阵的 Muon 理论更新 RMS = √(1/max(A,B))，导致不同 shape 矩阵更新尺度不一致。通过缩放因子 0.2·√(max(A,B)) 统一所有矩阵参数的更新 RMS，鲁棒的训练行为消除了针对不同 shape 矩阵的手动调参需求。

  **对应缺陷 2（优化方向多样性不足）→ Spectral Norm 约束 + Weight Decay**
  - Muon 的 spectral norm 约束（当 Newton-Schulz 精确计算时）比 AdamW 的 Max-of-Max norm 更匹配权重矩阵作为 operator 的数学本质。SVD entropy 实验证实：Muon 训练的权重矩阵在 90%+ 情况下 SVD entropy 高于 AdamW，singular value 分布更平坦，意味着模型在学习更丰富的特征方向。
  - Weight decay 解决原始 Muon 在 long-training regime 中权重 RMS 持续增长问题：vanilla Muon 初期收敛快但长期权重发散超出 bf16 范围，加入 λW 项后 Muon 在过训练区间持续优于 AdamW。

  **对应缺陷 3（分布式兼容性）→ Distributed Muon + ZeRO-1**
  - 分布式 Muon（Algorithm 1）：在 Megatron-LM 的 ZeRO-1 框架下，增加 bf16 DP Gather 操作从分片梯度恢复到全梯度矩阵进行 Newton-Schulz 迭代，计算完成后丢弃非本地分片。额外通信开销仅为 Distributed AdamW 的 0~25%（在多个 DP 组下几乎无感知）。
  - Muon 仅需 1 个动量 buffer（vs AdamW 的 2 个），内存消耗减半。
  - 非矩阵参数（RMSNorm、embedding、LM head）继续用 AdamW 处理，两优化器共享 lr 和 weight decay，无缝集成。

  全栈执行例子（Muon 训练 Moonlight MoE 模型）：token → embedding (AdamW) → 逐层 attention：QKV 投影矩阵 [H,H] → Muon 正交化更新使 QKV 学习多样化 query/key/value 子空间 → attention → output 投影 [H,H] (Muon) → FFN experts：各 expert 的 up/gate/down 矩阵 (Muon) → router → top-k 选择 (AdamW 对 router 权重) → shared expert (Muon) → RMSNorm (AdamW) → 反向传播得到各矩阵梯度 G → Muon: reduce-scatter(G) → momentum → gather → Newton-Schulz 5 步迭代 → 0.2·√(max(A,B))·O_t + 0.1·W → 结果：所有矩阵更新奇异值均匀（SVD entropy 高），router 权重多样性显著提升（专家选择更差异化），最终在 5.7T tokens 后 MMLU=70.0, GSM8K=77.4, HumanEval=48.1。
