## Mixture of LoRA Experts

- baseline方法是什么？
  Baseline 为 **Normalized Linear Arithmetic (NLA) Composition**（Eq.2）：对 N 个已训练的 LoRA 进行加权求和 $\hat{\boldsymbol{W}} = \boldsymbol{W} + \sum_{i=1}^{N} w_i \cdot \Delta \boldsymbol{W}_i$，其中 $\sum w_i = 1$。以 DreamBooth + Stable Diffusion V2.1 三概念多主体生成（Dog + Cat + Sunglasses）为例的 LoRA 组合全栈执行路径：
  - **算法层（LoRA 组合推理）**：对每个 Transformer block，所有 N 个 LoRA 的增量权重 $\Delta \boldsymbol{W}_i = A_i B_i$（rank decomposition）按全局统一标量 w_i 线性叠加到预训练权重 W 上，形成组合权重 $\hat{\boldsymbol{W}}$。单个 forward pass 实质等价于用 $\hat{\boldsymbol{W}}$ 进行一次标准推理。Attention 的 Q/K/V/O projection 和 FFN 的 fc1/fc2 均按相同的 {w_i} 权重组合。**权重组合是静态的**：w_i 在推理前确定、所有层共享，不存在层间差异。**无反向传播**：NLA 是纯前向算术操作，不涉及训练或梯度。
  - **系统框架层**：论文未明确说明。标准 Stable Diffusion pipeline（Diffusers/HuggingFace），LoRA 权重通过 PEFT 库加载和 merge（merge_and_unload 或 set_adapter 方式），无调度或并行策略定制。
  - **编译框架层**：论文未明确说明（标准 PyTorch）。
  - **kernel 调度层**：论文未明确说明。合并后的权重矩阵进行标准 GEMM 计算，无定制 kernel。
  - **硬件架构层**：论文未明确说明 GPU 平台。
  - Baseline 核心缺陷：
    1. **全局统一权重导致 LoRA 特性稀释**：所有层使用相同的组合权重 w_i，但不同层的 LoRA 参数编码了不同特征（如 Observation 2 所示：NLP 中 LoRA 的 0%-20% 层擅长 QNLI，80%-100% 层擅长 ANLI-R1）。当 N≥3 时，归一化将每个 LoRA 的 w_i 压缩到 1/N，导致关键层中的区分性特征被平均噪声淹没。
    2. **组合灵活性差**：一旦确定 {w_i}，无法在不重新计算所有权重的情况下增删 LoRA。若要排除某个 LoRA，需重新归一化剩余权重。
    3. **缺乏数据驱动的适应性**：NLA 的权重 w_i 由人工指定或启发式搜索（如 LoRAHub 的 gradient-free optimization），无法根据具体下游数据自适应调整。
    4. **直接算术组合（Eq.1 w/o 归一化）在 N 增大时破坏生成能力**：如 Fig. 3 I 所示，直接叠加 3 个以上 LoRA 会导致生成图像无意义输出；NLP 中 FLAN-T5 组合 4+ LoRA 输出混乱。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MOLE 通过 **hierarchical weight control（逐层可学习 gating function）+ gating balancing loss + 双推理模式** 解决上述缺陷。以 DreamBooth + Stable Diffusion V2.1 三概念生成（Dog + Cat + Sunglasses）为例的 MOLE 全栈执行路径：
  - **算法层（MOLE 逐层组合）**：
    1. 预训练 block 前向：$F_\theta(x) = \text{Attn}(\text{LN}(x|\theta)) + \text{FFN}(\text{LN}(\cdot|\theta))$（Eq.5-6）
    2. 每个 LoRA expert i 的独立前向：$E_{\Delta\theta_i}(x) = \text{Attn}(\text{LN}(x|\Delta\theta_i)) + \text{FFN}(\text{LN}(\cdot|\Delta\theta_i))$（Eq.7-8）—— 为每个 LoRA 单独计算 full block 输出（float32 精度，N 路并行或串行）。
    3. Gating 函数逐层计算权重：concat 所有 $E_{\Delta\theta_i}(x)$ → Normalization → flatten → dot-product 映射到 N 维（Eq.9-10）→ softmax 归一化（Eq.11, learnable temperature τ）→ gating values $\mathcal{G}_i$。
    4. 加权组合：$\tilde{E}_\Omega(x) = \sum_i \mathcal{G}_i \cdot E_{\Delta\theta_i}(x)$（Eq.12）。
    5. 残差融合：$O(x) = F_\theta(x) + \tilde{E}_\Omega(x)$（Eq.13）。
    6. **训练阶段**：仅优化 gating function 参数（e 和 τ），冻结所有 LoRA 和预训练模型权重。V&L 域使用 CLIP local+global guidance 作为无监督训练信号（L_CLIP），NLP 域使用 FLAN-T5 的 cross-entropy。同时施加 gating balancing loss $\mathcal{L}_{\text{balance}} = -\log(\prod_i q^{(i)})$（Eq.14-15）防止 gating 坍塌。
  - **系统框架层**：论文未明确说明。实现层面需在 PyTorch/PEFT 基础上，为每个 Transformer block 注入 gating 模块，并支持逐 LoRA 独立前向计算（memory 开销 = N × 单 LoRA 前向）。推理时两种模式：(1) 全专家模式——使用所有 LoRA + 已学习 gating weights；(2) mask 模式——手动排除某些 LoRA 后，gating 重新按比例分配剩余权重，无需重训练。
  - **编译框架层**：论文未明确说明（标准 PyTorch）。
  - **kernel 调度层**：论文未明确说明。每个 LoRA 的 $E_{\Delta\theta_i}(x)$ 计算可并行（batch N），但论文未讨论 kernel fusion 或内存优化。
  - **硬件架构层**：论文未明确说明 GPU 平台。
  - 对比 baseline 的改进映射：
    - **全局统一权重 → 逐层 learnable gating**：NLA 所有层共享 {w_i}（1 组 N 维标量）→ MOLE 每层独立学习 gating 分布（M 组 N 维 softmax 输出）。对应 Observation 2：不同层编码不同特征 → 不同层应有不同组合权重。Table 9 的 coarse-to-fine 分析验证了 layer-wise/block-wise gating（1-MoLE/b-MoLE）优于 network-wise（n-MoLE），证明逐层控制的必要。NLP 域中，gating 可视化（Fig. 7）显示 0%-20% 层对 LoRA A 的权重达 45%，80%-100% 层对 LoRA C 的权重达 52%——自动复现了 Observation 2 的层特异性规律。
    - **特征稀释 → gating 动态"增强/抑制"**：NLA 将每个 LoRA 的贡献强制均分（w_i ≈ 1/N）→ MOLE 的 gating 对期望特征赋予高权重（如 Dog LoRA 在"耳朵/鼻子"相关层权重 0.45）、对不期望特征赋予低权重（如 Dog LoRA 在"背景/风格"层权重 0.05）。V&L 域 Text-alignment 从 NLA 的 0.678 提升到 MOLE 的 0.759（+0.081），Image-alignment 从 0.694 提升到 0.757（+0.063，Table 1 平均）。
    - **组合灵活性差 → 双推理模式**：NLA 增删 LoRA 需重新计算所有权重 → MOLE 推理模式 2 通过 mask 排除特定 LoRA 后，gating 自动按比例重新分配权重（无需重训练）。Fig. 8 验证了从 3-LoRA MOLE → 2-LoRA MOLE 的平滑降级能力。
    - **无数据驱动适应性 → 下游数据微调 gating**：NLA 权重人工指定 → MOLE 用 domain-specific loss（V&L: CLIP guidance, NLP: task cross-entropy）微调 gating 参数。NLP 域泛化实验（Table 8）：NLI 任务训练的 gating → BBH 评估，MOLE 仍然优于 LoRAHub（+2.4），证明 gating 学习到的是结构性的组合策略而非过拟合到特定任务。
    - **Gating 坍塌 → gating balancing loss**：无约束 gating 会收敛到仅激活 1 个 LoRA（Fig. 5b: w/o L_balance 时 LoRA β 权重大 68%）→ L_balance 鼓励均匀分布，保持多 LoRA 利用。Table 7 消融：MOLE w/o L_balance（77.57）< MOLE（78.07），且仅调大 τ（温度上升）导致性能更差（76.35-77.45），因为高温使 softmax 过平坦、丧失区分能力。
    - **大规模 LoRA 组合**：NLP 域 128 LoRA 时 MOLE（38.5）远优于 LoRAHub（35.5），因 LoRAHub 的 gradient-free 优化常将多数 LoRA weight 置零，而 MOLE 的 gating balancing loss 保持较均匀分布。
