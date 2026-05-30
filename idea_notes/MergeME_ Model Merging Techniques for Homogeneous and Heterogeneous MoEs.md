## MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

- baseline方法是什么？
  Baseline 为 **BTX (Branch-Train-Mix)** (Sukhbaatar et al. 2024)，它将多个从同一 ancestor model 分支出来的 dense expert 合并为 MoE 模型。全栈执行路径（以 4 个 1B expert 合并为例）：
  - **算法层**：将各 expert 的非 FFN 层（embedding、attention、norm、head）通过 **unweighted averaging** 逐参数相加取平均合并为一套共享参数；FFN 层保持独立作为 MoE expert；插入一个随机初始化的 MLP router 在 attention 和 FFN 之间做 token-level top-K routing。合并后 MoE 在全部数据源混合数据集上 fine-tune 约 40B tokens 来训练 router 并恢复因参数干扰导致的性能损失。
  - **系统框架层**：基于标准 PyTorch 分布式训练，fine-tuning 阶段因 MoE 的 expert parallelism 引入跨 GPU all-to-all 通信开销（论文引 BTX 原论文指出 "fine-tuning MoEs is expensive due to communication cost between GPUs"）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明（标准 PyTorch CUDA kernel）。
  - **硬件架构层**：论文未明确说明具体 GPU 型号和集群配置。
  - BTX 的核心缺陷：
    1. **参数干扰 (Parameter Interference)**：当 experts 的参数空间发散较大时（不同 ancestor、激进不同的训练数据），简单平均无法处理 sign conflict 和 magnitude disparity——大 magnitude 参数与小 magnitude 参数/符号冲突参数平均后输出接近零的小值，削弱 task vector 效果，导致合并后 MoE 性能下降，需要大量 fine-tuning 恢复。
    2. **Fine-tuning 成本高且不可行**：MoE fine-tuning 需要多 GPU（跨 expert 通信），且需要访问所有 expert 的训练数据。当 expert 训练数据不公开或计算资源受限时，fine-tuning 无法执行。
    3. **无法处理异构专家**：BTX 要求所有 expert 具有相同架构（相同层数、hidden dimension），无法合并如 CodeLlama + Olmo 等不同架构的 expert 模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MergeME 通过三类技术分层解决上述缺陷。全栈执行路径（以 4-expert MoE, Base-1B + Math + Code + Knowledge 为例）：
  - **算法层 — 同构合并：Dare/Ties 替代平均**：
    1. 计算 task vector τᵢ = θ_b - θᵢ（base 与 expert 参数差）。
    2. Ties merging: drop bottom (100-p)% 最小 magnitude 参数 → 确定每个位置主导符号方向 → 仅累加同符号 task vector 值，消去异符号冲突。
    3. Dare merging: 随机 drop (100-p)% 参数 → rescale τᵢ/(0.01p) 补偿丢弃 → 简单求和所有 τᵢ。
    4. 合并回 base: θ_m = θ_b + λ·τ_m（λ=1/3, p=80%）。
    5. 保留 FFN 独立 + 插入 MLP router + fine-tune。
  - **算法层 — 无 Fine-tuning：PPL 路由 + 分离 Attention**：
    1. 不合并 attention 层：各 expert 保留自己的 attention 参数，避免合并后 attention 受 l 个 task vector 影响而 FFN 仅受 K 个 task vector 影响的不一致性。
    2. PPL 路由：对推理输入 x_inf 在每个 expert 上计算 PPL(x_inf|θᵢ) = exp(−1/t·Σ log P(xⱼ|x_{<j},θᵢ))，取 1/PPL 为 confidence，SoftMax(top-K(confidence)) 作为路由权重。
    3. 仅需一次额外 forward pass（远少于 inference 时 generate 多 token 的 forward pass 次数），无 fine-tuning 开销。
  - **算法层 — 异构合并：Projector + Sequence-level Router**：
    1. 共享 embedding/head 层：各 expert 的 embedding/head 参数 padding 零对齐到最大维度 d_m 后取平均。
    2. Proj-inᵢ: R^{d_m}→R^{dᵢ} 和 Proj-outᵢ: R^{dᵢ}→R^{d_m}（随机初始化 MLP），为每个异构 expert 提供维度适配。
    3. Sequence-level routing：因异构 expert 的 attention 层不能合并，所有 token 必须路由到同一 expert。将全部 token embedding 平均 → avg_e = 1/t·Σ eⱼ → router θ_r·avg_e → SoftMax(top-K) 做序列级路由。
    4. Fine-tune 所有参数（含 projector + router）。
  - **系统框架层**：基于标准 PyTorch 实现，与 BTX 相同。同构合并 fine-tuning 阶段仍需多 GPU MoE 并行；无 fine-tuning 模式避免了分布式训练开销；异构合并因不合并 attention 导致总参数略多于 BTX（~4B vs ~3.7B），fine-tuning 成本相应增加。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明具体 GPU 型号。
  - 对比 baseline 的改进映射：
    - **参数干扰 → Dare/Ties 替代平均**：Ties 通过符号剪枝 + 主导方向选择消除 sign conflict，Dare 通过随机 drop + rescale 避免大 magnitude 被小值稀释。Table 1 显示 Dare merging 平均 12.86 vs BTX 11.72（+9.72% 相对提升），Ties merging 平均 12.52（+6.94%）。Figure 10 显示 Dare/Ties 在 fine-tuning 早期阶段优势更明显（早期 token 数少时性能差距大），随着 fine-tuning 进行差距缩小但始终优于 BTX。路由分析（Figure 5）表明 Dare/Ties 合并的 MoE 更准确地将 token 路由到领域专家（如 GSM8K 上 Math Expert 路由概率从 BTX 的 0.28 升至 Dare 的 0.46）。
    - **Fine-tuning 成本高且不可行 → PPL 路由 + 分离 Attention 实现无 fine-tuning MoE**：PPL 路由仅需一次额外 forward pass（overhead 极低，因 inference 时 forward pass 数 = generate token 数 >> 1），Table 2 显示 PPL 路由能有效将各 benchmark 的输入导向对应专家（GSM8K: Math 43%, Knowledge 32%；HumanEval: Code 43%, Math 45%）。分离 attention 解决了合并 attention 和 FFN 的 task vector 数量不一致问题，Table 3 显示 separate attention + PPL routing 平均 8.08 vs merge attention + PPL routing 平均 7.32（+10.4% 相对提升），且优于 SoTA dense merging（Dare Dense 7.11）。
    - **无法处理异构专家 → Projector + Sequence-level Router 实现异构合并**：Proj-in/Proj-out 提供维度桥接（类似 Roberts et al. 2024 的 dense 模型异构合并），sequence-level router 因异构架构 attention 不能合并而采用序列级路由。Table 4 显示 MoE w/ Math TinyLlama 平均 13.34 vs 3-expert MoE (same data) 10.54（+26.6%），MoE w/ Math Olmo 平均 11.17 vs 3-expert MoE 10.54（+6.0%），证明了异构合并的有效性。局限性：异构合并因 embedding 层平均导致 router 在 math benchmark 上不一定将最高路由概率给 math expert（Figure 6），论文建议未来添加 load balancing loss 解决。
