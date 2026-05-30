## MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

- baseline方法是什么？
  - Baseline 为 LoRA（r=80, alpha=160, 应用于 q,k,v,o + w1,w2,w3）和 DoRA（同配置，weight-decomposed LoRA 变体），均为标准 PEFT 方法，在单个下游任务上独立微调。
  - 全栈执行路径（以 LoRA + LLaMA-2 7B 在 BoolQ 上单任务微调为例）：
    - **算法层**：LoRA 将权重更新分解为低秩矩阵 B·A（B∈R^{d1×r}, A∈R^{r×d2}, r=80），前向计算 W' = W + B·A。单任务微调时，LoRA adapter 仅学习该任务的知识。多任务场景下，同一套 LoRA 参数在混合任务数据上训练，缺乏对不同任务模式的显式分离机制——所有 token 经过相同的一组 LoRA adapter 计算，无任务/ token 级差异化处理。
    - **系统框架层**：HuggingFace Transformers + PEFT 库。单模型训练/推理，无 MoE 路由开销。
    - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。
    - **kernel 调度层**：标准 PyTorch linear kernel，无自定义 CUDA kernel。
    - **硬件架构层**：24GB consumer GPU（RTX 3090/4090/A5000），half precision。
  - Baseline 核心缺陷：
    1. **多任务学习中性能退化**：LoRA 单任务→多任务切换中 average accuracy 下降 4.4%（69.9%→65.5%，Table 2），DoRA 下降 8.0%（74.3%→66.3%）。原因是有限的 trainable parameters 在混合任务上缺乏对不同任务模式的显式分离，导致 catastrophic forgetting 和跨任务干扰。
    2. **模型容量受限**：单套 LoRA adapter 的容量受限于 rank r，无法像 MoE 模型那样通过增加 expert 数量来扩展模型容量。LoRA/DoRA 在 LLaMA-3 8B（78.2%/78.5%）和 LLaMA-2 13B（81.5%/81.9%）之间存在显著差距（3.3%/3.4%），表明 model capacity 是性能瓶颈。
    3. **无 token 级差异化计算**：所有输入 token 经过相同的 LoRA adapter，无法像 MoE 那样根据 token/task 特性选择不同的计算路径。不同任务（如推理任务 ARC vs 知识任务 OpenBookQA）受益于不同的参数特化，但标准 LoRA 无法提供。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MixLoRA 通过"FFN 层 LoRA 专家化 + Top-K 路由 + Attention 层 LoRA 解耦 + 负载均衡"四层设计解决上述缺陷。
  - 全栈执行路径（以 MixLoRA + LLaMA-2 7B 多任务学习为例，K=8 experts, top-2 router, r=16）：
    - **算法层 — MoE 化 FFN 构造**：
      1. 每个 expert = 共享冻结 FFN 权重（W1,W2,W3）+ 独立 LoRA adapter（A^k_W1,B^k_W1, A^k_W2,B^k_W2, A^k_W3,B^k_W3），而非传统 LoRA-MoE 方法中将整个 LoRA 模块作为 expert。这使 MixLoRA 更接近 Mixtral 等预训练 MoE 模型的架构。
      2. Router 为线性层 W_r·x → Softmax → KeepTop-2，为每个 token 选择最优 2 个 expert。
      3. Expert 输出由 router probability 加权求和：h = Σ R(x)_k · (W·x + B_k·A_k·x)。
      4. Self-attention 层使用独立的 LoRA adapter（q,k,v,o 投影），不参与 MoE 路由——因为 ST-MoE 研究表明微调 attention 层可显著提升 MoE 模型性能。
      5. Auxiliary load balance loss（a=1e-2）确保 8 个 expert 负载均衡（平均 std dev 0.0223）。
    - **算法层 — 性能优化**：
      - **共享计算**：先计算 W1·x 和 W3·x，再按路由权重切片给各 expert 的 LoRA 计算，避免每个 expert 重复计算 FFN 骨干。W2 无法共享因其依赖 W1/W3 输出。
      - **多模型高吞吐**：多个 MixLoRA 模型的输入合并为一个 batch，共享预训练权重，各模型独立路由。
    - **系统框架层**：基于 HuggingFace Transformers + PEFT 库实现。优化使用了 m-LoRA 风格的多 LoRA 并行技术。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明（标准 PyTorch CUDA kernel，优化通过减少冗余计算而非自定义 kernel 实现）。
    - **硬件架构层**：24GB consumer GPU（RTX 3090/4090/A5000），half precision。
  - 对比 baseline 的改进映射：
    - **多任务学习中性能退化 → MoE 路由实现任务/token 级计算路径分离**：不同 token 被 Top-2 Router 分配到不同 expert 组合，每个 expert 通过 LoRA adapter 学习到不同的参数特化。MixLoRA 多任务学习 accuracy 仅下降 -0.6%（74.7%→75.3% 反而提升），而 LoRA 下降 -4.4%、DoRA 下降 -8.0%。MoE 路由机制天然缓解了多任务间的 data conflict 和 catastrophic forgetting——不同任务的数据倾向于被路由到不同的 expert 子集（Figure 5 显示 8 个 expert 负载均衡，各任务间分布均匀）。
    - **模型容量受限 → MoE 结构以低成本扩展模型容量**：8 个 expert 的 MixLoRA（r=16, 2.9% trainable params）在 LLaMA-3 8B 上取得 83.5% avg accuracy，超过 LLaMA-2 13B 上 LoRA 的 81.5%。每个 expert 通过独立 LoRA 提供不同的参数特化，总体可学习参数量虽与 baseline LoRA（r=80）相近，但通过 MoE 路由实现了条件计算——不同 token 使用不同的参数子集，有效扩展了模型的表征能力。
    - **无 token 级差异化计算 → Top-2 Router 实现 token-wise dynamic routing**：Router 为每个 token 独立计算 expert assignment，不同 token 被分配给不同 expert 对（top-2 from 8），实现细粒度的计算路径差异化。Ablation 显示 rank=16 且 8 experts 的 MixLoRA 优于 rank=32 单体的变体——说明条件计算带来的收益超过了单体 rank 增加。
    - **LoRA 容量 vs MoE 效率的 Pareto 改进**：朴素 MixLoRA token 计算延迟 535.2 µs（LoRA 的 218%），但共享计算优化后降至 462.5 µs（188%），同时 accuracy 远超 LoRA。多模型模式下 per-model GPU memory 从 15.1GB 降至 8.8GB（训练），为 consumer GPU（24GB）上同时微调多个模型提供了可能。
    - **DoRA 兼容性（MixDoRA）**：将 expert 基础单元从 LoRA 替换为 DoRA，MixDoRA 在部分场景下（如 Gemma 2B 单任务 71.6%）优于 MixLoRA（69.9%），但在 LLaMA-2 7B 多任务中 MixDoRA（74.9%）与 MixLoRA（75.3%）性能相近，且 MixDoRA 对负载均衡 loss coefficient 更不敏感。
