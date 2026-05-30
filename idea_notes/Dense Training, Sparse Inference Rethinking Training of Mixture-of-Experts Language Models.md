## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 是 **Sparse MoE (SMoE) 训练 + Sparse Inference**：传统 MoE 训练中，每个 token 仅激活 top-K 个 expert（通常 K=2），反向传播时梯度仅通过这 K 个激活的 expert 和对应的 Router 分数传递。Baseline 使用 dMoE (MegaBlocks) 实现，Switch Transformer style 训练（Fedus et al. 2022），使用 switch loss 进行负载均衡。这种稀疏训练范式导致 MoE 模型的参数效率远低于 Dense 模型——需要 2-4× 更多参数才能匹配 Dense 模型性能。

  **Baseline 全栈执行例子（SMoE-5B, D_emb=3072, N_ffd=16, top-2 sparse training, 训练 step）**：
  - **算法层**: token X [3072] → Router S = Softmax(h(X)) → TopK=2 选择 expert 3, 7 → 仅 E_3(X), E_7(X) 计算前向 → O = S_3·E_3(X) + S_7·E_7(X) → 反向传播仅更新 Router 对应 expert 3,7 的行和 E_3, E_7 的参数 → expert 0,1,2,4,5,6,8-15 不获得梯度。Switch Loss 作为辅助损失 per layer: L_switch = α · N · Σ_i f_i · P_i，其中 f_i 是分给 expert i 的 token 比例，P_i 是 Router 给 expert i 的平均概率。
  - **系统框架层**: dMoE (Gale et al. 2023, MegaBlocks) 实现 MoE 训练（expert parallelism + data parallelism），FSDP (Zhao et al. 2023; Rajbhandari et al. 2020) 分片优化器状态和参数。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: MegaBlocks 的 block-sparse GEMM kernel 处理 grouped expert FFN 的批量前向/反向计算。稀疏训练中仅有 top-2 expert 被调度计算。
  - **硬件架构层**: NVIDIA H100 80GB × 32。训练 3B/6B-scale 需要 64h/124h。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 DS-MoE（Dense Training, Sparse Inference）：

  **(1) Sparse Training 的参数低效** → 解决：Dense Training（所有 expert 在训练时全部激活）：
  
  Baseline SMoE 的 Router 梯度在反向传播时被 binary mask M ∈ {0,1}^N 截断：∇S = [e_1(X),...,e_N(X)]^T ∇O ⊙ M，仅激活的 top-K expert 对 Router 梯度有贡献。DS-MoE 保留完整梯度：∇S = [e_1(X),...,e_N(X)]^T ∇O，所有 N 个 expert 的梯度均参与 Router 更新。每个 expert e_i 的梯度也变为 ∇e_i(X) = S_i · ∇O，获得按其 Router score S_i 加权的完整梯度。这意味着：(a) Router 能学习所有 expert 的能力分布（类似 Dense model 的 FFN 全参数学习）；(b) 所有 expert 参数均被持续优化（而非只有被激活的）；(c) 参数效率达到 Dense 模型水平。

  **(2) 训练后如何实现 Sparse Inference** → 解决：Mutual Information (MI) Loss：
  
  Dense Training 不能直接退化为 Sparse Inference——如果没有额外的正则化，Router 会均匀分配所有 expert（无 sparsity）。DS-MoE 引入 MI Loss：L_MI = -H(e) + (1/|X|)Σ H(e|X)，其中：
  - 最大化 H(e) = -Σ p(e_i) log p(e_i)：保证 expert 负载均衡（所有 expert 被充分训练）。
  - 最小化 H(e|X) = -Σ p(e|x) log p(e|x)：鼓励 Router 为每个 token 做出确定性选择（sparse concentration）。
  
  两者形成"对抗"平衡：负载均衡 vs 专家集中。训练后，Router 自然产生 sparsity——仅 top-K 或超阈值 expert 被激活（30-40% parameters during inference）。α 参数（MI loss weight）控制 sparsity 程度。

  **(3) Baseline SMoE 的 Attention 层仍是 dense** → 解决：Mixture of Attention Head (MoA)：
  
  多数 SMoE 模型仅在 FFN 层使用 expert（Attention 仍为 dense）。DS-MoE 将 Attention 也改为 MoA (Zhang et al. 2022)：每个 expert 计算 N_head 个 query vectors，共享 KV cache。MoA 在推理时也可 sparse，进一步减少计算。

  **论文方法全栈执行例子（DS-MoE-3B, D_emb=3072, N_ffd=32, D_ffd=384, N_att=8, 训练 step）**：
  - **算法层**: 
    1. Dense Training Forward: token X → Router S = Softmax(h(X)) → 计算所有 32 个 expert 的输出 E_i(X) = GeLU(X@W_up_i + b_up_i)@W_down_i + b_down_i → O = Σ_{i=1..32} S_i · E_i(X)。计算量 ≈ 32 × 2 × 3072 × 384 = 75.5M FLOPs/token/layer（≈ Dense-3B 的相当水平，因 Dense-3B 的 D_ffd=12288 而非 32×384=12288，实际 FLOPs 相同）。
    2. Dense Training Backward: ∇O → ∇S = [E_1(X),...,E_32(X)]^T ∇O（所有 32 个 expert 对 Router 梯度有贡献）→ ∇e_i(X) = S_i · ∇O → Router 和所有 expert 参数同步更新。Router 学习到所有 32 个 expert 的全景分布。
    3. MI Loss per batch: 统计 P(e_i) = mean over batch of S_i（expert 边际分布）→ H(e) = -Σ P(e_i) log P(e_i)；统计 P(e_i|x) = S_i（per-token expert 分布）→ H(e|x) = -Σ S_i log S_i → L_MI = -H(e) + mean(H(e|x))。
    4. Sparse Inference: Router 计算 S → TopK=6 → ParallelLinear dispatch X to selected experts → 仅 6 experts 执行 → O = Σ_{i∈top-6} S_i·E_i(X)。Active parameters: 6/32 = 18.75% of expert params, ~34% of total hidden params (accounting for attention and norms)。
  - **系统框架层**: PyTorch + FSDP (fully sharded data parallelism) + activation checkpointing。使用 Flash Attention (通过 PyTorch SDPA 或手动指定) 优化注意力的 HBM I/O。dMoE (MegaBlocks) 仅用于 SMoE baseline 实现。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: MLP 层使用 SimpleMoE 的 ParallelLinear（原生 PyTorch eager, group-based expert parallelism kernel）；Attention 层使用 torch.nn（dense inference, 因 sparsity >60% 时 sparse overhead 超过 dense）。训练使用 PyTorch 原生 eager execution + FSDP 分片通信。
  - **硬件架构层**: H100 80GB × 8 (1B) / × 32 (3B, 6B)。推理 evaluation on A100-80GB 和 H100-80GB。DS-MoE-3B 训练 64h on 32×H100；DS-MoE-6B 训练 124h on 32×H100。

  **核心设计洞察**：传统观点认为 MoE 的稀疏性来自训练阶段的稀疏激活（sparse gradient）。DS-MoE 发现稀疏性可以作为训练后的"退火"行为自然浮现——通过 dense training 保留参数效率，通过 MI loss 在训练过程中隐式塑造 sparsity pattern，最终在推理时仅激活 top-K expert。这从根本上不同于"稀疏训练后推理"的经典范式。
