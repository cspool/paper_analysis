## Load Balancing Mixture of Experts with Similarity Preserving Routers

- baseline方法是什么？
  - **Load Balancing Loss (LBL) [Fedus et al. 2022]**：MoE 训练中广泛使用的辅助负载均衡损失 L_LBL = α · E · Σ f_i · P_i，其中 f_i 为 expert i 被路由的 token 比例、P_i 为平均路由概率。LBL 通过鼓励接近均匀的 expert 分布来防止路由 collapse（所有 token 被路由到少数几个 expert）。该方法是当前 SOTA MoE 模型的标准组件（OLMoE, DeepSeek-V3, DBRX 等）。LBL 的缺陷：
    1. **知识冗余**：强制 uniform distribution 导致不同 expert 接触到相似的 token 集合，模型容量被浪费于在多个 expert 中学到冗余知识
    2. **路由不稳定**：训练早期 embedding 快速变化 + near-uniform 分配 → 微小输入扰动可导致 token 被重新分配给不同 expert → 进一步加剧 expert 间知识冗余
    3. **路由不一致**：相似 token 可能被路由到完全不同的 expert，使模型无法利用 token 间语义结构
    4. **需要超参调优**：损失系数 α 需要在主任务损失和负载均衡之间平衡，且对 batch size 敏感
  - 全栈执行例子（Baseline LBL，MoE-L 在 8× AMD MI300X 上训练一个 Transformer decoder layer 的前向传播）：
    - **算法 Pipeline 层**：Token 输入 x ∈ R^{B×S×1536} → Router R ∈ R^{1536×32} → softmax → top-4 选择 → SwiGLU expert FFN（D_E=1536）→ expert 输出加权和。LBL 作为辅助损失加入 L_total = L_lm + 0.01 · L_LBL，鼓励 32 个 expert 的 f_i · P_i 接近均匀。训练早期 LBL 可能主导梯度，导致路由决策不稳定。
    - **系统框架层**：PyTorch + OLMo 训练框架 + DDP（Distributed Data Parallelism）。数据来自 DCLM-pool-400m-1x，cl100k_base tokenizer。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **Kernel 调度层**：PyTorch bfloat16 GEMM 用于 Router (1536×32)、Expert FFN (1536×1536)。无需自定义 CUDA kernel。
    - **硬件架构层**：8× AMD MI300X 192GB per node 或 8× NVIDIA A100 40GB。训练 FLOPs ≈ 2.84×10^20 (MoE-L)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **SIMBAL 方法**：
    1. **正交 Router Loss**：L_orth = ||R^T R - I_E||_1，鼓励 Router 权重矩阵 R ∈ R^{D_M×E} 逼近正交矩阵（正交矩阵保留点积即角度，因此保留 token 间成对相似性）。相似 token 得到相似的 expert 分布 → 一致的 routing 行为 → 减少冗余。
    2. **Loss-based 而非显式参数化**：相比于通过 QR 分解强制正交（计算昂贵、频繁重正交化、数值不稳定），SIMBAL 使用辅助损失实现软约束，直接在 bfloat16 中训练，无需 float32 转换。
    3. **数据无关**：L_orth 仅依赖 Router 权重，不依赖数据分布或 batch size，消除 LBL 对 batch size 的敏感性。
    4. **正交初始化**：使用 Saxe et al. 2014 初始化使 Router 接近正交，加速收敛（或简单执行少量 router-only 训练步也可）。
    5. **PES 指标**：提出 Pairwise Expert Similarity (PES) = mean over token batches of pairwise expert output cosine similarity，作为轻量级 expert 冗余度度量。
  - 对应解决 Baseline 缺陷：
    - 缺陷1（知识冗余）→ 正交 Router 保留 token 相似性 → 相似 token 获得一致 routing → 各 expert 专精于处理特定类型的 token（PES 从 0.0241 降到 0.0028）
    - 缺陷2（路由不稳定）→ 正交 Router 对 input perturbation 更加鲁棒（角度保持 property）→ 训练早期不出现频繁 routing shift → 冗余增长率显著低于 LBL
    - 缺陷3（路由不一致）→ 相似 token 获得相似的 expert 分布 → Router 输出间保持 pairwise angle → 结构化 routing
    - 缺陷4（超参调优）→ SIMBAL 系数不敏感（0.01/0.1/1.0 下 perplexity 13.687/13.685/13.716），无需分布式同步
  - 全栈执行例子（SIMBAL，MoE-L 在 8× AMD MI300X 上训练）：
    - **算法 Pipeline 层**：与 baseline 相同的 MoE forward path，但训练 loss 替换为 L_total = L_lm + 0.1 · ||R^T R - I||_1。Router 从正交初始化开始，每步 optimizer step 后权重被 L_orth 的梯度拉向正交方向。Gram matrix R^T R 的 L2 distance 从 ~0.03 (LBL) 降至 ~2×10^-8 (SIMBAL)——即 Router 高度正交。Token 相似性通过 Router 保留：cos(x1,x2) ≈ cos(x1·R, x2·R)。
    - **系统框架层**：PyTorch + OLMo + DDP，与 baseline 相同。SIMBAL loss 计算仅涉及 Router 权重矩阵（1536×32），计算代价可忽略。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：与 baseline 相同——bfloat16 GEMM。SIMBAL 不引入额外 kernel，仅增加一个 O(E^2·D_M) 的 Gram matrix + L1 norm 计算。
    - **硬件架构层**：8× AMD MI300X 192GB 或 8× NVIDIA A100 40GB。训练吞吐量与 LBL 相当或更好（因 SIMBAL 更快收敛，36% 更少 token）。
  - **关键设计选择**：
    - 选择 orthogonal router 而非 orthogonal experts（OMoE, MOORE）——Router 参数极少（0.018% total params）但编排 billions of parameters，施加正交约束于此更有 leverage
    - 选择 L1 norm（而非 L2/Frobenius）作为 Gram matrix deviation measure——L1 在数值上更稳定
    - 推理时与 expert pruning 的良好协同：SIMBAL 产生 less uniform routing → 低 weight expert 更可安全丢弃 → 7.4% throughput improvement
