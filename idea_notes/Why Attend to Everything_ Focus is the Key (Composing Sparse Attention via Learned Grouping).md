## Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

- baseline方法是什么？
  Baseline 是标准 full attention（O(n²) 全对全注意力计算），以及三类高效注意力方法：(1) **结构化稀疏**（Longformer, BigBird）：使用固定位置模式（局部窗口+block/global pattern），内容无关，retrofit 到预训练模型时丢失长程依赖；(2) **近似方法**（Performer, Linformer）：用 kernel 近似或低秩投影替代 softmax 注意力矩阵，近似误差逐层累积，retrofit 时 PPL 退化惨重（Performer +75.6 PPL）；(3) **Token 选择方法**（SparQ, MagicPIG）：选择 top-k 最相关 token 但 PPL 退化 5-10 点。三者的共同缺陷是**不能学习哪些 token pair 真正需要互相关注**——固定模式缺乏内容感知，近似方法损失信息，token 选择方法缺少全局分组结构。

  全栈执行例子（GPT-2 124M 推理，full attention + FlashAttention on H100-80GB）：
  **算法pipeline**：对于长度为 T 的序列，每层计算 Q,K,V ∈ R^{T×d}，通过 softmax(QK^T/√d) 计算全部 T² 个 token pair 的注意力分数，复杂度 O(T²)。每个 token 的 softmax 概率分布在全部 T 个 token 上，导致：(a) softmax 稀释——一个代词要与其先行词竞争注意力权重，必须与数百个无关 token 共享概率质量；(b) 噪声累积——无关 KV pair 对注意力输出贡献微小噪声，12 层 × 12 heads 累积后显著降质。
  **系统框架**：HuggingFace Transformers / PyTorch 加载 GPT-2 124M 权重，使用 FlashAttention fused kernel 完成 attention 计算。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention 做 tiled QK^T + online softmax + V 加权，O(T²d) 计算 + O(T²) 中间结果，在 H100-80GB 上 T=1M 时约 1.5s。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出 Focus —— 添加少量 learnable centroid 向量（K 个 centroid，dg=16，仅 148K 参数）到每层注意力中，centroid 决定哪些 token pair 可以互相关注（routing），原 QKV 注意力决定关注多少（content）。核心设计：(1) **可学习 centroid + Sinkhorn 归一化**：通过投影 W_g 将 token 映射到 centroid 空间，Sinkhorn 迭代强制双随机均衡分组，阻止 group dominance（类似 MoE 中的 expert collapse）；(2) **门控注意力**：s_ij = q_i^T k_j · (1_local + (1-1_local)·σ(λ·g_i^T g_j))，局部窗口内全注意力，远距离仅同组 pair 保留；(3) **分离 routing 与 attention**：centroid 仅控制谁关注谁，内容流经预训练 QKV 不变——这是 composability 的关键，原始权重完全冻结；(4) **推理时 FlashAttention 分解**：将稀疏 mask 分解为两个不相交 FA 调用（same-group causal + cross-group local），logsumexp 精确合并，8.6× 加速无自定义 kernel。

  全栈执行例子（Focus on GPT-2 124M，推理 on H100-80GB）：
  **算法pipeline**：对长度为 T 的序列，每层首先计算 g = sinkhorn(W_g·h^T · C / τ, N=10) 得到每个 token 的 group assignment。局部窗口 w=128 内 token 全注意力；远距离 token 仅当 g_i^T g_j ≈ 1（同组）时参与 softmax。结果是 softmax 概率质量集中在较小但更相关的 token 子集上：(a) 消除 softmax 稀释——同组内竞争 token 更少且语义相关；(b) 消除噪声——无关跨组 pair 不参与注意力计算（而非被缩放到近零值）。效果：124M 上 PPL 30.3 vs full attention 31.4（稀疏超越密集），所有 benchmark 零退化。
  **系统框架**：HuggingFace Transformers + PyTorch，加载 GPT-2 权重、添加 centroid 参数（148K），仅 centroid 训练（4000 steps on PG-19），原权重冻结。
  **编译框架**：论文未明确说明。
  **kernel调度**：推理时 token 按 group 做 stable sort，reshape 为 K 个独立序列，对每个调用 flash_attn_func(causal=True)；同时计算 cross-group local 窗口注意力；两个输出通过 logsumexp merge 精确合并。A ∩ B = ∅（无重复计数），A ∪ B = 全部应关注的 pair。O(T²/K) + O(Tw)，T=1M, K=8 时 8.6× 加速。Sort overhead ~12ms 常数，长序列下可忽略。320 行 Python，无自定义 CUDA kernel。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。
