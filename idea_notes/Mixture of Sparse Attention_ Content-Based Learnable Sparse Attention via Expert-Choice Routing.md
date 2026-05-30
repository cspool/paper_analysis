## Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

- baseline方法是什么？
  Baseline 有三类：(1) **Dense self-attention**；(2) **Fixed Sparse Attention**（位置固定稀疏，stride-based）；(3) **Routing Transformer**（online K-means 聚类）。

  全栈执行例子（以 Dense baseline, Small model 113M, T=1024, 单 A100）：

  - **算法层（Dense baseline）**：标准 multi-head self-attention，9 heads，每 head 计算完整 Q=XW^Q, K=XW^K, V=XW^V（各 ∈ R^{1024×64}），然后 A = softmax(QK^T/√64 + M) @ V（QK^T ∈ R^{1024×1024}），FLOPs = 9×(8×1024×64×1024 + 4×64×1024²) ≈ 9×(0.537×10^9 + 0.268×10^9) = 7.25 GFLOPs/层。KV-cache: T × H = 1024 × 9 = 9.2K key-value pairs。所有 token 都参与计算，无论其重要性如何。

  - **算法层（Fixed Sparse baseline, ρ=32, k=32）**：每个 head 固定选择位置 [0, 32, 64, ..., 992] 的 32 个 token，计算 Q/K/V 投影和 attention。关键缺陷：(a) 稀疏模式与内容无关，无法根据当前输入动态调整关注点；(b) 预选 token 必须在早期层聚合周围信息，在后续层再将信息路由回原始位置——这一信息路由开销限制了模型的表达能力；(c) 所有 head 使用完全相同的 token 选择，缺乏 head 间专业化。

  - **算法层（Routing Transformer baseline）**：每 head 用 online K-means 将 tokens 聚为 ρ 个簇（各 k 个 token），基于 dot-product 距离将 token 分配给最近簇中心。簇中心通过移动平均更新。关键缺陷：(a) online K-means 收敛极慢 [Bottou & Bengio, 1994]，即使在数十万步训练后簇分配仍不稳定；(b) 必须计算所有 T 个 token 的 Q 和 K 投影才能聚类（2hh'T = 2×1024×64×1024 ≈ 0.134 GFLOPs overhead per head），且每一 Routing head 的 FLOPs ≈ ρ 个 MoSA head；(c) 每 head 内所有 ρ 个簇共享同一套线性变换 W^Q/W^K/W^V/W^O；(d) 为让 source 和 destination 选同样的 token，必须设 W^Q=W^K，限制灵活性。

  - **kernel调度层**：PyTorch native einsum/scatter/gather 操作，无专用 CUDA kernel。FlashAttention 可用于 dense head 但 MoSA 的 sparse attention 未被 FlashAttention 定制优化。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Dense attention O(T²) 的 FLOPs 和 KV-cache 随序列长度平方增长，造成训练和推理的巨大开销
  2. Fixed sparse attention 无法内容感知——固定稀疏模式对某些任务（如需要精确 retrieval）无效，且所有 head 共用同一选择 = 无 head 专业化
  3. Routing Transformer 的 online K-means 收敛慢、投影开销 T 级别（无法减至仅 k 个 token）、权重共享限制 expressiveness
  4. 现有稀疏方法在 IsoFLOP 比较下无法超越 dense baseline（论文实验证实 Fixed 和 Routing 均表现更差）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoSA 的核心理念：将 Expert-Choice Routing 的思想应用于 attention 机制——让每个 attention head 作为一个"专家"，从输入序列中学习选择自己需要处理的 k 个 token。这使得稀疏模式是**可学习、内容感知、head 专属**的。

  MoSA 全栈执行例子（以 Small model, ρ=32, k=32, hybrid: 4 dense heads + 381 MoSA heads, T=1024, 单 A100）：

  - **算法层**：
    1. 输入 X ∈ R^{1024×1024}，对每个 MoSA head i：
    2. Router 计算 r = σ(X @ W^r_i) ∈ R^1024（sigmoid 非竞争激活，遵照 σ-MoE 的发现）
    3. TopK(r, k=32) → r_topk ∈ R^32, 索引 I ∈ {0..1023}^32
    4. X^s = gather(X, I) ∈ R^{32×1024}——**仅对被选的 32 个 token 执行后续计算**
    5. Q/K/V = X^s @ W^Q/K/V_i ∈ R^{32×64}
    6. Causal mask M_{a,b}=0 if I_a≥I_b else -∞（保持了自回归约束）
    7. A = softmax(QK^T/√64 + M) @ V ∈ R^{32×64}
    8. X^o = diag(r_topk) @ A @ W^O_i——router score 乘到输出上，使路由梯度可反向传播
    9. Y = scatter(X^o, I) ∈ R^{1024×1024}——放回原位置，未选中位置填 0
    10. 最终输出 = Σ dense heads + Σ MoSA heads

    FLOPs: 4 dense heads × 0.805 GFLOPs + 381 MoSA heads × (8×1024×64×32 + 4×64×32² + 2×1024×1024 + 64×32) ≈ 3.22 + 381 × (16.78M + 0.26M + 2.10M + 0.002M) ≈ 3.22 + 381 × 19.14M ≈ 3.22 + 7.29 = 10.51 GFLOPs/层（vs dense baseline 7.25 GFLOPs/层的 9 dense heads, FLOP-matched）。

  - **kernel调度层**：纯 PyTorch einsum/scatter/gather 实现（无专用 CUDA kernel）。论文指出可结合 FlashAttention 加速 dense head 中的标准 attention 计算，并可开发专用 CUDA kernel 进一步加速 MoSA 的 sparse attention。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  MoSA 对 baseline 缺陷的解决（设计-缺陷映射）：

  1. **O(T²)→O(k²+T)**：MoSA 将每 head 的 Q/K/V/O 投影从 T 个 token 减至 k 个，attention 从 T×T 减至 k×k。例如 T=1024, k=32 时，投影成本降至 3.1%，attention 成本降至 0.1%。节省的 FLOPs 用于增加 head 数（从 9→385），实现更细粒度的专业化。

  2. **内容感知选 token → 解决 Fixed sparse 的内容无关性**：Router W^r 通过语言模型目标（cross-entropy loss）的梯度联合训练，直接学习哪些 token 对当前 head 最重要。这避免了 Fixed sparse 的固定 stride 选择，可以动态跳转到任意位置的关键 token。

  3. **Expert-Choice 完美负载均衡 → 解决 MoE routing collapse**：每个 head（专家）独立选择自己的 top-k token，天然保证每个 head 处理恰好 k 个 token。无需 auxiliary load-balancing loss，避免 token-choice routing 中的 expert collapse 问题。

  4. **仅对 k 个 token 做投影 → 解决 Routing Transformer 的 T 级投影开销**：Routing Transformer 在聚类前必须计算所有 token 的 Q/K，而 MoSA 先 router 选 token 再做投影，使投影成本正比于 k 而非 T。这使 MoSA 的 FLOP 成本约等于 Fixed sparse（内容感知却无额外计算开销）。

  5. **每个 head 独立权重 → 解决 Routing Transformer 的权重共享**：每个 MoSA head 有自己的 W^Q/W^K/W^V/W^O/W^r，head 间无共享，允许不同 head 专注于不同类型的 token 模式。384 个 head 各自学习独特的稀疏模式——这在 Routing Transformer 中不可行（因为单 head 内 ρ 个簇共享权重）。

  6. **混合架构（4 dense + M MoSA heads）→ 解决纯 MoSA 训练不稳定**：Router 和 attention weights 需联合学习，初期 router 随机选择导致 attention 学不到有用模式 → 恶性循环。4 个 dense head 提供稳定的全局信息流，稳定训练。实验证明 0 dense head 时性能崩溃（perplexity 从 22.46 升至 29.76 at ρ=16），4 dense head 时 -27% perplexity。

  7. **Perplexity-matched 资源节省**：即使无专用 CUDA kernel（仅 PyTorch），MoSA 在匹配相同 perplexity 时同步减少 wall-clock time（-2.1%~-12.9%）、GPU memory（-1.6%~-10.0%）和 KV-cache（-51.1%~-69.5%）。
