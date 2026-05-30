## Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- baseline方法是什么？
  - **Transformer (self-attention)**：隐藏状态为 KV cache（一个随 t 线性增长的列表），更新规则为 `K_t,V_t` 追加到列表，输出规则为 `z_t = V_t softmax(K_t^T q_t / √d)`。优势：显式存储所有历史上下文，长上下文表达能力强。缺陷：每个 token 的 cost 随 t 线性增长 O(t)，总复杂度 O(T²)，在长上下文时计算和内存开销巨大。
  - **Mamba (现代 RNN)**：隐藏状态为固定大小的 state space model 状态，更新规则为输入依赖的选择性 SSM 扫描。优势：线性复杂度 O(T)。缺陷：固定大小的隐藏状态表达能力有限，在超过 16k 上下文后无法有效利用额外 token 信息（perplexity 不再下降）。更新规则为手工设计的选择机制，缺乏灵活性。
  - Baseline 全栈执行（以 Transformer 推理一个 token 为例）：
    - **算法层**：x_t 经 θ_Q,θ_K,θ_V 投影 → q_t,k_t,v_t → 与历史所有 k_s 计算点积注意力分数 → softmax 归一化 → 加权求和 v_s → 输出 z_t。每个 token 需 O(t) 次内积。
    - **系统框架层**：vLLM 管理 KV cache（PagedAttention），调度 prefill 和 decode 阶段。
    - **编译框架层**：论文未明确说明（使用 JAX XLA 自动编译）。
    - **kernel调度层**：FlashAttention kernel 将 Q,K,V 矩阵分块加载到 SRAM，tiled matmul + online softmax 减少 HBM 访问。
    - **硬件架构层**：NVIDIA A100 TensorCores 执行 16×16 matmul，HBM 存储 KV cache。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **TTT 层**：将隐藏状态定义为一个机器学习模型 f 的权重 W_t，更新规则为对自监督 loss ℓ 的一步梯度下降 `W_t = W_{t-1} - η ∇ℓ(W_{t-1}; x_t)`，输出规则为 `z_t = f(θ_Q x_t; W_t)`。核心理念：**自监督学习能将大规模训练集压缩进模型权重**——这正是 LLM 的工作原理——因此也将这种"压缩启发式"用于 RNN 隐藏状态的更新。自监督任务（多视角重建）本身通过外循环学习，而非手工设计。
  - 两个实例化：
    - **TTT-Linear**：f(x) = Wx（线性模型），隐藏状态为 d×d 矩阵
    - **TTT-MLP**：f 为两层 MLP（hidden dim 4×，GELU，LN + 残差），表达能力更强
  - 解决 Baseline 缺陷的具体设计：
    1. **线性复杂度 + 强表达能力**：内循环梯度下降将任意长度的上下文压缩进固定大小的 W，复杂度 O(d²) 与 T 无关。相比于 Mamba 的固定状态更新规则，TTT 的梯度更新是数据自适应的——产生大梯度的 token 被"记住"更多。解决了 Mamba 在 16k 后无法利用长上下文的痛点。
    2. **可学习的自监督任务**：θ_K（training view）、θ_V（label view）、θ_Q（test view）通过外循环学习，使得内循环的 reconstruction 任务专门为最终的下一个 token 预测目标服务。解决了手工设计 reconstruction（如 denoising autoencoder）可能不是最优自监督任务的痛点。
    3. **mini-batch TTT (b=16)**：从 online GD (b=1, 序列化) 和 batch GD (b=T, 仅一步) 之间取折中，既保持了多步梯度下降的搜索空间（perplexity 接近 online GD），又利用 mini-batch 内的并行化（比 online GD 快得多）。解决了内循环梯度更新无法并行的痛点。
    4. **Dual form**：将 `W_b = W_0 - 2η(W_0X̂ - Y)X̂^T` 和 `Z = W_0X̄ - 2η(W_0X̂ - Y)mask(X̂^TX̄)` 全部表达为 matmul 操作，避免显式计算逐 token 梯度 G_t（外积），从而充分利用 GPU TensorCores 的 16×16 matmul 单元。TPU 上比 primal form 快 5× 以上。
    5. **与 linear attention 的理论等价与超越**：TTT-Linear + batch GD 等价于 linear attention（Theorem 1）。由此出发，mini-batch GD 贡献最大改进（PPL 12.35 vs 15.23），LN+residual in f 次之（PPL 14.05 vs 15.27）（Table 1）。这些设计在 attention 框架下难以自然产生。
  - TTT-Linear 全栈执行（以内循环一个 mini-batch 为例）：
    - **算法层**：x_1,...,x_b 经 θ_K,θ_V,θ_Q 投影 → 公式 (4) loss ℓ(W_0; x_t) = ||W_0 x̂_t - y_t||² → 外循环学习 θ_K,θ_V,θ_Q 使此自监督任务对最终语言建模最优 → 内循环梯度下降更新 W → 公式 (5) 输出 z_t = f(θ_Q x_t; W_t)。每个 token O(d²)，与 T 无关。
    - **系统框架层**：EasyLM (JAX) 训练框架，与 Transformer 相同的训练循环和 recipe（Chinchilla），TTT 层可即插即用替换 self-attention。Gradient checkpointing through time 节省内循环中间 W_t 的内存。
    - **编译框架层**：JAX XLA 自动编译。dual form 使所有关键操作为 matmul + element-wise，适配 XLA fusion。
    - **kernel调度层**：Forward (prefill) 使用 dual form kernel——所有操作为 matmul + mask，最大化 TensorCore 利用率；Decode 使用 primal form kernel——单 token 的梯度外积和权重更新。最终 TTT-Linear 1.3B 在 A100 上 prefill latency 略高于 Mamba 但远低于 Transformer（32k 时约 1/3）。
    - **硬件架构层**：与 Transformer 相同（A100 TensorCores + HBM），但 dual form 将全操作转为 matmul，卸载了 softmax 和 attention pattern 的非 matmul 开销。
