## TTT (Test-Time Training) Layers（测试时训练层）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TTT（Test-Time Training）层是一种将 RNN 的隐藏状态定义为一个机器学习模型 f 的权重 W，将更新规则定义为对自监督损失 ℓ 的一步梯度下降的序列建模层。核心理念：自监督学习能将大规模训练集压缩进模型权重（如 LLM 将互联网知识压缩进参数），因此将同样的"压缩启发式"用于 RNN 隐藏状态的更新。具体地，对于输入序列 x_1,...,x_T，隐藏状态为 W_t（模型 f 的权重），更新规则为 W_t = W_{t-1} - η ∇ℓ(W_{t-1}; x_t)，输出规则为 z_t = f(θ_Q x_t; W_t)。由于即使在测试序列上也会执行此训练过程，因此称为"测试时训练"（Test-Time Training）层。TTT 层具有线性复杂度 O(T × d²)，与序列长度 T 无关的每 token 开销。与 self-attention（KV cache 线性增长，O(T²) 总复杂度）相比，在长上下文下具有渐近优势；与传统 RNN（Mamba）相比，通过梯度更新的自适应性避免了固定大小隐藏状态的表达瓶颈。TTT 层可即插即用替换 Transformer 中的 self-attention，或集成到 Mamba backbone 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TTT 层在语言模型中的 pipeline 流程（以 TTT-Linear 为例）：

```
# ===== 外循环（常规训练）=====
# 优化目标：next-token prediction loss
# 可训练参数：θ_rest（网络其余参数）, θ_K, θ_V, θ_Q, θ_init, θ_lr

# ===== 内循环（TTT，每个序列执行）=====
# 输入：序列 x_1,...,x_T ∈ R^d
# 超参：mini-batch size b, base learning rate η_base

W = θ_init  # 初始权重，形状 d×d（TTT-Linear）或 MLP 参数（TTT-MLP）

for each mini-batch of b tokens:
    X_block = [x_{t},...,x_{t+b-1}]  # ∈ R^{d×b}

    # Step 1: 多视角投影（learned views）
    X̂ = θ_K @ X_block   # training view ∈ R^{d'×b}
    Y  = θ_V @ X_block   # label view ∈ R^{d'×b}
    X̄ = θ_Q @ X_block   # test view ∈ R^{d'×b}

    # Step 2: 自监督损失（multi-view reconstruction）
    # ℓ(W; x_i) = ||f(x̂_i; W) - y_i||²

    # Step 3: 梯度下降更新（mini-batch GD）
    # W_new = W - η Σ_i ∇ℓ(W; x_i)
    # 通过 dual form 高效计算（避免显式外积）

    # Step 4: 输出 token
    # z_i = f(x̄_i; W_i)，其中 W_i 是处理 x_i 时的权重
    Z_block = compute_outputs(X̄, X̂, Y, W, η)

    W = W_new  # 隐藏状态传递到下一个 mini-batch

# 输出：z_1,...,z_T
```

关键设计特点：
1. **隐藏状态本身是模型**：W 的维度为 d×d（TTT-Linear），参数量远大于传统 RNN 的隐藏向量（d 维），因此具有更强的表达能力。
2. **更新规则是梯度下降**：与手工设计的门控机制（LSTM）或选择机制（Mamba）不同，梯度下降使更新天然具有数据自适应性——产生大梯度的 token 被"记住"更多。
3. **自监督任务可学习**：θ_K, θ_V, θ_Q 通过外循环学习，使内循环的 reconstruction 任务专门为外循环的 next-token prediction 目标优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TTT 层的两个主要实例化：
- **TTT-Linear**：f(x) = Wx，W ∈ R^{d×d}（方形矩阵）。与 linear attention 有理论等价性（Theorem 1：batch GD + W_0=0 + η=1/2 时等价）。实现最简单，计算效率最高。
- **TTT-MLP**：f(x) = x + LN(MLP(x))，MLP 为两层（hidden dim 4×, GELU 激活, LN + 残差连接）。表达能力更强，在长上下文下优势更大，但 wall-clock 开销更高。

实际使用中：
- 训练：基于 EasyLM (JAX) 框架，遵循 Chinchilla recipe（与 Transformer 相同的训练配置）。外循环使用 AdamW 优化器，内循环使用 SGD（mini-batch GD）。
- 推理：forward (prefill) 使用 dual form（matmul 并行），decode 使用 primal form（逐 token 序列化）。
- 代码开源：JAX 版本 https://github.com/test-time-training/ttt-lm-jax，PyTorch 版本 https://github.com/test-time-training/ttt-lm-pytorch。

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
