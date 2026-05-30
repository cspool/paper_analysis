## Dual Form (TTT)（TTT 对偶形式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual form（对偶形式）是 TTT 层中将内循环梯度更新计算转化为矩阵乘法（matmul）操作的数学重写技术。问题背景：primal form（原始形式）需要显式计算每个 token 的外积梯度 G_t = ∇ℓ(W; x_t) = 2(W x̂_t - y_t) x̂_t^T，得到 d×d 矩阵后逐 token 更新 W。这导致两个效率问题：(1) 外积操作无法充分利用 GPU TensorCores（TensorCores 专门优化 matmul，而非外积）；(2) 每个 d×d 矩阵 G_t 的 I/O 开销远大于 d 维向量 x_t。Dual form 的解决方案：通过数学恒等式将 W_b 和 Z = [z_1,...,z_b] 表达为纯 matmul 操作，避免显式存储 G_t。具体地，W_b = W_0 - 2η(W_0X̂ - Y)X̂^T（一次 matmul），Z = W_0X̄ - 2η(W_0X̂ - Y)mask(X̂^TX̄)（两次 matmul + 上三角 mask）。Dual form 在 TPU 上比 primal form 快 5× 以上。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dual form 在第一个 TTT mini-batch 中的 kernel 执行伪代码：

```
# ===== 输入 =====
# X = [x_1,...,x_b] ∈ R^{d×b}     # mini-batch of b tokens
# W_0 ∈ R^{d×d}                    # initial weight (from previous mini-batch)
# η                                # learning rate
# θ_K, θ_V, θ_Q ∈ R^{d×d}         # projection matrices

# ===== Dual Form Kernel =====
# Step 1: 投影（三个独立 matmul，利用 TensorCore）
X̂ = matmul(θ_K, X)     # training view, R^{d'×b}
Y  = matmul(θ_V, X)     # label view, R^{d'×b}
X̄ = matmul(θ_Q, X)     # test view, R^{d'×b}

# Step 2: 计算 mini-batch 结束时的权重（matmul）
# W_b = W_0 - 2η Σ_t (W_0 x̂_t - y_t) x̂_t^T
#      = W_0 - 2η (W_0 @ X̂ - Y) @ X̂^T
E = matmul(W_0, X̂) - Y           # error matrix, R^{d'×b}
W_b = W_0 - 2η * matmul(E, X̂^T)  # weight update, R^{d×d}

# Step 3: 计算所有输出 token（matmul + mask）
# 引用 Fact 1: V · mask(A^T Q) = [Σ_{s=1}^t a_s^T q_t · v_s]_t
# 设置 A=X̂, Q=X̄, V=E, 得到中间量 Δ
S = matmul(X̂^T, X̄)               # similarity matrix, R^{b×b}
S_masked = upper_triangular_mask(S)  # 保留上三角，下三角置零
Δ = matmul(E, S_masked)           # correction term, R^{d'×b}

# Step 4: 最终输出
Z = matmul(W_0, X̄) - 2η * Δ      # output tokens, R^{d'×b}

# ===== 输出 =====
# W_b: 更新后的权重（传递给下一个 mini-batch）
# Z: 当前 mini-batch 的输出 tokens
```

关键观察：
- 所有核心操作均为 matmul（TensorCore 友好）
- 没有显式的外积操作（G_t = 外积）
- S = X̂^T X̄ 的计算是 O(b² × d')，但由于 b=16 很小，实际开销极低
- mask 操作是 element-wise，开销可忽略

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实际 TTT 实现中：
- **Forward (prefill) 模式**使用 dual form：需要并行处理整个 prompt 的 tokens，dual form 将全操作转为 matmul，最大化吞吐量
- **Generate (decode) 模式**使用 primal form：每次仅生成一个 token，无需批处理的 dual form
- Dual form 在 JAX 实现中通过 XLA 自动融合 matmul + mask 操作
- 论文指出 dual form 对 TTT-MLP 同样适用（附录 A），只是符号更复杂（需要处理多层非线性激活），核心思想不变——通过标准反向传播计算 Σ_t G_t^k，再通过 vjp 计算输出

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
