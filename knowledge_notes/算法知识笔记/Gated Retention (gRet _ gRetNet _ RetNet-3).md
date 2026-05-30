## Gated Retention (gRet / gRetNet / RetNet-3)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gated Retention（gRet，也称 gRetNet 或 RetNet-3）是 Retentive Network (RetNet) 的增强版本，由 YOCO 论文提出作为默认的高效自注意力模块。它在 Retention 的基础上引入**数据依赖的门控机制 (data-dependent gating)**，使 decay rate 不再是固定值而是由输入数据动态决定：γ_n = sigmoid(X_n W_γ)^{1/τ}，其中 τ 是温度参数鼓励 γ 趋向 1 以获得更好的记忆能力。门控是 head-wise 的（而非 element-wise），使计算可以充分利用 NVIDIA Tensor Core。核心创新在于统一了三种等价的计算范式：(a) Parallel 模式用于训练（充分利用 GPU 并行）；(b) Recurrent 模式用于自回归推理（O(1) 常量 KV 内存）；(c) Chunkwise Recurrent 模式用于 prefill（chunk 内并行 + chunk 间 recurrent，节省 FLOPs 并减少迭代次数）。这三种范式在数学上等价，输出结果相同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 YOCO Self-Decoder 中使用 gated retention 的自回归推理（recurrent mode）为例：

```python
# Recurrent Gated Retention (decode mode, single head)
# Input: x_n ∈ R^d (current token), S_{n-1} ∈ R^{d×d} (previous state)
# Weights: W_Q, W_K, W_V ∈ R^{d×d}, W_γ ∈ R^{d×1}
# Hyperparams: τ (temperature), θ (RoPE base)

Q_n = (x_n @ W_Q) * Θ_n          # Θ_n = e^{inθ}, RoPE applied
K_n = (x_n @ W_K) * Θ̄_n          # Θ̄_n = e^{-inθ}, conjugate RoPE
V_n = x_n @ W_V
γ_n = sigmoid(x_n @ W_γ)^{1/τ}   # data-dependent gate, head-wise

# State update: O(d²)
S_n = γ_n * S_{n-1} + K_n^T @ V_n   # outer product K_n^T · V_n ∈ R^{d×d}

# Output: O(d²)  
O_n = Q_n @ S_n                   # vector-matrix product
O_n = GroupNorm_h(O_n)            # per-head GroupNorm
O_n = swish(x_n @ W_G) * O_n      # swish gate for non-linearity
O_n = O_n @ W_O                   # output projection
```

**Annotations**: d = head_dim (e.g., 128 for YOCO-3B)。S_n 是唯一的中间状态（O(d²) = 128² = 16K floats），不存储 per-token KV cache。γ_n ∈ (0,1) 控制信息保留程度——趋近 1 时接近等权保留（更好记忆），趋近 0 时快速遗忘。温度 τ 默认较大（如 8），将 sigmoid 输出推向 1。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Gated Retention 的实现分为三种模式在不同阶段使用：(1) **训练阶段**使用 Parallel 模式，QK^T⊙D（causal decay mask）再乘 V，与标准 self-attention 接口兼容，可直接替换；(2) **Prefill 阶段**使用 Chunkwise Recurrent（chunk_size=256），chunk 内用 parallel 计算利用 Tensor Core，chunk 间通过 recurrent state R 传递信息；(3) **Decode 阶段**使用 Recurrent 模式，仅维护 S ∈ R^{d×d} 状态矩阵，每 token 更新为 O(d²) 计算量。实现基于 Triton kernel（FLA 库：https://github.com/sustcsonglin/flash-linear-attention）。数据依赖门控 γ 为 head-wise 而非 element-wise，使 decay mask D 形成低秩结构，可利用 Tensor Core 高效计算。与标准 self-attention 的 O(N²d) 内存不同，gated retention 推理时仅需 O(d²) 常量内存。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
