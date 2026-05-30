## Structured State Space Duality (SSD) / Mamba2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured State Space Duality (SSD) 是 Dao 和 Gu (2024) 在 Mamba2 中提出的理论框架，揭示了**选择性 SSM 与线性注意力在数学上的对偶关系**——两者都通过半可分矩阵（semiseparable matrix）表达序列变换，仅是同一矩阵的两种分解视角。Mamba2 的核心改进：(1) **Multi-input SSM**：引入 head 结构（类似 Multi-Head Attention），$B_t, C_t$ 在 channel 间共享（类比 grouped-query attention），state group size 可配（默认为 8 组，每组 128 channel）；(2) **简化 block 设计**：$x_t, B_t, C_t, \Delta_t$ 由单次投影并行生成 $(x_t, B_t, C_t, \Delta_t) = F(u_t)$，消除 Mamba1 的 sequential linear，使 tensor parallelism 仅需 1 次 all-reduce（vs Mamba1 的 2 次）；(3) **SSD 算法**：利用分块矩阵分解将 SSM 计算转化为 chunked matmul + short scan，充分利用 GPU Tensor Core（BF16 matmul 约 16× 快于 FP32 逐元素运算），训练速度比 Mamba1 快 2-8×；(4) **更大 state dimension**：$d_{state}$ 从 16 扩展到 64-256；(5) **channel-wise 计算**：SSD scan 是逐 channel 独立计算，因此输出 channel 顺序保持与输入一致（channel order preserving）。Mamba2 的 head 结构和 grouped state 设计直接启发了 Quamba2 的 per-state-group quantization 和 sort-and-cluster 技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Mamba2 SSD block（单 token 推理）
# 输入: u_t ∈ R^D
# 参数: W_in ∈ R^{D×P}, A ∈ R^{d_state}, W_out ∈ R^{P×D}

# Step 1: 并行投影 (x, z, B, C, Δ 一次生成)
proj = W_in @ u_t                                  # R^{P}
x_t = proj[:d_inner]                                # 激活输入
z_t = proj[d_inner:2*d_inner]                       # residual branch
B_t = proj[2*d_inner:2*d_inner+n_heads*d_state]     # 每 head B
C_t = proj[2*d_inner+n_heads*d_state:...]           # 每 head C
Δ_t = softplus(proj[...])                           # 时间步长

# Step 2: Causal conv1d (替代 Mamba1 的独立 conv)
x_conv = causal_conv1d(x_t, W_conv)

# Step 3: SSD scan (channel-wise, 逐 head 独立)
h_t = A_t * h_{t-1} + B_t * x_conv_t               # state update
y_ssd = C_t @ h_t                                    # output via state

# Step 4: Hadamard + output projection
y_out = W_out @ (y_ssd * SiLU(z_t))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/state-spaces/mamba（mamba-2 分支）。Mamba2 发布 checkpoint：2.7B (300B tokens Pile), 8B (1.2T tokens)。适合：(1) 作为量化/压缩研究的目标 backbone（Quamba2, MambaQuant）；(2) 长序列推理（SSD chunked scan 支持可变长度序列）；(3) hybrid 架构的 SSM 组件。注意 SSD 的 channel order preserving 特性是 Quamba2 sort-and-cluster 量化的基础前提。

涉及论文标题：
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
