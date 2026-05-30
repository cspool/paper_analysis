## mLSTM (Matrix LSTM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
mLSTM (Matrix LSTM) 是 xLSTM 架构（Beck et al., 2024, NeurIPS 2024）的核心组件，将传统 LSTM 的标量 cell state c_t ∈ R 扩展为矩阵记忆状态 C_t ∈ R^{d_qk × d_hv}，通过 outer product 更新：C_t = f_t · C_{t-1} + i_t · (k_t ⊗ v_t)（⊗ 表示外积 v_t k_t^T），使记忆容量从标量提升到矩阵级别。其关键特性包括：(1) **全并行化训练**：由于递归的线性性质（C_t 更新为线性组合），可通过 chunkwise-parallel 模式训练，速度与 Flash Attention 相当甚至更快；(2) **常量推理记忆**：自回归生成时仅需 O(d_qk × d_hv) 的常量 GPU 内存，不随序列长度增长（vs Transformer 的 O(T) KV Cache）；(3) **指数门控**：标量输入门 i_t 和遗忘门 f_t 使用指数激活（i_t = exp(ĩ_t - m_t), f_t = exp(log σ(f̃_t) + m_{t-1} - m_t)），由 max state m_t 控制数值稳定性；(4) **Multi-head 结构**：类似 Transformer 的多头注意力，xLSTM 有 N_head = d/d_hv 个独立 mLSTM cell，每个 head 维护独立的 (C^(i), n^(i), m^(i))，输出拼接后投影。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// mLSTM Cell Recurrent Step (推理时)
输入: x_t ∈ R^d, 前一状态 (h_{t-1}, C_{t-1}, n_{t-1}, m_{t-1})
参数: W_{q,k,v} ∈ R^{d_{qkv}×d}, W_o ∈ R^{d_hv×d}, w_{i,f} ∈ R^d

// 1. 投影 (per head, head dim d_hv, d_qk = d_hv/2)
q_t, k_t = W_{q,k} @ x_t  // ∈ R^{d_qk}
v_t = W_v @ x_t          // ∈ R^{d_hv}

// 2. Gate pre-activations (scalars per head)
ĩ_t = w_i^T @ x_t + b_i  // input gate pre-activation
f̃_t = w_f^T @ x_t + b_f  // forget gate pre-activation
õ_t = W_o @ x_t + b_o    // output gate ∈ R^{d_hv}

// 3. Gate activations with max state stabilization
m_t = max(log(σ(f̃_t)) + m_{t-1}, ĩ_t)
f_t = exp(log(σ(f̃_t)) + m_{t-1} - m_t)
i_t = exp(ĩ_t - m_t)

// 4. Memory state update (outer product)
C_t = f_t · C_{t-1} + i_t · (v_t ⊗ k_t)  // C_t ∈ R^{d_qk × d_hv}
n_t = f_t · n_{t-1} + i_t · k_t          // n_t ∈ R^{d_qk}

// 5. Hidden state retrieval with normalization
q̃ = q_t / sqrt(d_qk)
h̃_t = C_t^T @ q̃ / max(|n_t^T @ q̃|, exp(-m_t))
h_t = σ(õ_t) ⊙ Norm(h̃_t)

// Multi-head: concat all heads, project
H = Concat(h_t^(1), ..., h_t^(N_head)) @ W_proj^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：
- 官方 PyTorch: https://github.com/NX-AI/xlstm
- 官方 JAX: https://github.com/NX-AI/xlstm-jax
- Triton kernel 库: https://github.com/NX-AI/mlstm_kernels
- 训练时使用 chunkwise-parallel kernel（Tiled Flash Linear Attention），将序列分块、块内 tiled matmul（利用 Tensor Core）、块间通过 recurrent state 传递。
- 推理时使用 recurrent mode（单个 kernel 即可完成 Eq. 2-9 全部计算）或 TensorRT-LLM 部署。
- xLSTM 7B 配置：8 heads, d_hv=512, d_qk=256, d=4096, 32 blocks, 总记忆状态 134.2 MB（float32）。

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---
