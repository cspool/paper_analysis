## Fused Generation Kernels for Recurrent LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Generation Kernels for Recurrent LLMs 是将递归神经网络的多个独立 GPU kernel 调用融合为单个 kernel 的优化技术。以 xLSTM 7B 的 mLSTM cell 为例，其自回归生成时的 recurrent 公式（Eq. 2-9）涉及 outer product（v_t ⊗ k_t）、多个 dot product、max 操作、exp 操作和 pointwise 乘法——在标准实现中每个操作都是一个独立 GPU kernel 调用。每个 kernel 需要从 HBM 加载输入并将输出写回 HBM，大量慢速内存操作成为瓶颈。Fused kernel 将所有操作在单个 kernel 中完成，中间结果（gate values、outer product 部分和、normalizer update 等）保持在 GPU SM 的 SRAM/Register File 上，仅最终 hidden state h_t 和更新后的 recurrent state (C_t, n_t, m_t) 写回 HBM。xLSTM 7B 的 Triton-based fused generation kernel 开源在 https://github.com/NX-AI/mlstm_kernels。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Fused mLSTM Recurrent Kernel 伪代码 (单次 timestep)
// GPU Grid: 1 block per head (共 N_head 个 block 并行)
// 每个 block 在 SM SRAM 上执行:

输入: x_t ∈ R^{d_model}, state_{t-1} = (C_{t-1}, n_{t-1}, m_{t-1}) (均在 HBM)
输出: h_t ∈ R^{d_model}, state_t = (C_t, n_t, m_t) (写回 HBM)

// 1. 从 HBM 加载 state 到 SRAM
C_prev = load_from_hbm(C_{t-1})   // d_qk × d_hv floats
n_prev = load_from_hbm(n_{t-1})   // d_qk floats
m_prev = load_from_hbm(m_{t-1})   // 1 float

// 2. Gate computation (在 SRAM)
q = W_q @ x_t    // d_qk
k = W_k @ x_t    // d_qk
v = W_v @ x_t    // d_hv
i_tilde = softcap_15(w_i^T @ x_t + b_i)    // scalar
f_tilde = softcap_15(w_f^T @ x_t + b_f)    // scalar

// 3. State update (全部在 SRAM, 不写 HBM)
m_cur = max(log_sigmoid(f_tilde) + m_prev, i_tilde)
f = exp(log_sigmoid(f_tilde) + m_prev - m_cur)
i = exp(i_tilde - m_cur)

// 4. Memory update (outer product in SRAM)
C_cur = f * C_prev + i * (v ⊗ k^T)  // rank-1 update
n_cur = f * n_prev + i * k

// 5. Hidden state retrieval
q_norm = q / sqrt(d_qk)
h_tilde = C_cur^T @ q_norm / max(|n_cur^T @ q_norm|, exp(-m_cur))
o = sigmoid(W_o @ x_t + b_o)
h = o ⊙ LayerNorm(h_tilde)

// 6. 写回最终结果到 HBM
store_to_hbm(h)      // d_model floats
store_to_hbm(C_cur)  // d_qk × d_hv floats (next step state)
store_to_hbm(n_cur)  // d_qk floats
store_to_hbm(m_cur)  // 1 float

// 关键优化: C_prev, C_cur 的加载/存储在 SRAM 内完成
// outer product v⊗k^T 直接用 register tile 在片上计算
// 相比 unfused: 中间 C_prev 和 C_cur 的多次 HBM 读写被消除
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源 Triton kernel 实现：https://github.com/NX-AI/mlstm_kernels
  - `recurrent` mode: 用于自回归推理的 fused step kernel
  - `chunkwise` mode: 用于训练的 chunkwise-parallel kernel (TFLA)
  - `parallel` mode: 二次复杂度 attention-like kernel（用于短序列或验证）
- 在 HuggingFace transformers 中使用：加载 xLSTM 7B 模型后，`model.generate()` 自动调用 fused recurrent kernel
- PyTorch 集成：`torch.compile` + CUDA Graphs 进一步减少 kernel launch overhead
- 适用范围：任何具有固定大小 recurrence state 的线性 RNN（mLSTM、Mamba、RWKV、GLA 等）均可受益于此类 fused kernel

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference
