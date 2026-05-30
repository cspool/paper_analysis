## Exponential Gating (in xLSTM/mLSTM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Exponential Gating 是 xLSTM 架构中 mLSTM cell 使用的门控机制，与传统 LSTM 的 sigmoid/tanh 门控不同。mLSTM 的输入门 i_t ∈ R 和遗忘门 f_t ∈ R 是标量（per head），使用指数函数激活而非 sigmoid：(1) **遗忘门** f_t = exp(log σ(f̃_t) + m_{t-1} - m_t)，结合 sigmoid 的对数和 max state 的差值来得到指数形式；(2) **输入门** i_t = exp(ĩ_t - m_t)，直接通过指数激活。这种设计允许门值超出 [0,1] 范围，为记忆状态更新 C_t = f_t·C_{t-1} + i_t·(v_t⊗k_t) 提供更灵活的缩放。**max state** m_t = max(log σ(f̃_t) + m_{t-1}, ĩ_t) 用于数值稳定，防止指数溢出。指数门控使 mLSTM 能在大范围值上进行记忆更新，这是其与 SSM/Mamba 的关键区别（Mamba 使用 selective scalar gating 但无独立的输入/遗忘门对）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Exponential Gating 的完整流程（per head, per timestep）
// Gate pre-activations (应用 soft-capping a=15)
ĩ_tilde = softcap_15(w_i^T @ x_t + b_i)  // scalar
f̃_tilde = softcap_15(w_f^T @ x_t + b_f)  // scalar

// Max state update (数值稳定关键)
m_t = max(log(σ(f̃_tilde)) + m_{t-1}, ĩ_tilde)

// Exponential gate activations
f_t = exp(log(σ(f̃_tilde)) + m_{t-1} - m_t)  // ∈ (0, 1] 实际
i_t = exp(ĩ_tilde - m_t)                       // ∈ (0, ∞)

// Memory update
C_t = f_t · C_{t-1} + i_t · (v_t ⊗ k_t)

// 为什么用指数？相比 sigmoid:
// - sigmoid: i_t ∈ (0,1), 值域受限, 不能"超量"写入
// - exponential: i_t 可 >1, 允许新信息以更高权重写入记忆
// - f_t 始终 ≤1 (因 m_t 定义), 实现稳定遗忘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现在 https://github.com/NX-AI/xlstm（PyTorch）和 https://github.com/NX-AI/xlstm-jax（JAX）
- 重要实现细节：
  - 输入门 bias 初始化为 -10（大的负值），使初始状态 i_t ≈ exp(-10) ≈ 0，训练初期模型依赖前一步记忆而非新输入，有效降低早期梯度尖峰
  - Gate pre-activations 使用 softcap_a(x) = a·tanh(x/a)（a=15 用于 gates, a=30 用于 logits）
  - 在 EOD token 处通过置 f_t = 0 使完整重置记忆（序列打包时防止跨文档信息泄露）
- 与 GLA (Gated Linear Attention) 的关系：两者都使用门控线性递归，但 mLSTM 使用标量指数门（per head）而非向量门

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---
