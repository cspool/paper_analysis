## SSM Recurrent States (for LLM Inference)

术语是什么？
SSM Recurrent State是State Space Model层在LLM推理时维护的固定大小hidden state h_t ∈ R^{d_state × d_model}（或Mamba2的变体形式）。与Attention的KV cache（per-token存储、大小随序列长度线性增长）不同，SSM state通过in-place递推更新：h_t = f(h_{t-1}, x_t)，每个新token更新整个state而非追加新条目。这使得SSM state具有两个关键特性：(1) 固定内存占用——无论序列多长，state大小始终为d_state × d_model × sizeof(fp32)；(2) 不可回滚——state是sequence-level的累积表示，无法像KV cache那样通过切片回退到前缀的任意中间位置。

从算法pipeline角度拆解术语：
```
// Mamba2 SSM state更新 (简化):
Input: x_t ∈ R^d_model, h_{t-1} ∈ R^{d_state × d_model}
Parameters: A, B, C, Δ (输入依赖)

Step 1: Δ_t = softplus(W_Δ·x_t + b_Δ)         // 选择性时间步长
Step 2: Ā_t = exp(Δ_t · A)                     // 离散化状态转移矩阵
Step 3: B̄_t = (Δ_t·A)^{-1}·(exp(Δ_t·A)-I)·Δ_t·B  // 离散化输入投影
Step 4: h_t = Ā_t·h_{t-1} + B̄_t·x_t             // In-place递推更新!
Step 5: y_t = C_t·h_t                           // 输出

// 关键: h_t 直接覆盖 h_{t-1}，不保留历史版本
// h_5 可以表示序列[1..5]，但无法回退表示[1..3]
```

Implication for prefix caching:
- Attention: K_{1..5}, V_{1..5} → 可直接取子集 K_{1..3}, V_{1..3} 表示前缀
- SSM: h_5 → 无法从h_5推导h_3 → 必须单独checkpoint h_3 才能复用前缀[1..3]
```

术语一般如何实现？如何使用？
Mamba/Mamba2 CUDA kernel实现recurrent state更新。Training时使用parallel scan（所有时间步并行计算），Inference时使用recurrent mode（逐token递推）。SSM state在GPU memory中的大小：Mamba2-Hybrid-7B d_state=128, d_model=4096 → 约128×4096×4=2MB per layer。24 SSM layers → 约48MB per sequence（远大于单token KV但远小于完整序列KV）。Marconi通过每序列至多2个checkpoint控制总缓存中SSM state数量。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---
