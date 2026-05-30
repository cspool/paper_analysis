## Parallel Associative Scan (Blelloch Scan) for Recurrent Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parallel Associative Scan (Blelloch scan) 是将顺序递推 h_t = f(h_{t-1}, x_t) 并行化为 O(log L) depth 的算法，前提是 f 满足结合律。在 Mamba SSM 中，递归 h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t 的关联操作为 elem=(a,b), binop: (a,b) ⊕ (a',b') = (a'⊙a, a'⊙b + b')。S5 (Smith et al., 2023) 首次将 parallel scan 用于 SSM，但需切换为 MIMO 降低 state 维度。Mamba 保持 SISO 高 state 维度 + 硬件感知实现克服计算问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Blelloch Scan for SSM: elem=(a,b), ⊕=(a'·a, a'·b+b')

// Up-Sweep (Reduce): combine adjacent pairs
for d = 0 to log₂(L)-1:
    stride = 2^{d+1}, gap = 2^d
    for k in 0..L/stride-1 (parallel threads):
        left = k*stride + gap - 1
        right = k*stride + stride - 1
        (a_r, b_r) = data[right]
        (a_l, b_l) = data[left]
        data[right] = (a_r·a_l, a_r·b_l + b_r)  // combine

// Down-Sweep (Distribution): propagate prefix
data[L-1] = (1, 0)  // identity element
for d = log₂(L)-1 down to 0:
    stride = 2^{d+1}, gap = 2^d
    for k in 0..L/stride-1 (parallel threads):
        left = k*stride + gap - 1
        right = k*stride + stride - 1
        tmp = data[left]
        data[left] = data[right]
        data[right] = data[right] ⊕ tmp

// Output: data[t] = h_t for all t
```
在 GPU 上以 tile 为粒度实现：每 SM 处理 chunk 内 scan → 跨 SM partial scan → 组合。Mamba 的 fused kernel 将 scan 完全限定在 SRAM 完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
适用于任何满足结合律的递推关系（RNN/GRU/LSTM 等 gate-based RNN）。Work = O(L), Depth = O(log L)。在 Mamba 中与离散化和输出乘加融合为单一 kernel。关键限制：需要 L 对齐到 2 的幂（pad if needed）；仅支持关联操作（不要求交换律）。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
