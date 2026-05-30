## Sliding Window State Computation for Weighted-Sum RNNs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sliding Window State Computation 是 Stuffed Mamba 论文提出的无需训练的推理时遗忘诱导方法。利用 Mamba-2 状态可写为加权和 h_t = Σ_{i=1}^{t} α_{i:t}·B̄_i·x_i 的性质，通过 h_t^{(w)} = h_t - α_{t-w+1:t}·h_{t-w} 精确计算最近 w 个 token 的状态，等价于在序列上滑动一个 w 大小的窗口。维护 h_t（正常状态）、h_{t-w}（w 步前的状态）和 Δ_sum（Δ 的累积和），每步计算 α_window = exp(-Δ_sum·exp(A)) 并通过矩阵减法得到窗口状态。该方法的优势：(1) 无需重新训练；(2) 数学上精确（非近似）；(3) 额外计算和内存开销极小（两个额外状态张量 + 一个标量乘法和矩阵减法）；(4) 适用于所有可写为加权和的 RNN（GLA、RWKV、RetNet 等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sliding Window 推理算法：
```
# 推理时维护三个量
h_t      = 0  # [N, P] 正常递归状态
h_{t-w}  = 0  # [N, P] w 步前的状态（用于减法）
Δ_sum    = 0  # scalar, Δ 累积和
window_size = w

for each token at step t:
    # 1. 正常 Mamba-2 状态更新
    Δ_t = Softplus(W_Δ @ u_t + b_Δ)
    α_t = exp(-Δ_t * exp(A))
    B̄_t = Δ_t * B_t
    h_t = h_{t-1} * α_t + B̄_t * x_t   # [N, P]

    # 2. 维护 Δ 累积和（避免浮点不稳定）
    Δ_sum = Δ_sum + Δ_t

    # 3. 计算窗口衰减因子
    α_window = exp(-Δ_sum * exp(A))

    # 4. 精确窗口状态 = 完整状态 - 窗口前的状态
    h_t^{(w)} = h_t - α_window * h_{t-w}   # [N, P]

    # 5. 使用窗口状态 query
    y_t = C_t @ h_t^{(w)} + D ⊙ x_t        # [1, P]

    # 6. 更新 h_{t-w}（延迟 w 步）
    if t > w:
        h_{t-w} = update_buffer(h_{t-w})   # FIFO 或循环缓冲区
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 用 Ring Buffer 维护 h_{t-w} 的历史；(2) 维护 Δ_sum 而非直接计算 α_window 的乘积，因为 Δ ∈ R，求 exp(-sum·exp(A)) 比连乘 exp(-Δ_i·exp(A)) 更数值稳定；(3) 窗口大小 w 是超参数，对短上下文性能有影响（窗口太小则信息不足，太大则遗忘不足）。Stuffed Mamba 实验表明，Sliding Window 在 32K 上下文上将 Mamba-2 370M 的 LM loss 从 ~15 降至 ~8-10，但短上下文性能略有下降。适用场景：已有训练好的 Mamba-2 模型、需要处理超训练长度的上下文、不希望或无法重新训练时的推理时干预。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
