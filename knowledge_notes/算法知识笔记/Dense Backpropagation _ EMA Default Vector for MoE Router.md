## Dense Backpropagation / EMA Default Vector for MoE Router

术语解释
Dense Backpropagation 是一种 MoE 路由器训练技术，通过为每个 expert 维护其历史输出 EMA（指数移动平均）的 default vector，在反向传播时为 Router 提供来自所有 N 个 experts 的"dense"梯度信号，而非仅有 Top-K 激活 expert 的稀疏梯度，同时保持前向传播的稀疏计算特性。

术语是什么？
Standard TopK MoE 的 Router 梯度为 ∂y/∂π_i = E_i(x) if i∈TopK else 0，N-K 个未激活 expert 的 Router 行不接收梯度更新。Dense Backpropagation 的目标是近似完整的 dense gradient ∂y/∂π = [E_1(x), ..., E_N(x)]^T（即 Straight-Through Estimator 的理论梯度），而无需实际计算所有 expert 的前向输出。

核心机制：
1. **EMA Default Vector**: 为每个 expert i 维护 Ê_i = β·Ê_i^{(t-1)} + (1-β)·E_i(x)（仅对激活的 expert 更新），近似 expert 输出的期望值 E[E_i(x)]
2. **Dense Forward**: y = Σ π_i · (E_i(x) if i∈TopK else Ê_i)，default vector 参与前向组合
3. **Dense Backward**: ∂y/∂π_i = E_i(x) for i∈TopK, Ê_i for i∉TopK → Router 所有行接收梯度
4. **Error Correction**: 相对于 true dense gradient 的误差 ε_default = (∂L/∂y) Σ_{i∉A} (E_i(x) - E[E_i(x)]) · ∂π_i/∂W，期望为 0

从算法pipeline角度拆解术语：
```
# DefaultMoE Forward + EMA Update (8 experts, TopK=1)
Input: x [B, H], router W [N, H], experts E_0..E_{N-1}
State: EMA_buf [N, H]  # default vectors

# 1. Router forward
pi = Softmax(W @ x)            # [B, N]
A = TopK(pi, K=1)              # indices of selected experts

# 2. Sparse expert computation (only K experts)
y = zeros(B, H)
for i in A:
    activated_x = x[mask[:,i]]  # tokens routed to expert i
    y_i = E_i(activated_x)      # [num_activated_i, H]
    y += gather(pi[:,i]) * scatter(y_i, mask[:,i])
    
    # 3. EMA update with router-weighted average
    weighted_y = pi[mask[:,i], i].unsqueeze(-1) * y_i
    mean_output = weighted_y.sum(dim=0) / pi[mask[:,i], i].sum()
    EMA_buf[i] = beta * EMA_buf[i] + (1-beta) * mean_output

# 4. Dense combination (EMA for non-activated)
for i not in A:
    y += pi[:,i].unsqueeze(-1) * EMA_buf[i]  # [1, H] broadcast

# 5. Backward: dense gradient signal
# dL/d(pi_i) = dL/dy * (E_i(x) for i in A, EMA_buf[i] for i not in A)
# dL/dW[i,:] = sum_b( dL/d(pi_{b,i}) * x[b] )  for ALL i in 0..N-1
```

术语一般如何实现？如何使用？
- **超参数 β**: sparser MoE 需要更低 β（如 32c1: β=0.65, 32c4: β=0.999），因为每个 expert 接收更少 token，default vector 需更快适应
- **Weighted EMA Update**: 按 Router probability 加权更新 EMA，消除 β 的敏感度。不加权时 β=0.9 与 β=0.999 性能差异显著；加权后多个 β 值收敛到相同性能
- **Forward EMA 必要性**: 仅在后向传递中注入 default vector 不如前向+后向都使用。原因是前向使用 default vector 参与模型输出计算使梯度误差项（Eq.9）被 loss 缩小
- **EMA 初始化**: 零初始化优于随机初始化（避免早期噪声信号误导 Router）
- **开销**: O(1) memory per expert × hidden_dim（如 1024 维 × 8 experts × 16 layers ≈ 0.03% 参数增量），throughput 下降 <2%（小模型）或 <0.2%（大模型）
- **训练框架**: gpt-neox + MegaBlocks + liger kernel (Triton)，dropless MoE，AdamW optimizer

涉及论文标题：
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

---
