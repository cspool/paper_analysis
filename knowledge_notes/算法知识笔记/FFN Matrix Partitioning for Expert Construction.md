## FFN Matrix Partitioning for Expert Construction

术语解释
FFN Matrix Partitioning 是 DSMoE 的核心技术：将预训练 Dense 模型的 SwiGLU FFN 层的三个权重矩阵（W_gate、W_up、W_down）沿 intermediate/expansion 维度均等划分为 n 组，每组构成一个独立的 expert FFN，全部参数保留、知识零损失地在数学上保证等价性。这是 Dense-to-Sparse MoE 转换中最简洁的参数复用策略。

术语是什么？
给定预训练 SwiGLU FFN 的输出公式：
$$h = (SiLU(xW_{gate}) \odot (xW_{up})) W_{down}$$

将三个矩阵沿 intermediate dimension D 等分为 n 段：
$$W_{gate} = [W_1^{gate} \| \cdots \| W_n^{gate}],\quad W_{up} = [W_1^{up} \| \cdots \| W_n^{up}],\quad W_{down} = [V_1 \| \cdots \| V_n]^T$$

其中每组 W_i^{gate} ∈ R^{d×D/n}、W_i^{up} ∈ R^{d×D/n}、V_i ∈ R^{D/n×d} 构成 expert i 的参数。

**等价性证明**：通过矩阵分块乘法的分配律：
$$h = \sum_{i=1}^{n} (SiLU(xW_i^{gate}) \odot (xW_i^{up})) V_i$$

所有 expert 输出之和在数学上严格等于原始 Dense FFN 输出。这是 DSMoE "零知识损失"转换的理论基础。

从算法pipeline角度拆解术语：
Partitioning 的完整流程（以 LLaMA-7B, d=4096, D=11008, n=8 为例）：

```
=== Pre-trained Dense FFN → Partitioned Experts ===

原始 Dense FFN:
    W_gate: [4096, 11008]  → 划分为 8 个 [4096, 1376]
    W_up:   [4096, 11008]  → 划分为 8 个 [4096, 1376]
    W_down: [11008, 4096]  → 划分为 8 个 [1376, 4096]

Expert i (i = 0, ..., 7):
    Expert_i.W_gate = W_gate[:, i*1376 : (i+1)*1376]  # [4096, 1376]
    Expert_i.W_up   = W_up[:,   i*1376 : (i+1)*1376]  # [4096, 1376]
    Expert_i.W_down = W_down[i*1376 : (i+1)*1376, :]  # [1376, 4096]


=== Expert FFN Forward (per expert i) ===

def expert_i_forward(x, expert_i):
    # x: [B, 4096]
    gate = silu(x @ expert_i.W_gate)  # [B, 1376]
    up   = x @ expert_i.W_up          # [B, 1376]
    out  = (gate * up) @ expert_i.W_down  # [B, 4096]
    return out


=== Full Partitioned FFN (all experts active) ===

def partitioned_ffn_full(x, experts):
    # 所有 expert 激活时 = 原始 Dense FFN（等价性保证）
    outputs = [expert_i_forward(x, exp) for exp in experts]
    h = sum(outputs)  # [B, 4096] ≡ 原始 FFN(x)
    return h


=== Sparse Partitioned FFN (DSMoE inference) ===

def dsmo_e_ffn_sparse(x, experts, Y, tau=0.5):
    gate_probs = sigmoid(x @ Y)  # [B, 8]
    active_mask = gate_probs > tau
    num_active = active_mask.sum()
    
    h = zeros_like(x)  # [B, 4096]
    for i in range(8):
        if active_mask[:, i].any():
            h += expert_i_forward(x, experts[i]) * gate_probs[:, i:i+1]
    
    # 激活数归一化: 保持输出范数稳定
    h = h * (8 / num_active.clamp(min=1))
    return h
```

术语一般如何实现？如何使用？
- **划分策略**：等分为最简单的方案，论文未探索非均匀划分（如根据 expert 重要性分配不同大小的 intermediate dimension slice）
- **expert 数量选择**：论文固定使用 n=8（LLaMA-1B: D=1024×8, LLaMA-7B: D=1376×8），更多 expert → 更细粒度的激活控制但每个 expert 容量更小
- **与其他 Dense-to-MoE 方法的对比**：
  - LLaMA-MoE (Zhu et al., 2024)：类似的分区方案但使用传统 top-k softmax 路由
  - MoEfication (Zhang et al., 2022)：基于 ReLU 激活的 expert 分区，需要额外转换步骤适配 SiLU/GeLU
  - FactorLLM (Zhao et al., 2024)：多阶段训练（teacher-student），路由器先训练后冻结
  - DSMoE 的优势：最简分区方案 + 端到端训练 + 数学等价性保证
- **与 Expert Decomposition (Low-Rank) 的区别**：FFN Matrix Partitioning 是沿 intermediate dimension 的结构化切分（preserves full rank per expert），而非低秩近似
- **局限性**：等分策略隐含假设 intermediate dimension 各部分的"知识"是均匀分布的，如果预训练模型的 FFN 神经元存在显著的功能聚类，等分可能破坏这些功能单元

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

---
