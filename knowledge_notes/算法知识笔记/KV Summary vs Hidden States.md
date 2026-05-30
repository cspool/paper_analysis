## KV Summary vs Hidden States

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Summary 是 MHLA 论文引入的术语，用于严格区分 MHLA 中的 key-value summary 与传统 linear attention 中的 Hidden State。虽然二者在符号上可能相似（均为 d×d 矩阵），但底层计算和依赖图有本质区别：

1. **依赖关系**：传统 linear attention 的 Hidden State h_t 通过严格递推链 h_t = f(h_{t-1}, k_t, v_t) 更新，h_t 依赖于 h_{t-1}，存在状态传播。MHLA 的每个全局 KV Summary S_g 独立计算：S_g = Σ_b m_{g,b} S_b，各局部 summary S_b 相互独立并行计算，无状态传播。

2. **聚合模式**：传统 Hidden State 是一对一递推更新（one-to-one）；MHLA 的 KV Summary 是多对一聚合（many-to-one）——每个 S_g 由所有局部 S_b 通过特定混合系数 m_{g,b} 聚合而成。

这种设计避免了 Hidden State 中历史信息的刚性继承，使 MHLA 的 summary 具有更高的表达能力和灵活性。

从算法pipeline角度拆解术语。

```
// Traditional Hidden State (recurrent chain)
h_0 = 0
for t in 1..N:
    h_t = h_{t-1} + phi(k_t)^T @ v_t    // 严格递推，h_t ← h_{t-1}
    o_t = phi(q_t)^T @ h_t

// MHLA KV Summary (independent + mixture)
for b in 1..M:  // 并行
    S_b = sum(phi(k_j)^T @ v_j for j in block b)  // 独立计算
for i in 1..M:
    S̃_i = sum(m_{i,b} * S_b for b in 1..M)       // 多对一聚合
    o_i_block = q_i_block @ S̃_i
```

术语一般如何实现？如何使用？

该术语区分本身不涉及具体实现，但其概念影响架构设计：由于 MHLA 的 KV summary 独立计算，各 block 可完全并行化（训练时），且推理时可增量更新（causal inference 中仅更新当前 block 的 S_b 并重新计算受影响的 S̃_i）。这种设计更适合 GPU 的大规模并行计算。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---
