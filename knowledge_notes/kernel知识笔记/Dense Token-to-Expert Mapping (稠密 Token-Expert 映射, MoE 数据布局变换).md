## Dense Token-to-Expert Mapping (稠密 Token-Expert 映射, MoE 数据布局变换)

术语解释
Dense Token-to-Expert Mapping 是 DeepSpeed-MoE 推理系统提出的 MoE 优化技术：用稠密的 token-to-expert 映射表（expert_id[S] + local_id[S]）替代传统 sparse one-hot 掩码（S×E）进行 MoE token routing，将 token 排序和反排序实现为显式数据布局变换而非 sparse-dense einsum 乘法。

术语是什么？
传统 MoE 实现中 token routing 通过 sparse einsum 完成：创建 S×E 的 one-hot 掩码矩阵 M（M[t][e]=1 if token t → expert e），执行 M @ X 将 token 分配到各 expert。此操作复杂度 O(S×E×M)，而 one-hot 矩阵中仅 S 个非零元素（top-1 gating 下），(E-1)/E 的运算为与零相乘。

Dense Token-to-Expert Mapping 替代方案：
- 用 expert_id[S]（稠密数组，每元素为 0..E-1 的 expert id）替代 S×E 的 sparse 矩阵
- 用 local_id[S]（每 token 在其目标 expert 中的局部位置）替代 sparse scatter
- 用 expert_offset[E+1]（每 expert 的起始位置，由 cumsum 计算）组织输出缓冲区
- Token 排序 = 直接按 expert_id + local_id 索引 memcpy → 无需矩阵乘法
- Token 反排序 = 逆向索引 memcpy + gate probability 缩放 → 无需矩阵乘法

从kernel调度角度拆解术语：
```
// 传统方法：Sparse Einsum
// M: [S, E] one-hot, X: [S, M]
// O = M^T @ X       // [E, S] @ [S, M] → [E, M], Sparse×Dense, O(S×E×M) ops
// 其中 (E-1)/E 为零乘法

// 优化方法：Data Layout Transform
// Input:  X[S][M], expert_id[S], local_id[S], expert_offset[E+1]
// Output: X_sorted[E][ce][M]

// Sort (by expert_id):
for t in 0..S:
    e = expert_id[t]
    pos = local_id[t]                    // 已由 cumsum 计算
    X_sorted[e][pos] = X[t]              // Direct memcpy, no multiply

// Unsorted (back to original order) with gate probability:
for t in 0..S:
    e = expert_id[t]
    pos = local_id[t]
    output[t] = gate_prob[t] * Y_expert[e][pos]   // 融合 probability scaling

// 复杂度对比：
// Sparse Einsum: S × E × M × ce → O(S·E·M·ce)（立方+零运算）
// Layout Transform: S × M × ce → O(S·M·ce)（仅非零元素）
```

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理系统（开源：https://github.com/microsoft/DeepSpeed）
- 需要 Gating Kernel Fusion 提前计算 expert_id[], local_id[], expert_offset[]
- 内存布局要求：所有 expert 的 token buffer 预分配为 [E, capacity, M]
- 对于 capacity > local_id 的空余位置填充为零（不影响后续 FFN 计算）

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale
