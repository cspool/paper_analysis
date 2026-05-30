## Memory State in Linear Attention (线性注意力中的记忆状态)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Memory state M ∈ R^{d×d} 是线性注意力递推形式中的核心状态。定义：M_s = Σ_{i=1}^s k_i^T v_i，即前 s 个 token 的 key-value 外积的累积和。M_s 编码了从序列开始到位置 s 的所有历史信息，使得第 s 个 token 的输出可以直接通过 o_s = q_s M_s 计算，无需访问任何之前的 key 或 value。

关键特性：
- M_s 大小固定为 d×d，与序列长度或 token 位置无关
- 递推更新：M_s = M_{s-1} + k_s^T v_s，开销为 O(d²)
- 推理时仅需存储一个 M_s（而非整个 KV cache），实现常量内存推理
- 在分布式 SP 中，每个 chunk t 的 local memory state M_t = K_t^T V_t 也是 d×d 大小，AllGather 通信量仅为 BHd²，与序列长度无关

从算法pipeline角度拆解术语。

**Memory state 在 LASP-2 分布式训练中的使用**：

```
// Chunk t 在设备上计算 local memory state
Q_t, K_t, V_t = X_t @ W_Q, X_t @ W_K, X_t @ W_V   // [C, d]
M_t = K_t^T @ V_t                                    // [d, d]

// AllGather 全局同步
[M_1, ..., M_T] = AllGather([M_1, ..., M_T])

// 全局累积
M_{1:t} = M_{1:t-1} + M_t    // 缓存到 HBM 用于 backward
O_t = Q_t @ M_{1:T}           // 使用全局 memory state
```

术语一般如何实现？如何使用？

Memory state 以 FP16/BF16 存储，形状为 [B, H, d, d]。对于 Linear-Llama3-1B (H=16, d=2048, B=1)，单个 M_t 约 1.07B 参数（~2.14GB FP16）。在 LASP-2 中，M_{1:T} 被缓存到 HBM 以避免 backward 时的重复计算。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
