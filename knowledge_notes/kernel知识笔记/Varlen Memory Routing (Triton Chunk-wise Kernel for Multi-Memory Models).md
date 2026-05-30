## Varlen Memory Routing (Triton Chunk-wise Kernel for Multi-Memory Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Varlen Memory Routing 是 MoM（Mixture-of-Memories）论文提出的一种硬件高效实现技术。核心思想：将 MoM 的多 memory 更新转化为 varlen（variable-length）Triton kernel 操作，通过 token reordering + varlen computation 避免 naive 实现的低效。

Naive 实现问题：如果逐 token 对每个 memory 执行 update，每个 token 需要 dispatch 到不同 memory 并 gather 结果，导致大量 GPU kernel launch overhead 和 memory access 不连续。

MoM 解决方案（Fig 2）六步流程：
1. Token 按 routing 结果分组到 memory bucket
2. 同 bucket tokens concat 为 varlen 序列
3. Triton kernel F_m 对每个 segment 独立计算
4. 输出返回各 bucket
5. 按原始 token 顺序拆分
6. Weighted sum 恢复最终输出

从kernel调度角度拆解术语。

**Varlen Memory Routing 的 Triton Kernel 实现**：

```
# 输入准备 (Step 1-2):
对于 batch b:
  for each memory m:
    I_{b,m} = {t : token t routed to memory m}  # 按路由分组
    L_{b,m} = |I_{b,m}|                          # 各 memory 的 token 数
    s_p = CumulativeSum(L)                       # varlen boundaries

# 展平序列:
X̃ = concat([X[I_{1,1}], ..., X[I_{B,M}]])       # [total_tokens, d]

# Triton kernel (Step 3):
grid = (B*M, )
for each block (b, m) in parallel:
    start, end = s_{p-1}, s_p
    X_seg = X̃[start:end]                         # 加载该 memory 的 token 段

    # QKV projection
    Q = X_seg @ W_Q                               # 共享 Q projection
    K = X_seg @ W_K^{(m)}                         # memory-specific K
    V = X_seg @ W_V^{(m)}                         # memory-specific V

    # Chunk-wise parallel scan (复用已有 linear model kernel):
    O_seg = chunk_parallel_scan(Q, K, V, update_fn=GatedDeltaNet)

    O[start:end] = O_seg

# 输出恢复 (Step 4-6):
对每个原始位置 t:
  # 从各 memory 的输出中 gather
  O_t = {o_t^m for m in activated_memories(t)}
  y_t = Σ g_t^{(m)} · o_t^m          # weighted sum
```

关键优化：
- Token reordering 按 memory 分组，将稀疏的路由模式转化为密集的连续内存访问
- Varlen 操作避免 padding，每个 (batch, memory) 组独立处理
- 可复用已有 linear model 的 Triton chunk-wise parallel scan kernel
- Q projection 在所有 memory 间共享（W_Q 不分离），K/V projection 各 memory 独立

术语一般如何实现？如何使用？

基于 Triton 实现，集成在 Linear-MoE 框架中。MoM 的实验使用 32×A800 GPU 训练。varlen kernel 的关键是正确管理 token reordering 的索引映射——前向时按 routing 分组，反向时按原始顺序恢复梯度。与 FlashAttention varlen 类似，使用 cu_seqlens 指定 segment boundaries。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories
