## Attention-MoE Pipelining

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention-MoE Pipelining 是 FOLDMOE 论文提出的核心创新——将 token-level 的通信-计算重叠从仅 MoE 层扩展到整个 Transformer block（同时包含 attention 层和 MoE 层）。传统 MoE-only overlapping（如 Tutel）仅在 MoE 层内部做 token-level pipelining，但 expert computation 计算量小，无法充分隐藏 A2A 通信延迟（32K seqlen 时 expert 仅占 21% 执行时间）。FOLDMOE 利用 attention 层的 O(n²) 计算量（随序列长度平方增长）覆盖 A2A 通信，将 Transformer block 重组为四级流水线：

Stage 1: Attention computation (Token micro-batch i)
Stage 2: A2A dispatch (Token micro-batch i)
Stage 3: Expert computation (Token micro-batch i)
Stage 4: A2A combine (Token micro-batch i)

通过 causal attention 的 KV cache 累积特性（计算 token t 只需前 t-1 个 token 的 K/V），可在 sequence 维度上对 attention 做微批次划分，使不同微批次的 attention 计算和 MoE 通信/计算在分离的 CUDA stream 上并行执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Attention-MoE Pipelining 在一个 Transformer block 内的执行伪代码：

```
# 序列 X[0..L-1], 切片方案 S={l1,...,ld}, Token Buffer B
K_prev, V_prev = [], []
start = 0

for j in 0..d-1:                         # attention 按时间均匀切片
    l_j = S[j]
    X_mb = X[start : start+l_j]
    # === Stage 1: Attention (Compute Stream) ===
    K_mb, V_mb = W_k(X_mb), W_v(X_mb)
    K_all = [K_prev; K_mb], V_all = [V_prev; V_mb]
    Z_mb = FlashAttn(Q=W_q(X_mb), K=K_all, V=V_all, causal=True)

    B.enqueue(Z_mb)                      # 存入 token buffer

    # === Stages 2-4 (Comm Stream, 可与 Stage 1 重叠) ===
    while B.size >= ceil(L/d):           # MoE 侧按 token 数量均匀取
        Z_moe = B.dequeue(ceil(L/d))
        Z_disp = A2A_dispatch(Z_moe)     # Stage 2
        Y_exp = Experts(Z_disp)          # Stage 3
        Y_moe = A2A_combine(Y_exp)       # Stage 4
        Y.append(Y_moe)

    K_prev, V_prev = K_all, V_all
    start += l_j

# Drain buffer (cool-down phase)
while B not empty:
    ...  # 同上 Stages 2-4
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FOLDMOE 基于 Megatron-LM 框架实现，修改了 Transformer block 的执行流程。与 FlashAttention 兼容（因 micro-batch causal attention 与全序列 causal attention 的 mask pattern 一致），与 TP 正交（TP 切分算子，FOLDMOE 切分序列），与 SP 兼容（SP 仅操作 layernorm/dropout 等非 attention/MoE 区域）。配置参数为 overlap degree d（微批次数量），通过 runtime profiling 确定最优 d（d 过小则 bubble 大，过大则 kernel launch overhead 超重叠收益）。论文未开源代码。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
