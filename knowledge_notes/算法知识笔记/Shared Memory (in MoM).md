## Shared Memory (in MoM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Shared Memory 是 MoM 架构中的一个特殊 memory state，始终被所有 token 激活（不走 Router 的 top-k 筛选），用于捕获全局序列信息。设计动机：多 memory 分离虽然消除了 memory interference，但也导致每个 memory 只能看到部分序列。Shared memory 获取完整序列上下文，弥补分离式 memory 可能丢失的跨 memory 长程依赖。

Shared memory 的灵感来自 DeepSeek-MoE 中的 shared experts——捕获跨不同上下文的共性知识。在 MoM 中，shared memory 作为所有 memory 的"背景知识库"，输出时与 top-k activated memories 的输出一起做加权混合。

从算法pipeline角度拆解术语。

```
# MoM 含 Shared Memory 的前向流程:
M_shared = 0                                    # d×d
M_1...M_M = 0                                   # M 个 d×d

for t in 1..T:
  # Router: 选 top-k 个 memory
  scores = TopK(softmax(x_t @ W_g), k)

  # Shared memory: 始终更新
  k_t^shared = x_t @ W_k^shared
  v_t^shared = x_t @ W_v^shared
  M_shared = GatedDeltaNet(M_shared, k_t^shared, v_t^shared)

  # Top-k memories: 选择性更新
  for m in topk_indices:
    M_m = GatedDeltaNet(M_m, k_t^m, v_t^m)

  # 混合输出:
  M̃_t = Σ g_t^{(m)} · M_t^m + M_t^shared
  o_t = q_t @ M̃_t
```

术语一般如何实现？如何使用？

Shared memory 使用独立的 K/V projection weights（W_k^shared, W_v^shared），不与其他 memory 共享。MoM 实验（Table 6 ablation）证实 shared memory 对 performance 有显著增益：w/ shared memory → Recall avg 28.16, w/o shared memory → 26.06。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
