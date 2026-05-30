## Asymmetric Expert Assignment (Asym-EA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Asymmetric Expert Assignment (Asym-EA) 是 HeterMoE 中消除 Zebra Parallelism 流水线气泡的细粒度负载均衡机制。当 expert GPU 计算慢于 attention GPU（常见于短序列），attention GPU 产生 idle bubbles。Asym-EA 将部分 expert 计算迁回 (offload) 到 attention GPU 以 balance 计算时间。

核心算法为 "gather and squeeze"（Algorithm 1）：accumulate 跨多层的 bubble（T_gather = T_E^Exp - T_A^Attn，每 microbatch 每层 expert GPU 比 attention GPU 多花的时间），直到累积量 ≥ T_squeeze（offload 一个最小 chunk 可消除的 bubble），然后在 accumulation 最多的层 squeeze。最小 offload chunk: n_1 = max(1, N/M) 个 experts per attention GPU 获得，n_2 = n_1·M/N 个 experts per expert GPU 被 offload。考虑 memory 约束：α 系数 enforce 上限 n_max（attention GPU 内存），β 系数 enforce 下限 n_min（expert GPU 内存），α 和 β 至多一个激活。

效率：Asym-EA 在 4K 序列上提供 1.14-1.20× 额外加速，在 >20K-28K 序列上不再需要（T_A^Attn ≥ T_E^Exp）。

从kernel调度角度拆解术语：

```
Algorithm 1: Gather and Squeeze
Input: n (experts), L (layers), M, N (GPU ratio)
       T_A^Attn, T_E^Attn, T_E^Exp (profiled per-microbatch times)
Output: O = {o_1,...,o_L} (experts to offload per layer)

n_1 ← max(1, N/M)                          // per-attn-GPU min acquire
n_2 ← n_1 · M/N                            // per-exp-GPU min offload
T_gather ← T_E^Exp - T_A^Attn              // bubble per layer
T_squeeze ← T_E^Exp·N/n·n_1 + T_E^Attn·N/n·n_2

α = min(⌊n_max/n_2⌋·T_squeeze/(L·T_gather), 1)  // memory upper bound
β = max(⌈n_min/n_2⌉·T_squeeze/(L·T_gather), 1)  // memory lower bound

t_bubble ← 0
for l ← 1 to L:
    t_bubble += α·β·T_gather
    if t_bubble ≥ T_squeeze:
        o_l ← ⌊t_bubble/T_squeeze⌋ · n_2
        t_bubble -= o_l/n_2 · T_squeeze

// 可整除要求: M | N 或 N | M（与 EP 中 GPU 数须整除 expert 数类似）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Asym-EA optimizer 依赖 Profiler 提供的 T_A^Attn, T_E^Attn, T_E^Exp 和 n_min, n_max
- Profiler 在每个 setup 上运行一次
- Offload 后 attention GPU 先完成所有 microbatch attention，再计算被 offload 的 experts
- 选择性逐层 offload——不同层 offload 不同数量的 experts，避免简单统一 offload 导致气泡转移到 expert GPU
- 仅在可整除的 GPU 比例下有效（如 4:2, 4:4, 4:8），其他比例（如 4:3）无法使用 Asym-EA
- 使用 profiled forward 时间优化，backward 时间成比例减少

涉及论文标题：
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs
