## Gather and Squeeze Optimization Algorithm (HeterMoE Asym-EA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gather and Squeeze 是 HeterMoE 中 Asymmetric Expert Assignment 的 offline 优化算法（Algorithm 1），用于确定哪些 MoE layer 需要从 expert GPU offload 多少 experts 到 attention GPU，以最小化 Zebra Parallelism 流水线气泡。将气泡累积和消除形式化为贪心过程：跨多层累积 bubble（T_gather = T_E^Exp - T_A^Attn），直到足够消除至少一个 offload chunk（T_squeeze），然后在该层 squeeze。

从编译框架角度拆解术语：

```
Algorithm 1: Gather and Squeeze
n_1 ← max(1, N/M);  n_2 ← n_1·M/N
T_gather ← T_E^Exp - T_A^Attn
T_squeeze ← T_E^Exp·N/n·n_1 + T_E^Attn·N/n·n_2
α = min(⌊n_max/n_2⌋·T_squeeze/(L·T_gather), 1)
β = max(⌈n_min/n_2⌉·T_squeeze/(L·T_gather), 1)

for l=1..L:
  t_bubble += α·β·T_gather
  if t_bubble ≥ T_squeeze:
    o_l = ⌊t_bubble/T_squeeze⌋·n_2
    t_bubble -= o_l/n_2·T_squeeze
```

关键洞察：与其每层均匀 offload 少量 experts（导致气泡转移到 expert GPU），不如累积多层的气泡后集中在一层 squeeze。选择性逐层 offload 不同数量 experts。仅在 M|N 或 N|M 时有效。在 4K 序列上提供 1.14-1.20× 加速，>28K 时不再需要。

术语一般如何实现？如何使用？

- offline optimization，训练前执行一次
- 输入来自 HeterMoE Profiler（T_A^Attn, T_E^Attn, T_E^Exp, n_min, n_max）
- 输出 per-layer offload count 直接配置 ZP engine
- 使用 profiled forward 时间，backward 成比例减少
- Profiler 每个 setup 运行一次，仅需一个 attention GPU 和一个 expert GPU

涉及论文标题：
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs
