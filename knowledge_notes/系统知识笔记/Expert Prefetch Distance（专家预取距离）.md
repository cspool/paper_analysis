## Expert Prefetch Distance（专家预取距离）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Prefetch Distance 指在 MoE expert offloading 中，从当前正在计算的 layer l_now 提前 d 层发出 expert prefetch 指令，目标是使 l_now + d 层的 expert weights 在该层开始 forward 时已经到达 GPU memory。类似于 CPU memory prefetching 中的 prefetch distance 概念。理想 d 应恰好使 prefetch overhead（prediction + CPU-to-GPU PCIe transfer）与前 d 层 computation time 完全重叠。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
Prefetch Distance 对 FineMoE 双搜索策略的影响：

d 的决定因素：
  1. Hardware: PCIe bandwidth (32GB/s on 3090) + per-expert weight size
  2. Computation: 每层的 attention + gate + expert computation time
  3. Search method: 
     - Layers [1, d]: 仅 semantic-based search (无 trajectory history)
     - Layers [d+1, L]: trajectory-based search (前 d 层已积累 trajectory)

d 的调优 trade-off：
  d 太小 → prefetch 来不及完成 → expert miss 增加 + on-demand loading latency
  d 太大 → 初始层只能用 semantic (无 trajectory) → hit rate 下降 (图 4)
  FineMoE per-model profiled optimal d：
    Mixtral-8×7B: d=3   (32 layers, 8 experts/layer)
    Qwen1.5-MoE:  d=6   (24 layers, 60 experts/layer) — 更多 experts 需要更大 d 来保障覆盖
    Phi-3.5-MoE:  d=4   (32 layers, 16 experts/layer)
```

语义搜索弥补了大 d 的缺陷：初始 d 层虽然没有 trajectory history，但 semantic embedding 仍能提供有效的 expert map 检索（Pearson correlation 验证 semantic similarity 与 expert hit rate 正相关）。大 d 对 Qwen1.5-MoE（60 experts/layer，搜索空间大）尤为关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 FineMoE 中 d 通过 profiling 确定：测量不同 d 值下 TTFT 和 TPOT（图 15），选择使 inference latency 最小的 d。MoE-Infinity 固定 d=0（synchronous prefetch），ProMoE 和 Mixtral-Offloading 用较小 d（受限于 trajectory-based prediction 的有效范围）。FineMoE 的双搜索策略使 d 可以更大，从而更好隐藏 PCIe transfer latency。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
