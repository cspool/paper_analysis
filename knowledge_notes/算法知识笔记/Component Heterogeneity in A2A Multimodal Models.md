## Component Heterogeneity in A2A Multimodal Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Component Heterogeneity（组件异构性）在 A2A 模型中表现为两个维度：(1) Request Type Heterogeneity——不同 request type（不同输入/输出模态组合）遍历模型 component graph 的不同子图，导致每个 component 面临不同的 request rate；(2) Computational Heterogeneity——不同 component 有巨大差异的资源需求和计算特性。Cornfigurator 论文的 Table 2 量化了这一点：Qwen 3 Omni 在 A100-80GB 上，audio encoder 的吞吐是 21.43 req/s 而 vocoder 仅 0.12 req/s（178× 差异），thinker LLM 2.15 req/s vs talker LLM 0.12 req/s（18× 差异）。两种异构性叠加导致各 component 负载极度不均衡，使固定部署策略在不同 workload 下性能差异显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Component heterogeneity 对 Serving 的影响（Qwen 3 Omni 例子）：

```
给定 workload: π_text=2/3, π_audio=1/3
  各 request type 的 component 调用:
    text-output types (①-④): 需要 E_img/E_vid/E_aud + L_th
    audio-output types (⑤-⑧): 需要 E_img/E_vid/E_aud + L_th + L_ta + G_aud

Per-component request rate:
  E_img: 100% of requests (所有 type 都含 image input)
  L_th:  100% of requests (所有 type 都经过 thinker)
  L_ta:  33% of requests (仅 audio-output types)
  G_aud: 33% of requests (仅 audio-output types)

瓶颈分析 (假设各 component 独立部署):
  L_th 吞吐 = 2.15 req/s  →  100% load → 需要 1/2.15 ≈ 0.47 GPU-seconds/req
  L_ta 吞吐 = 0.12 req/s  →   33% load → 需要 0.33/0.12 ≈ 2.75 GPU-seconds/req
  G_aud 吞吐 = 0.12 req/s →   33% load → 需要 0.33/0.12 ≈ 2.75 GPU-seconds/req

→ L_ta 和 G_aud 是瓶颈 (每 req 消耗的 GPU 资源远多于 L_th)
→ 最优部署: 大量 GPU 分配给 L_ta+G_aud, 少量给 L_th+encoders
  Cornfigurator 16GPU plan: 4×(E_img+E_vid+L_th) + 11×(L_ta+G_aud)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Component heterogeneity 是 Cornfigurator 规划的核心动机。传统方法（monolithic 或 fixed disaggregation）无法自动适应 heterogeneity——monolithic 使 slowest component 成为全部模型的瓶颈；fixed disaggregation 可能将低负载 component（如图像 encoder 在 audio-heavy workload 下）分配到过多 GPU 导致资源浪费。Cornfigurator 通过 per-request-type reasoning 和计划枚举自动找到匹配 heterogeneity pattern 的最优 colocation/disaggregation 组合。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
