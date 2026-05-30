## Prefill-Guided Expert Placement (预填充引导的专家放置)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill-Guided Expert Placement 是利用 MoE 模型 prefill 阶段收集的 expert selection 信息，在 decode 阶段开始前预测并优化 expert 在各 GPU 上的放置布局的策略。其核心依据是论文发现的 Insight 1（Ob3）：prefill 和 decode 阶段的 expert selection 模式高度相似——cross-layer/cross-token heatmap 的 Spearman's ρ ≥ 0.7，top-5 prefill experts 覆盖 ~60% 的 top-5 decode experts（top-20 覆盖 ~90%）。传统 expert placement 方法（如 EPLB）依赖周期性收集的 decode 阶段 profiling data（每 3000+ steps 触发调整），在初始 ~1000 decode tokens 期间没有 profiling data 可用，无法优化 placement，对于短输出请求（fewer than 3000 tokens）整个生命周期都无法触发调整。

从系统架构角度拆解术语：
论文提出两种 placement 算法（Algorithm 2），均在 serving 初始化时执行：

**Remap-based Placement**（保持 expert 总数不变，重新分配）：
```
Input: prefill traces D, GPU count G
Output: per-layer expert-to-GPU assignment {S_q}

for each layer l:
    从 prefill traces 计算每个 expert e 的频率 f_{l,e}
    Sort experts by decreasing roofline cost(f_{l,e})
    初始化所有 GPU 负载 L_g = 0
    for each expert e in sorted order:
        找到负载最小的 GPU g* 且 |S_{g*}| < E/G
        分配 expert e 到 GPU g*
        L_{g*} += cost(f_{l,e})
```

**Duplication-based Placement**（额外槽位，复制热门 expert）：
```
Input: prefill traces D, GPU count G, extra slots per GPU R
Output: per-layer expert-to-GPU assignment (含复制)

for each layer l:
    计算频率 f_{l,e}，生成默认连续布局 S_q (experts 0-15 on GPU 0, etc.)
    每 GPU 预留 R 个额外槽位: r_g = R, total = E + R*G experts per layer
    计算初始 L_g = Σ cost(f_{l,e}) for e in S_g
    for i = 1 to R*G:  # 每次加入一个复制
        (e*, g*) = argmin_{e,g: r_g>0, g not in hosts(e)} δ_{e,g}
        # δ_{e,g} = max_g' L_g' 的变化（减少 bottleneck load 最大的变体）
        将 expert e* 复制到 GPU g*; r_{g*} -= 1; 更新 L_g
    # 复制 expert 的 tokens 均匀分配到所有副本
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现于 SGLang serving framework：(1) 通过 `init_expert_location` 接口设置 custom expert placement；(2) 使用 DeepEP 作为 MoE backend，`ep_dispatch_algorithm="dynamic"` 使 tokens 均匀分配到复制 expert 的各副本；(3) 通过 inserted CUDA event timers 独立测量每个 GPU 上各操作的时间。
- 在 Qwen3-235B / 8×H100 上的实验：Remap 和 Dup 分别实现 15.5% 和 12.5% 的 MoE 计算加速（vs default contiguous placement），均在 oracle Best 的 10% 以内。在 EP8 下（每 GPU 16 experts），默认布局的 max/min execution-time ratio 仅 ~1.3×（较小），预期在更大 EP scale 下效果更显著。
- 开源：https://github.com/zhongkaiyu/moe_exp_placement, DOI: 10.5281/zenodo.19617695。
- 特别适用场景：(1) PD-disaggregated serving（prefill 和 decode 在不同机器上执行，prefill 机可将 traces 传给 decode 机）；(2) 短输出请求（<3000 tokens, EPLB 无法积累足够 profiling data）；(3) 大规模 EP（EP16+, 负载不均更显著）。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---
