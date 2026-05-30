## Flexible Stage Disaggregation with Physical Co-location

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Flexible Stage Disaggregation with Physical Co-location 是 EPD-Serve 的部署拓扑策略。在逻辑层面，Encode/Prefill/Decode 三阶段被拆分为独立实例进程，各自拥有独立调度和弹性伸缩能力。在物理层面，不同阶段可共享相同的 Ascend NPU 硬件资源（共置），通过 operator-level 的空间复用（如 MatMul 用 AI Core、AllReduce 用 AI Vector 交替执行）提升硬件利用率。这一策略支持多种部署拓扑的动态切换：E-P-D（全解耦）、EP-D（Encode+Prefill 共置+Decode 独立）、ED-P（Encode+Decode 共置+Prefill 独立）、E-PD（Encode 独立+Prefill+Decode 共置）、(E-P)-D、(E-D)-P、(E-PD) 等。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

不同部署拓扑的性能特征与推荐场景（EPD-Serve Figure 17 & Section 4.7）：

```
场景分类:
  1) 高性能平衡 (Low TTFT + Low TPOT):
     推荐: (E-P)-D
     - Encode+Prefill 共置 1 NPU, Decode 独立 1 NPU
     - 高并发 12 req/s: TPOT 降低 79.99-93.31% vs TP1
     - SLO attainment = 84.96% (strict: TTFT<800ms, TPOT<30ms)

  2) 快速首Token (Low TTFT, 可接受 moderate TPOT):
     推荐: (E-D)-P
     - Encode+Decode 共置 1 NPU, Prefill 独立 1 NPU
     - TTFT 降低 39.22-54.56% vs EP-D
     - Encode(compute-heavy) + Decode(memory-heavy) 资源互补

  3) 最大化吞吐 (放松 TTFT/TPOT 约束):
     推荐: (E-PD)
     - Encode 与 Prefill+Decode 共置 1 NPU
     - 吞吐提升 12.87-14.88% vs monolithic TP1
     - 适合高负载多用户或 RL post-training inference

  4) 严格 SLO 最高达成率:
     推荐: E-P-D (3 NPU 全解耦)
     - SLO attainment = 94.34% (10 req/s, TTFT≤2000ms, TPOT≤50ms)
     - per-NPU throughput = 192.70 (7.95× EP-D)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

物理共置的关键实现：通过 Ascend NPU 上不同算子的硬件资源需求互补实现算子级并行。Figure 6 展示了 MatMul（AI Core compute-heavy）和 AllReduce（AI Vector communication-heavy）在共置 NPU 上的交替执行——当一个阶段等待通信（如 P-D 传输）时，另一阶段利用空闲计算单元执行算子，减少 NPU idle 时间。部署拓扑的选择由 Proxy 在运行时根据 workload 特征和 SLO 目标决定。在 EPD-Serve evaluation 中，(E-P)-D 在 12 req/s 高并发下比 PD-disaggregated EP-D 提升吞吐 57.37-69.48%。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
