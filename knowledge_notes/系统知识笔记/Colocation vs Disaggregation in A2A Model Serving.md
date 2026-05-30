## Colocation vs Disaggregation in A2A Model Serving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Colocation 和 Disaggregation 是 A2A 模型推理 Serving 中图级别的两种基本部署策略。Colocation 将多个 model component（如 encoder + LLM）合并到同一个 executor 中运行，共享 GPU 资源；Disaggregation 将不同 component 解耦到独立 executor（通常在不同 GPU 上），允许各自独立扩展。在 Cornfigurator 的 model graph 中，每条 colocatable edge 有 KEEP（disaggregation）或 MERGE（colocation）两种选择。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Tradeoff 分析：

```
Colocation 优势: 共享 GPU 资源, 减少 NCD 传输延迟 (~10ms), 无 GPU 碎片
Colocation 劣势: 无法独立扩展, slowest component bottlenecks all

Disaggregation 优势: 每 component 独立扩展, component-specific 硬件匹配
Disaggregation 劣势: GPU 碎片, NCD 传输开销

Cornfigurator 最优决策示例 (Qwen 3 Omni 16GPU 1/3 audio):
  1×(E_aud) + 4×(E_img+E_vid+L_th) + 11×(L_ta+G_aud)
  = audio encoder disaggregated (低吞吐独立扩展),
    其余 encoder+thinker colocated (共享资源),
    talker+vocoder colocated (audio output pipeline)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Disaggregation 并非普遍有利：Qwen-Image（2-component）中 monolithic 最优（LLM prefill 轻量，encoder GPU 会浪费）；InternVL 3 中当 image input 概率从 25%→75%，planner 自动从 monolithic→encoder-disaggregated→增大 batch size 过渡。Cornfigurator 不预设固定策略，将所有策略视为搜索空间中的点并系统化评估。

EPD-Serve 将 Colocation 扩展到 Encode-Prefill-Decode 三阶段的空间复用：逻辑层独立调度，物理层通过 Ascend NPU 上 AI Core（MatMul）和 AI Vector（AllReduce）的资源互补实现 operator-level 共置。部署拓扑符号 "-" = 分置不同硬件，"()" = 物理共置。支持 E-P-D / EP-D / ED-P / E-PD / (E-P)-D / (E-D)-P 等多种拓扑，按 SLO 优先级灵活切换（高性能平衡 / 快速首Token / 最大化吞吐）。实验证明 (E-P)-D 共置比 PD-disaggregated EP-D 提升吞吐 57.37-69.48%，(E-D)-P 共置的 Encode(memory-heavy) + Decode(compute-heavy) 资源互补大幅优化 TTFT。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
