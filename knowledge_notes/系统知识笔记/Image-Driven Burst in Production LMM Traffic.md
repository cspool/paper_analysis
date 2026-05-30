## Image-Driven Burst in Production LMM Traffic

术语是什么？

Image-Driven Burst 是 ModServe 论文在 Azure 生产 LMM 推理 trace 分析中发现的独特流量模式——image-text 请求出现显著的 bursty 行为，不仅因请求到达率增加，更因单个请求中的 image 数量增加（如 video workload）。关键特征：(1) image-text 的 prompt token rate 是 text-only 的 5×；(2) image-text 和 text-only 的 peak/trough 模式独立且低相关；(3) 现有 LLM traffic prediction 方法（基于 diurnal pattern）对 multimodal traffic 的预测误差高达 79%；(4) image-per-request 分布呈 heavy-tailed（power-law），不同 service 间差异大（video service 处理 16× 更多 images/request）。

从系统架构角度拆解术语：

Image-driven burst 的两个维度：
```
维度 1: 请求到达率 surge（类似常规 web burst，但仅发生在 image-text service）
维度 2: image-per-request surge（如 video understanding workload 的 frame 序列处理）
```

对 monolithic serving 的影响：
```
Image-driven burst 时:
  → Monolithic: 所有 instance 均受冲击（image encoding + LLM prefill 争用 GPU）
  → 扩容: 必须整体扩容 TP-N instance → LLM backend 也被扩容（浪费 GPU）
  → ModServe: 仅 scale out Image Instances → LLM backend 不受影响
  → 对 CroAttn 更有效: image tokens 仅影响 4/40 layers → LLM prefill 几乎不受 image burst 影响
```

术语一般如何实现？如何使用？

ModServe 使用 token-aware autoscaling（基于 image tokens/sec）应对 image-driven burst。此术语解释了为什么纯 text-centric LLM serving 优化（如 PD disaggregation）不足以应对 multimodal workloads——multimodal burst 的本质是 modality-specific 的，需要 modality-specific 的响应机制。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
