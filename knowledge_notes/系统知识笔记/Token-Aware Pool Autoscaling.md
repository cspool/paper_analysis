## Token-Aware Pool Autoscaling

术语是什么？

Token-Aware Pool Autoscaling 是 ModServe 提出的基于 token throughput（而非请求速率）的动态资源扩缩容策略。传统 web service autoscaling 基于请求到达率（QPS），但 LMM 推理中请求的 token 数量差异极大（heavy-tailed distribution，α=4.4 for image-text, 2.9 for text-only），相同 QPS 下 token 负载可差数个数量级。Token-aware autoscaling 计算每个 stage 的 modality-specific load（image tokens/sec for Image Instances, prompt tokens/sec for Text Instances），基于 offline profiling 的 per-instance maximum capacity 决定所需 instance 数。

从系统架构角度拆解术语：

Autoscaling 决策公式：
```
N_i = ⌈ML_i / MC_i⌉

ML_i = modality-specific load of stage i
  Image Instances: image_tokens_per_sec（基于 image dimensions → tokens 的静态映射预计算）
  Text Instances: prompt_tokens_per_sec（CroAttn: text tokens only; DecOnly: total tokens）
MC_i = per-instance maximum capacity（来自 offline LMM profile，不违反 SLO 的最大吞吐）
```

Autoscaling 循环（每 5 分钟）：
```
Pool Manager 接收实时 load metrics:
  → 计算所需 instance 数
  → 若不足: warm-start 新 instance（cached model profile）
  → 若过剩: gracefully drain + 关闭
  → Hysteresis prevention: 避免 transient workload fluctuation 触发频繁 scaling
```

ModServe Insight 6 的应用：image-text 和 text-only traffic 的 burst pattern 独立且不可互预测（LLM traffic prediction 方法对此误差 79%）。Token-aware autoscaling 在 image-driven burst 时仅 scale out Image Instances——对 CroAttn 模型尤其高效（image tokens 仅影响 4/40 layers）。这是 ModServe 实现 41.3% cost saving 的核心机制。

术语一般如何实现？如何使用？

Image Pool Manager 和 Text Pool Manager 各维护独立 autoscaling loop。初始 instance 数 = median_QPS × median_latency（from historical traces）。无 history 时默认 overprovision。论文评估使用 SLO-driven heuristic（SLO attainment 低于 99% threshold 时触发扩容）。Offline profile（Section 4.1）为 autoscaling 提供 lookup table：f(TP, batch_size, load) → latency。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
