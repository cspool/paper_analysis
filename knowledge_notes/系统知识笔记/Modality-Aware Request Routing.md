## Modality-Aware Request Routing

术语是什么？

Modality-Aware Request Routing 是 ModServe 提出的按输入模态分别路由请求的策略。传统 LLM serving 使用 round-robin 或 memory-based load balancing（如 Llumnix），但这些方法对 LMM 无效——因为它们忽视 image encoding 的计算密度（Insight 2），在 image burst 时导致 load imbalance 和高 tail latency。Modality-aware routing 的核心是：(1) image-text 请求路由到 pending image token load 最少的 Image Instance；(2) 大请求（多图像）跨多个 Image Instance 并行编码；(3) text 请求路由到 pending token load 最少的 Text Instance，CroAttn 按 text tokens 计，DecOnly 按 total tokens 计。

从系统架构角度拆解术语：

Routing 决策流程：
```
请求到达 Router:
  if 请求包含 image:
    → 查询所有 Image Instance 的 pending image token count
    → 选择 count 最小的 k 个 Image Instances
    → k = ceil(total_images / max_images_per_instance)
    → images[i] → Image Instance[j]，并行 encoding
  if 请求含 text:
    → 等待 image tokens ready（如有）
    → CroAttn: 查询 Text Instance pending text token count（不计 image tokens）
    → DecOnly: 查询 Text Instance pending total token count（计 image tokens）
    → 选择 load 最小的 Text Instance
```

Insight 7 驱动的架构差异：CroAttn 的 LLM prefill 延迟随 image token 比例增加而减少（因仅有 4/40 layers 受 image tokens 影响），因此 Text Instance routing 仅需 load balance text tokens。DecOnly 中 image 和 text tokens 在 self-attention 中同等处理，因此 routing 必须计入 total tokens。

术语一般如何实现？如何使用？

ModServe 的 router 实现为 Pool Manager 的一部分（gRPC server）。与 Pull-based RDMA 机制协同：Image Instance 完成 encoding 后不立即 push——等所有 tokens ready 后由 Router 做出 fully informed 的 Text Instance 选择决策。Router 同时考虑 queue size、prefix caching 机会和 payload 大小。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
