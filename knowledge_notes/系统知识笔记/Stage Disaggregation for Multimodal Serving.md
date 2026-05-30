## Stage Disaggregation for Multimodal Serving

术语是什么？

Stage Disaggregation for Multimodal Serving 是将 LMM 推理 pipeline 的异构阶段（image preprocessing、image encoding、LLM prefill、LLM decode）物理分离到独立资源池的技术。与传统的 monolithic LMM serving（所有阶段打包为单个 instance）和 text-centric PD disaggregation（仅分离 prefill/decode）不同，multimodal stage disaggregation 的核心是在 image-specific stages 和 text-specific stages 之间划分解耦边界。ModServe 论文将 pipeline 拆分为 Image Instances（CPU image preprocessing + GPU image encoding）和 Text Instances（LLM prefill + decode），每个池独立管理 TP 度、batch size 和 autoscaling 策略。

从系统架构角度拆解术语：

ModServe 的 stage disaggregation 运转流程：
```
请求到达 → Modality-Aware Router:
  ├─ Text-only request → 直接路由到 Text Pool Manager
  └─ Image-text request → Image Pool Manager 分配 Image Instance

Image Instance (独立资源池):
  CPU: numactl-pinned image preprocessing
  GPU (TP-1): ViT encoder forward → image tokens
  → 完成信号 + RDMA 地址 → Image Pool Manager

Image Pool Manager:
  独立 autoscaling: N_i = ⌈image_tokens_per_sec / per_instance_cap⌉
  独立 TP 配置: TP-1 (encoder 仅 630M-6B)
  独立 batch: 小或无 batching (compute-bound)

Text Pool Manager:
  独立 autoscaling: N_t（基于 prompt tokens/sec）
  独立 TP 配置: TP-4/8 (LLM backend 7B-72B)
  独立 batch: continuous batching via PagedAttention

Pull-based Transfer:
  Image Instance → 注册 RDMA buffer → Text Instance pull via GPU Direct RDMA
```

关键设计决策：(1) 解耦边界选在 image encoding 之后——因 image preprocessing 和 encoding 均为 compute-bound，而 LLM prefill/decode 混合 compute/memory-bound；(2) Connector 共置于 Text Instance——因 connector 极轻量(<0.4% TTFT)，独立 GPU 会严重 underutilize；(3) 允许同 server 内 colocation——如 1× TP-4 Text + 2× TP-2 Image on 8-GPU server，但配置独立。

术语一般如何实现？如何使用？

ModServe 基于 vLLM v0.7.2（Text Instance）+ HuggingFace Transformers（Image Instance），约 5000 行 Python。Pool Manager 为轻量 gRPC server（dedicated VM）。通过 heartbeat-based membership management 做 failure detection。可组合 PD disaggregation——ModServe + PD disaggregation = full EPD (Encoder-Prefill-Decode) disaggregation，额外提供最高 2.8× average TTFT reduction（图 19）。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

vLLM-Omni 将 stage disaggregation 扩展为更通用的 **fully disaggregated serving** 范式：不仅分离 encoder/prefill/decode，而是将 any-to-any 模型分解为任意数量的独立 stages（AR LLM stages、DiT stages、CNN modules），每个 stage 由独立的 execution engine 服务。以 Qwen3-Omni 为例，Thinker (30B AR LLM)、Talker (smaller AR LLM)、Vocoder (DiT/CNN) 三阶段各自独立运行在 vLLM engine 或 diffusion engine 上，Orchestrator 管理跨 stage 的请求路由。Per-stage 可独立配置 TP、memory budget 和 parallelism——Thinker TP-2 across devices，Talker 独立 device，Vocoder 独立 device。数据通过 Unified Connector（shared memory 或 Mooncake RDMA）在 stage 间传输。

与 ModServe 的 Image/Text 两池模型相比，vLLM-Omni 的 full disaggregation 可表达更复杂的 pipeline topology（如 Thinker→Talker→Vocoder 的 3-stage chain，或 AR→DiT 的 2-stage chain），且通过 stage graph 抽象（graph frontend）支持任意 DAG topology 的 pipeline 定义。
