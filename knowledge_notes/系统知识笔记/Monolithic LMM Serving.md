## Monolithic LMM Serving

术语是什么？

Monolithic LMM Serving 是当前主流 LMM serving 框架（vLLM、DeepSpeed、Transformers）的默认部署模式——将 image preprocessor、image encoder 和 LLM backend（prefill + decode）打包为单个 serving instance。所有组件 co-located 在同一硬件 server，使用统一的 tensor parallelism 和 batch size 配置。虽然 PD disaggregation 可在此 monolithic setup 内应用（分离 prefill/decode phases），但 multimodal components（image preprocessor/encoder）仍与 prefill instances 耦合。

从系统架构角度拆解术语：

Monolithic 执行的三个低效来源（ModServe 论文表征）：
1. 统一 TP 配置的低效：Llama3.2-11B 的 630M encoder 在 TP-8 时比 TP-1 更慢（inter-GPU communication > compute savings），但 LLM backend 需 TP-4/8——monolithic 被迫选择妥协 TP
2. 统一 batch size 的低效：image encoding 是 compute-bound（batch>1 无 throughput gain），decode 是 memory-bound（batch 越大 throughput 越高）——monolithic 被迫使用同一 batch
3. 不可独立扩容：image burst 时只能整体扩容（含不必要的 LLM backend 扩容），导致过度 provisioning

ModServe 图 1 展示了 monolithic 的问题：当请求 image 数量从 1 增至 16，TTFT 急剧退化——因 image encoding 成为 bottleneck 且无法独立 scale out。

术语一般如何实现？如何使用？

在 vLLM 中启用 LMM 支持时，vLLM 加载完整模型（encoder + connector + LLM）为单个 TP group。所有请求进入统一队列，FIFO scheduling。这是 ModServe 的 baseline——ModServe 在此基础上约 5000 行 Python 实现 stage decoupling。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
