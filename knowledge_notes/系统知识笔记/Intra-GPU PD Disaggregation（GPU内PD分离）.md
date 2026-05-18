## Intra-GPU PD Disaggregation（GPU内PD分离）

术语是什么？

Intra-GPU PD Disaggregation是一种介于Colocated（合设）和Disaggregated（分离式）两种经典架构之间的LLM服务架构。传统PD-Disaggregation将Prefill和Decode部署在不同GPU上，通过NCCL/RDMA传输KV Cache，但存在静态资源分割造成的利用率和KV Cache迁移开销问题。Intra-GPU PD Disaggregation在同一GPU内部通过SM空间分区实现PD分离，共享统一KV Cache池，避免跨GPU迁移。

从系统架构角度拆解术语：

架构对比：

1. **Colocated（合设）**：Prefill和Decode在同一GPU上用同一组SM顺序执行。问题：Prefill（计算密集）和Decode（内存密集）的GPU利用率特征冲突。
2. **Disaggregated（分离式）**：Prefill GPU和Decode GPU物理分离。问题：(a) 静态资源划分导致一侧闲置时另一侧资源不足；(b) KV Cache需跨GPU传输，增加延迟和带宽压力；(c) 计算冗余。
3. **Intra-GPU PD Disaggregation**：在单GPU内通过SM分区实现逻辑分离。优势：(a) KV Cache共享，无需迁移；(b) SM动态流动，无静态闲置；(c) 无跨GPU通信开销。

术语一般如何实现？如何使用？

通过NVIDIA GreenContext API创建两个CUDA Context，各自绑定不同的SM子集，实现单GPU内的资源隔离。在SGLang中已集成该方案（LMSYS PD-Multiplexing实现）。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
- Towards Resource-Efficient Serverless LLM Inference with SLINFER

**Serverless场景下的PD Disaggregation适配性**：SLINFER在serverless多模型场景下评估了PD disaggregation（prefill/decode分离为独立instance），发现该架构反而导致资源使用增加和serving capacity下降：在4×CPU+4×GPU上serving 64×7B模型时，disaggregated PD的SLO rate从99%降至70%（32 models）或从86%降至69%（128 models）。根本原因是serverless workload的请求稀疏且低频→prefill instance 93% lifetime处于cold-start或idle状态，与DistServe的结论一致——PD disaggregation不适合资源受限场景。

---
