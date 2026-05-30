## DLRM (Deep Learning Recommendation Model)

术语是什么？
DLRM（Deep Learning Recommendation Model）是Meta提出的推荐系统基础架构，将推荐任务分解为sparse feature处理（通过embedding tables将categorical inputs映射到dense vectors）和dense feature处理（通过MLP处理continuous features），两者在interaction layer中通过feature crossing（如dot product/Factorization Machine）结合后送入final MLP输出prediction。DLRM是Meta ads ranking的生产基础架构，服务每天超过数百trillion次推理。

从算法pipeline角度拆解术语：
传统DLRM pipeline：
```
Input Features:
  - Sparse features: categorical IDs (post ID, page ID) --[embedding lookup]--> dense vectors
  - Dense features: continuous values (age, CTR, engagement) --[MLP]--> dense vectors
  
Interaction Layer:
  - Pairwise dot product between all feature pairs: XX^T
  - Factorization Machine (FM): low-rank approximation (X · (X^T Y)) reducing O(N²D) → O(NKD)
  
Output: Click-through rate / engagement prediction
```

近年来DLRM architecture evolution带来了新的computational patterns：
- **HSTU (Hierarchical Sequential Transduction Unit)**: 将user history作为jagged tensor序列处理，引入Transformer-like attention机制
- **InterFormer**: bidirectional information flow between non-sequential (user demographics) and sequential (browsing history) features，通过Personalized FeedForward Network (PFFN)
- **WuKong**: 引入Optimized FM——通过learnable projection matrix Y降低interaction complexity
- **Generative Recommendation (OneRec)**: 将推荐建模为sequence generation，使用RQ-VAE/RQ-Kmean将continuous embeddings量化为discrete semantic IDs供LLM处理

这些新架构引入10-100× per-request complexity increase vs传统DLRM，并需要specialized attention kernels、jagged tensor operations和quantization primitives。

术语一般如何实现？如何使用？
Meta生产环境使用多阶段ranking pipeline：Retrieval (millions→10K-100K candidates, low-complexity models) → Early-stage ranking (thousands→hundreds candidates, moderate complexity) → Late-stage ranking (hundreds→final ranking, heavyweight models up to 2 GFLOPS/sample)。每个阶段有不同的kernel优化需求：retrieval优先throughput at large batch sizes；late-stage ranking需要compute-intensive fused interactions under sub-100ms latency；sequence models需要jagged tensor operations。KernelEvolve通过在异构硬件上自动生成优化kernel来服务这些diverse requirements。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
