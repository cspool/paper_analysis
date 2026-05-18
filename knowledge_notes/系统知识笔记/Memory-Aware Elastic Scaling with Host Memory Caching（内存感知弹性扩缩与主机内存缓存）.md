## Memory-Aware Elastic Scaling with Host Memory Caching（内存感知弹性扩缩与主机内存缓存）

术语是什么？通过联网搜索让回答具体和精准。

Memory-Aware Elastic Scaling是FlexPipe提出的serverless环境下LLM推理实例弹性扩缩容策略，利用host memory作为中间层参数缓存避免cold start。在serverless环境中，scaled-down实例的GPU资源立即被回收分配其他workload，后续scale-up必须从持久存储重新加载参数，产生多秒级cold start延迟。FlexPipe在host memory（256GB+ per server）中保留已evict的GPU参数副本，创建survive instance termination的middle-tier cache，使后续扩容能从host memory（PCIe ~32GB/s）而非持久存储（~2-4GB/s）加载参数。配合affinity-based scheduling将cold start转化为warm start。

从系统架构角度拆解术语：

FlexPipe中Memory-Aware Elastic Scaling的运转流程：
1. **Scale-down**：当CV下降触发Stage Consolidation，在释放GPU前将stage参数transfer到host memory。
2. **Host Memory Cache**：LRU管理——满时evict最久未使用模型参数。单server的256GB+ host memory可缓存多模型stage参数。
3. **Scale-up**：CV升高触发Pipeline Expansion→需新GPU instance。
4. **Affinity-based Selection**：s* = argmax [w_t·e^{−λ(t_now−t_s)} + w_g·|g_s∩G_avail|] ——temporal decay优先最近承载过该模型的server（host memory cache大概率仍warm），GPU availability确保有空闲GPU。
5. **Fast Loading**：若host memory存在所需参数→PCIe直接拷贝到GPU；若无→fallback持久存储加载。
6. **效果**：论文生产case study显示instance initialization latency平均降72%，resource allocation wait time降85%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) Best-effort cache——不保证命中，但affinity scheduling最大化命中率；(2) Stage-level缓存粒度——fine-grained stage使单stage参数远小于完整模型，host memory可同时缓存多stage；(3) 保持与GPU memory相同layout实现zero-copy PCIe transfer；(4) LRU eviction以stage为单位记录最后访问时间戳；(5) 与K8s node affinity结合——对经常承载同一模型的server设soft affinity但不禁止其他server使用；(6) 在production cluster case study中，该机制将always-on GPU从75%降至30%。

涉及论文标题：
- FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters


---
