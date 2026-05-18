## Hierarchical Resource Graph (HRG)（层次化资源图）

术语是什么？通过联网搜索让回答具体和精准。

Hierarchical Resource Graph (HRG)是FlexPipe提出的三级资源协调数据结构，用于在碎片化serverless集群中管理多模型并行扩容时的资源争抢。HRG覆盖三个物理层次：(1) Server级——GPU memory和PCIe bandwidth；(2) Rack级——network bandwidth；(3) Cluster级——storage I/O。每个层次维护annotated paths，标注scaling event marker来识别争抢模式和跟踪资源依赖。HRG将多模型同时扩容从资源争抢问题转化为资源协调机会——通过预判bottleneck并将scaling操作智能分布到不同物理path上。

从系统架构角度拆解术语：

FlexPipe中HRG的运转流程：
1. **Graph构建**：根据集群物理拓扑（server→rack→cluster）和当前GPU分配状态构建HRG，每个节点标注可用GPU memory、PCIe剩余带宽、network剩余带宽、storage I/O负载。
2. **Scaling Event注册**：当某模型的Pipeline Expansion触发时，HRG在对应path上标注scaling event marker，记录占用资源和预计持续时间。
3. **Contention预测**：新scaling请求到达时，HRG扫描所有path上的active scaling event marker→识别哪些server/rack/cluster存在并发资源竞争。
4. **Contention-aware placement**：Scheduler选择无active scaling marker的path部署新instance，避免多模型scaling堆积到同一物理位置。
5. **与GPU allocation协同**：HRG与GPU分配优化（公式6-9）联动——objective max Σ(T_{ij}/m_j) − γ(CV_i)·I(Σx_{i'j}>1)，multiplexing penalty γ(CV_i)=γ_0·(1+α·CV_i²)根据workload burstiness动态调整。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) HRG为内存中有向无环图，节点代表resource domain，边代表resource dependency和capacity；(2) Scaling event marker带TTL自动过期，防止stale marker误判；(3) 路径选择为最短无争抢路径优先，减少跨rack/跨cluster通信；(4) 与affinity-based scheduling协同：优先选择host memory仍保留模型参数副本的server（公式13），结合temporal decay和GPU availability；(5) 论文在82-GPU K8s集群上验证，HRG协同使always-on GPU reservation从75%降至30%且无service quality degradation。

涉及论文标题：
- FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters
