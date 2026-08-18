## 集合通信匹配器（Collective Communication Matcher，Pull-Push 通信匹配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- STAGE 的 Collective Communication Matcher 是自动识别并行策略所需集合通信的机制：分析 tensor 在各设备上的 producer 分布与 consumer 分布，把通信分解为两步——Pull（从 producer 分布重建完整 tensor）与 Push（把完整 tensor 分发到 consumer 分布），两者之间用虚拟头节点衔接。Pull：Duplicated=无需通信、Partition=Gather（拼接）、PartialSum=Reduce（求和）；Push：Duplicated=Broadcast、Partition=Scatter、PartialSum 一般不使用。组合即可匹配出 AllReduce/AllGather/ReduceScatter/AllToAll 及其复合，如 [B/dp,S,H@1/tp]→[B/dp,S,H/tp]=ReduceScatter、→[B,S/dp,H@1/tp]=AllToAll、→[B/dp,S,H]=AllReduce、→[B/tp,S,H/dp]=ReduceScatter+AllToAll、→[B,S,H]=AllReduce+AllGather（Table IV）。vault 证据：仅本论文（no note evidence for this term in knowledge_notes）；ASTRA-sim/集合通信相关背景见知识库_kernel调度.md 各集合通信条目。
- 从编译框架角度拆解：匹配器是"并行化编译"的通信推导引擎——把任意 producer/consumer 分布对映射为具体集合通信算子插入执行图。它让 STAGE 无需手工为每个并行策略编写通信模板，且能识别此前被忽略的、由任意张量分布组合产生的通信模式（如 ReduceScatter+AllToAll 复合），支持穷举探索并行配置空间。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 作为 Workload Distributor 的张量级分布组件实现：先传播图推断 tensor 形状，应用分布后重传播，对每个 mismatch 用 Pull+Push 规则表匹配集合原语；匹配结果作为通信节点（带类型与通信量）加入 DAG，与真实 NCCL 行为对齐验证（NCCL 的 AllToAll 分解为 Send/Recv 时 STAGE 也按 Send/Recv 分解对比，Table VII 通信误差 0.000%~2.980%）。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
