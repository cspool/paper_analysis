## 张量分布语义（Duplicated / Partition / PartialSum）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- STAGE 定义三种符号张量分布语义：(1) Duplicated——tensor 在所有设备上有完整副本；(2) Partition——tensor 沿某维在各设备间不相交分片；(3) PartialSum——每设备持有部分和结果，需归约。三种语义可组合表示复杂并行策略，如 x[B/dp, S/sp, H@1/tp] 表示 batch 维 dp 分片、seq 维 sp 分片、hidden 维为 tp 的 PartialSum。Table III 用符号记法枚举线性层在各策略下的张量表示：DP 切 batch（x[B/dp,H]）、TP Row 切输入维（x[B,H/tp@1]）、TP Column 切输出维（w[H,4H/tp]）、FSDP 切 batch+权重（x[B/fsdp,H],w[H/fsdp,4H]）、Hybrid-Parallel 等。vault 证据：paper_secs 本论文 IV.-STAGE 章节（score 4451）；knowledge_notes 中无独立条目（no note evidence，但知识库_kernel调度.md 已有 PartialSum 相关条目）。
- 从编译框架角度拆解：分布语义是通信匹配的输入——编译器先传播计算图推断每个 tensor 的形状与分布，再逐算子应用目标并行策略、重传播形状，从而暴露 producer 与 consumer 之间的"分布 mismatch"（如 producer 输出 [a,c@1/tp] 而 consumer 期望 [a,c]），据此推导需要的集合通信（该例为 AllReduce 聚合 partial sum）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 STAGE 中由 Workload Distributor 实现：用户指定并行度（--dp/--tp/--pp/--sp/--ep），distributor 把初始无分布计算图按策略标上分布符号并传播；张量分布与"通信匹配器"联动，自动为每个 mismatch 插入对应集合通信。PartialSum 语义还能建模框架中不存在的假设策略（如 Fully-Sharded Tensor Parallel FSTP：X[Batch/dp,D1/tp] 先 AllGather 再 einsum 得 Y*[Batch/dp,D2@1/tp] 再 ReduceScatter）。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
