## Interleaved Cluster Execution（ICE，交错集群执行）与动态负载迁移

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MERIDIAN（ISCA'26）为去中心化 RAG 推理设计的调度机制，解决文档注意力分解引入的集群间时序失衡：DAC（文档注意力集群）处理大量文档 KV、CEC（上下文执行集群）只处理少量 query/生成 token KV，attention 阶段 CEC 先完成闲置；而 FFN 等上下文重阶段 DAC 闲置。ICE 动态在本应空闲的集群上启动后续 batch：tensor 并行下交替把 batch 分给 DAC/CEC 使两集群并发推进；pipeline 并行下允许 DAC/CEC 在同一 stage 内处理不同 micro-batch 实现 intra-stage overlap。残余失衡（DAC 提前完成文档注意力）由**动态负载迁移**补足：初始化时把部分 CEC 参数静态复制到 DAC，DAC 空闲时协助上下文计算——因 DAC/CEC 微架构同构、迁移开销可摊销到全部推理请求。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ICE（tensor 并行，B 个 batch）：
for batch in 1..B:
    if batch 属文档注意力阶段: 派发到 DAC（doc KV 就地算）
    else:                      派发到 CEC（QKV/FFN/融合）
    # 关键：DAC 算完本 batch 文档注意力后不等待 CEC，而是预取并启动下一 batch 的文档注意力
    #       CEC 同理在空档启动其可独立部分 → 两集群交错推进、减少 idle bubble
# pipeline 并行（2 stage、micro-batch）：
#   stage 内 DAC 处理 micro-batch i 的文档注意力，同时 CEC 处理 micro-batch i-1 的上下文
# 动态负载迁移：DAC 空闲时执行复制来的 CEC 参数（FC/FFN 部分），DAC 满载时 CEC 自己算
```
对比集中式注意力的调度（NeuPIMs/HeterRAG 的直接复用）：它们无 DAC/CEC 时序失衡（注意力集中在一处），直接套用会导致集群利用率低下——ICE 是针对去中心化数据流的调度新机制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：host 侧统一调度器实现——初始化静态分配（tensor/pipeline/hybrid 并行策略，CEC 的 FC 跨设备分片、DAC 按 head 分配文档 KV 避免广播），运行时按设备负载动态下发推理任务；ICE 与"单 batch 计算通信重叠"（如 MoE 的 SBO 交错调度）同类思想，但作用对象是 PIM 集群而非通信 kernel。效果（组件消融，图 16）：M-pim 2.19× → M-ad（分解）2.12× → M-ad+ire（+ICE）再 +1.27×；扩展性：32 设备 pipeline 并行 4.19× vs tensor 并行 3.68×（pipeline 只传轻量激活、tensor 需同步部分和）。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
