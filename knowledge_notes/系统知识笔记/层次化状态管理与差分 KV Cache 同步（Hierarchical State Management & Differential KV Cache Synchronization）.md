## 层次化状态管理与差分 KV Cache 同步（Hierarchical State Management & Differential KV Cache Synchronization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DynoPipe（§4.3）为快速边界迁移设计的 pipeline 状态管理机制。状态按迁移关键度与重建复杂度分三 tier：(1) critical frontier states——KV cache + attention 权重，保持生成连续性所必需；(2) intermediate activations——layer 输出，可选择性重算（bounded overhead）；(3) auxiliary metadata——norm 统计、位置编码，高空间局部性。差分 KV cache 同步：跨域迁移只传"差分"而非全量 KV（最坏 P99 迁移 72ms），配合参数重叠缓存（两侧保留重叠参数集，sub-ms GPU-GPU 传输）与带宽感知状态分区（利用下行富余带宽异步流式传输）。L1/L2/L3 三层预测性预置（predictive staging）：L1 缓存热边界配置 sub-ms 访问、L2 维护次级迁移候选、L3 按需 fallback。带宽<1Gbps 时 adaptive recomputation fallback：维护轻量 checkpoint 于战略层边界、选择性 forward 重算中间状态，计算 +15-25%、带宽 -90%，最坏迁移 <120ms。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
最坏场景（网络争用 2.5 Gbps + QPS=8）：P99 边界迁移 85ms = 差分 KV cache 传输 72ms + 参数重预置 13ms。单用户 vs 多用户 KV 管理统一处理：单用户下 KV cache 随上下文线性增长、整驻单一域（edge 或 cloud）；多用户并发下 per-request KV cache 竞争边缘 GPU 内存，可能触发 memory-pressure（>90%）触发更早边界移动（向云重配置降低边缘内存压力），层次化 tier 结构按 LRU 逐出冷 per-request cache、保留热 cache。对比 baseline 的"状态迁移需数十秒重建 cache"——预测性预置把冷启动惩罚转为预置，迁移从秒级降到毫秒级。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：状态分解（三 tier 分级）+ 预测性预置（按预测边界提前把参数放入 L1/L2/L3 内存层次）+ 差分同步（只传变化量）+ 重计算回退（低带宽下用算力换带宽）。使用场景：边云流水线边界迁移、KV cache 跨域连续性保持、服务中断容忍上限内的重配置；数值一致性：模型按 block 边界切分保证跨配置浮点一致（<10⁻⁶ 相对误差），LLaMA2-7B WikiText-103 perplexity 变化 <0.3%。

涉及论文标题：
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
