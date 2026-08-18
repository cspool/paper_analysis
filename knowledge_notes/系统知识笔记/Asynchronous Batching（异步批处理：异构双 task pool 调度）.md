## Asynchronous Batching（异步批处理：异构双 task pool 调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HybridSpec 面向异构架构（XPU + HB 栈）的连续批处理扩展：每请求分解为 prefill/decode/verification 三类任务，每个处理单元维护自己的 task pool（存任务类型、上下文状态、token 数等元数据）；单元空闲时立即从池中取可用任务组批执行，忙碌时推迟到当前迭代结束——两个特征：(1) 异步执行（各单元迭代起止不必对齐）、(2) 动态批组成（批内请求集合逐迭代变化，区别于离线静态批）。XPU 任务来源：外部新请求的 prefill、HB 栈回传的 verification；HB 栈任务来源：XPU 派发的下一轮 decode、内部续 decode（未达 draft budget）。内存约束用 watermark 机制缓解：内存利用率超预设阈值时挂起 prefill 限制 KV cache 增长（图 9 显示稳定内存占用、更大内存减少处理时间）；计算约束：HB 栈 decode 联合批（共享权重、KV 按请求区分）、XPU 上 target prefill 与 verification 联合批、draft prefill 单独执行（参数不同、开销可忽略）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运行流程（图 8，请求率 4 req/s）：外部请求注入 XPU task pool（prefill）→ XPU 空闲即组批（PFS 优先 prefill、CHK 切块长 prefill 与 verification 混批）→ 完成 prefill 的请求作为 decode 任务注入 HB 栈 pool → HB 栈按当前 tree width 组批迭代 decode → 达 draft budget 回传 XPU 排队 verification → XPU 空闲时与待处理 prefill 一起组批 → accepted token 回传、清误推测 KV → 下一轮。watermark 检查在注入新任务前执行（内存超阈值则暂缓 prefill）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在扩展 SplitwiseSim 的事件驱动模拟器中实现双 task pool 调度器（事件驱动逐任务推进、XPU 用分段线性性能模型、HB 栈用 silicon-derived 参数）；baseline 用固定 batch 的离线批处理对比。使用要点：把"批组成"从全局同步改为每单元异步自治，配合 watermark 防 KV 超配；与 Continuous Batching 的区别是多了"多单元 × 多任务类型"的编排维度。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
