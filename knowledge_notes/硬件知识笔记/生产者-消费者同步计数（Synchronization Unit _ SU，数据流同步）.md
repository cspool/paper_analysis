## 生产者-消费者同步计数（Synchronization Unit / SU，数据流同步）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- M100 的数据流同步机制：一个 agent 更新自身执行状态（update 同步计数），另一 agent monitor 该状态决定是否继续（期望值判断）。TPB 内 Synchronization Unit（SU）管理硬件计数器：功能单元 claim 一个计数器、在 TPB 指令定义的执行阶段触发 update（计数 +1）；monitor 请求带期望值，计数器达到/超过才响应，否则请求单元暂停。该机制可双向（producer 更新 produce 计数并 monitor consume 计数，反之亦然，形成流水），可扩展到多 agent 同步网络；还支持 barrier、broadcast、reduction，跨 NPU 亦可。与传统 atomic/独占 load-store 不同：无需 cache 一致性，专用硬件开销极小，同步粒度软件控制。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（TCU→CVU 流水）：TCU 把卷积输出写 HBSM 预分配区间，写完更新 SC 计数 → CVU monitor 该计数达期望值后开始读/处理 → 处理完更新另一 SC 通知 TCU buffer 已释放 → 继续下一 block。HBSM 访问绑定同步：写访问仲裁获胜即全局可见（无后续请求可超越），保证数据流一致性。跨 TPB：SU 支持经 DRB/NoC 的远程同步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每 TPB 一个 SU，硬件同步计数器 + update/monitor 逻辑（本地 + 远程）。与 GPU 的 mbarrier/atomic 相比是专用、无一致性依赖的轻量机制（Web/文献中 dataflow 同步的常见形式）。使用：编译器/固件分配计数器与期望值、控制同步频率，替代传统 barrier/atomic 使 dataflow 流水开销最小；是 M100 prefill 高利用率（TCU/CVU/CSU/GSDU 持续重叠）的来源。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
