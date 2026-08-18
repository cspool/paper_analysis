## NVMe/TCP Engine（协议转换引擎：Command Scheduler / PDU Header Generator / PDU Stitcher / PDU Parser / PDU Decapsulator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVMe/TCP Engine = NTI 的核心 IP，在硬件里完成 NVMe 命令/完成 ↔ NVMe/TCP PDU 的协议转换与数据搬运编排（对应软件栈的 interposition 层 + NVMe/TCP initiator 栈）。五个子模块：(1) Command Scheduler——NVMe 队列到 NVMe/TCP 队列的映射；(2) PDU Header Generator——把 NVMe 命令包成 PDU 头，并抽取 PRP 条目入 PRP table；(3) PDU Stitcher——TX 侧按需 DMA 数据 + 头拼接成完整 PDU；(4) PDU Parser——RX 侧边收边切 PDU 头/数据；(5) PDU Decapsulator——从响应 PDU 头提取/生成 NVMe completion。
- 两个协议间的两个"鸿沟"由它桥接：(a) PRP 鸿沟——NVMe 命令的 PRP 指向发起端主机内存，远端 target 不可见，Engine 记录 PRP 使返回数据落到正确缓冲；(b) 完成语义——读数据可能跨多个 C2HData PDU 返回，Decapsulator 追踪"最后一个数据 PDU"到达才生成 completion。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Command Scheduler 的队列解耦（§V-B1）：NVMe 视角命令不关心由哪条 NVMe/TCP 队列处理，因此映射可灵活——Case 1 并行性：目标端 per-queue 吞吐受限时，激活更多更深 NVMe/TCP 队列、round-robin 分发命令吃满并行；Case 2 隔离：AI 负载 per-thread NVMe 队列，若 naive round-robin 会把不同线程命令挤到同一 NVMe/TCP 队列互相干扰，故对应队列可用时保持 1:1、必要时才回退 round-robin。调度器维护每队列状态（队列深度、空闲槽位）与映射决策。
- TX 例子（一次写，Fig.6 ②-⑦）：Header Generator 组 PDU 头 + 抽 PRP → 交给 Stitcher 并同时通知 TOE 准备发送 → TOE 按 TCP 状态发起取数 → Stitcher 判断 PDU 含 payload（Write）→ 按 PRP DMA 取主机数据 → 头+数据拼成完整 PDU → TOE 发出。RX 例子（⑧-⑪）：TOE 交 TCP payload → Parser 切头/数据 → 数据按 PRP DMA 直写主机、头送 Decapsulator → 提 completion → NHI 写 CQ。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：Engine 占 195K LUT/446K FF/464 BRAM/74 URAM——BRAM 主体是 PDU metadata table 与 PRP table；每连接状态（最老未确认序列号等）由硬件持续监控以决定何时向 TOE 发 TX 请求。支持 5000 万 IOPS。
- 使用场景：NVMe-oF initiator 硬件化的参考分区——"命令/完成/数据"全部硬件流水，"admin/配置/错误恢复"留软件；与 ANO（仅 RX、丢包回退软件、需内核补丁）与 XLIO（仅 TX）形成对照：NTI 是 TX/RX 双侧全硬件。信息缺口：论文未公开五子模块的内部流水深度与时钟域划分。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
