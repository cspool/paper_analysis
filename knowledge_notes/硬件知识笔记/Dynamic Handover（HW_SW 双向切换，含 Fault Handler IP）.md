## Dynamic Handover（HW/SW 双向切换，含 Fault Handler IP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Dynamic handover = NTI 的硬件↔软件无缝协作机制：I/O 处理过程中任意时刻，控制面软件可临时接管、处理完毕后硬件恢复；接管/恢复以内存映射硬件寄存器交接最小上下文（NVMe doorbell、TCP 连接状态等），粒度细到单条 NVMe 或 NVMe/TCP 队列——单队列故障只复位该队列上下文、重传其未完成命令，其他队列 I/O 不断流（主机视角 seamless）。
- 触发与裁决由 Fault handler IP 完成：分别位于 NVMe/TCP Engine 内部及 Engine↔NHI、Engine↔TOE 两个边界，三个职能——(1) 监控硬件状态/错误寄存器并向软件报告（如 C2HTerm PDU、非法 CID、digest mismatch）；(2) 给软件暴露发起 handover 的接口（如 NVMe/TCP keep-alive 超时：周期性目标响应未按时到达即触发）；(3) 可恢复错误直接硬件自愈（如 Engine-NHI 边界发现非法命令字段，立即回一条携带对应错误的 completion，不动软件）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 单队列错误恢复流程（§V-C）：某 NVMe/TCP 队列收到非法 PDU 头 → Fault handler 捕获 → 硬件仅复位该队列上下文（其他队列继续）→ 通知软件 → 软件经 MMIO 读最小上下文、执行控制面操作（如重置故障队列）→ 写控制寄存器令硬件恢复 → 硬件重传该队列未完成 NVMe 命令 → 主机无感知。反面教材对比：传统软件栈把"动态变更与错误处理"当作软件职责，硬件化后若没有这套机制，任何异常都会拆掉整个 I/O session 重建（停机 + 开销大）。
- 网络/热事件验证（§VI-A3）：active/active 路径故障后 NTI 把丢失命令经存活路径重传、恢复 200 Gbps；热管理事件中 90% 阈值节流、温度回落自动恢复满速——二者都不需要主机干预。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：MMIO 寄存器暴露最小上下文（doorbell、TCP 连接状态）而非全状态（软件只需能 resume）；Fault handler 按边界布点（Engine 内、NHI 边、TOE 边）覆盖三类错误源；硬件可处理的错误不升级到软件。
- 使用场景：所有"硬件数据面 + 软件控制面"的加速器（DPU、SmartNIC）设计弹性；与 F4T/FVM 的可更新性结合，构成"协议更新（软件）+ 数据面稳定（硬件）"的运维模型。信息缺口：论文未给出 handover 的切换延迟测量数据。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
