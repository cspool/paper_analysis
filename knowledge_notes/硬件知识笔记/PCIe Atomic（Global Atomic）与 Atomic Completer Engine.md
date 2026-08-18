## PCIe Atomic（Global Atomic）与 Atomic Completer Engine

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PCIe 3.0 起引入 PCIe Atomic 事务（原生 FetchAdd/CmpAndSwap/Swap），把原子性保证下沉到 PCIe 子系统：RNIC 的 Global Atomic 模式即让 PU 直接向 PCIe 链路层发起原子事务、完全绕过内部锁定表，多设备间原子性由 PCIe 保证。Atomic Completer Engine 是平台上执行 PCIe 原子事务的引擎（vendor-specific 实现，容量受限）。论文给出首次系统评估：CX-6 启用 PCIe Atomic 后吞吐与 stride 无关（无槽争用）但只有平均 14.4 Mops/s——是锁定表路径 42.3 Mops/s 的 34.0%，瓶颈即 Atomic Completer Engine；AMD EPYC 7281 平台上为 27.6 Mops/s。Web 证据：公开文献中"Atomic Completer Engine"无统一定义，但多篇工作确认 PCIe/RNIC 原子完成引擎是吞吐瓶颈（ConnectX-5 CAS 仅 ~8.4 Mops/s vs READ 65M，NSDI'22 RedN）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Global Atomic 执行流程：RNIC PU 收到原子请求 → 不查锁定表，直接生成 PCIe Atomic 事务经 Root Complex 送达 Completer Engine → 引擎在 PCIe 域内原子完成（读-改-写一体化）→ 结果按原路返回 RNIC。CX-5-PA 异象解释：stride<512B 时 Completer Engine 是瓶颈（吞吐与 CX-6-PA 接近）；stride>512B 时瓶颈回到锁定表（说明 CX-5 两条机制并用、互相拖累），所以 CX-5-PA 全区间最差。论文因此默认禁用 PCIe Atomic（HCA 模型），并论证任何 HCA 下正确的设计对 Global Atomic 同样兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开启条件：需要主板/处理器原生支持（Intel Xeon Scalable、AMD EPYC）+ BIOS 配置；Web 证据：Intel CPU 长期只支持 Completer 角色（device-to-host），Requester/Forwarding 支持受限（StackOverflow 修订记录），不同 Root Port 间设备对设备原子一般不可用。限制：不支持 masked CAS 等增强原子（Sherman、SMART 依赖）。Fusa 兼容场景（Exp#7）：启用 PCIe Atomic 后 Fusa 把 Completer Engine 当第三后端做混合卸载，YCSB-A Zipfian 下 +36.4% 吞吐、P99 延迟 -92.0%。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
