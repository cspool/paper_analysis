## DSA 工作队列与组抽象（WQ / DWQ / SWQ、Engine、Arbiter、Group）

术语解释
Intel DSA 的资源组织与隔离单元：一个 group 内可配置若干 work queue（WQ）与处理 engine，配组级 arbiter；DWQ 单客户端独占、SWQ 多客户端共享，用于在客户端间隔离队列与执行资源。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Intel DSA 的资源隔离单元是 group（组）：一个 group 内可配置一个或多个 work queue（WQ）与一个或多个 processing engine，并配一个 group 级 arbiter。客户端经 MMIO portal 把 64B work descriptor 提交到 WQ；arbiter 在 WQ 间按优先级/round-robin 派发 descriptor 到空闲 engine；engine 解码执行（读源 → 变换 → 写目标 → 写完成记录）。WQ 分两型：Dedicated WQ（DWQ，单客户端独占，配 MOVDIR64B 提交）与 Shared WQ（SWQ，多客户端共享，配 ENQCMD/ENQCMDS、PASID 支持）。DarkStream 实测（Xeon Platinum 8558）：单 DSA 设备共 128 个 WQ 项、4 个 engine；SWQ 在接近 64 项容量时提交延迟陡增，DWQ 下两客户端延迟无相关性；两 DWQ 共享同一 engine 时 Memory Move 慢 24%+，分组隔离 engine 后与 solo-run 无差异——即队列与 engine 的隔离是有效的。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
隔离作用链：descriptor 提交 → 客户端专属 DWQ 入队（其他客户端的饱和不影响本队列）→ group arbiter 只在其组内 WQ 间派发（跨组不共享 arbiter）→ descriptor 由本组 engine 独占执行。因此同一 DSA 设备内不同 group 的客户端在"队列-仲裁-执行"三段互不可见。DarkStream 的对照实验流程：aggressive 客户端持续提交 Memory Move 饱和 WQ，另一客户端测提交延迟相对 WQ occupancy 的变化——SWQ 低占用时延迟平稳、接近 64 项上限时陡增（争用跨客户端传播），DWQ 配置下延迟与 aggressive 客户端 occupancy 完全无关；engine 对照同理（solo-run / 共享 engine / 分组隔离三配置）。结论：WQ 与 engine 隔离有效，但都无法消除隔离边界之下 I/O fabric interface 的设备级共享吞吐（见本库"DSA I/O Fabric Interface"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
配置经 accel-config（idxd-config 仓库）完成：`accel-config config-wq --group-id <gid> --mode dedicated/shared --wq-size <N> <dev>/wq0.<n>`，再 `accel-config enable-device/group/wq`；Linux idxd 驱动把 WQ 暴露为字符设备（/dev/dsa/wqX.Y），用户态经 mmap portal + ENQCMD/MOVDIR64B 提交。DWQ 适合性能隔离（每客户端独立队列+engine），SWQ 适合多客户端复用提高利用率。DarkStream 的使用方式：Source/Sink 各持独立 group（DWQ+engine）提交 1-byte Memory Move，队列/engine 隔离被绕过、争用发生在共享的 I/O fabric interface。Web 证据：Intel DSA Architecture Specification 与 ASPLOS'24 DSA 量化分析（arXiv:2305.02480）描述 group/WQ/arbiter 结构与 portal/ENQCMD 提交机制。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks
