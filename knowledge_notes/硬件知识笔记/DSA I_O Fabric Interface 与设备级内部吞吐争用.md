## DSA I/O Fabric Interface 与设备级内部吞吐争用

术语解释
Intel DSA 设备内所有 engine 数据读写的共享出口通路，吞吐设备级共享（DarkStream 实测峰值约 30 GB/s），位于 group 隔离边界之下，是跨客户端时序争用的根因。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
I/O fabric interface 是 Intel DSA 设备内部所有 engine 与主存/片上互连之间的数据通路：DSA 执行的所有读/写数据（Memory Move、Fill、Compare 的实际搬移字节）都必须穿越该接口，其吞吐为整个 DSA 设备共享且存在上限（DarkStream 在 Xeon Platinum 8558 实测约 30 GB/s 峰值）。由于 group 隔离只划分了 WQ、engine 与 arbiter，I/O fabric interface 位于隔离边界之下：任何客户端都无法独占该吞吐，并发数据搬移相互挤占带宽、抬高对方操作延迟。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DarkStream 的定位实验（solo-run / co-run 同 DSA / co-run 异 DSA，客户端均持专属 group，传输尺寸跨数量级扫描）：随传输尺寸增大三者吞吐先同步爬升到约 15 GB/s；此后 co-run(同 DSA) 饱和，而 solo-run 继续升到 30 GB/s 峰值、co-run(异 DSA) 也不饱和；延迟在吞吐饱和点开始分化。这说明争用不来自 descriptor 处理（WQ/engine 已隔离）而来自设备内实际数据搬移——即 I/O fabric interface 的设备级共享吞吐。该争用点构成 timing channel：并发客户端操作延迟随对方活动升高（Sink 的 1-byte Memory Move：Source idle 约 1400 cycles，active 2000–4000 cycles；攻击者 1 MB Move 基线约 75000 cycles、受害负载争用时 >80000 cycles）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
这是既有微架构事实而非软件机制，Intel 架构文档描述 DSA 数据经 I/O fabric 与系统互联交互；DarkStream 的利用方式：(1) 隐蔽信道——Source 按 time slot 饱和/空闲提交 1-byte Memory Move 调制接口吞吐占用，Sink 测自身操作延迟解码 bit；(2) 侧信道——受害者经 DTO 卸载的 memcpy/memset 数据搬移与攻击者 1 MB 探测操作争用同一接口，延迟 trace 反映受害负载的操作频率与尺寸分布。缓解方向（论文讨论）：DSA 设备独占分配、engine 时间复用（吞吐 30→7.5 GB/s）、性能计数器检测，均带性能-安全权衡。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks
