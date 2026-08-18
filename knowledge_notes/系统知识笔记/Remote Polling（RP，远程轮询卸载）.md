## Remote Polling（RP，远程轮询卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Remote Polling 是设备中心视角（device-centric）的 CCM 卸载机制：把 CCM 当作加速器，主机-CCM 交互全部经 CXL.io 消息 + 远程 mailbox（设备 MMIO 寄存器）完成。主机先经 CXL.mem 写 kernel 描述符到设备内存，再经 CXL.io 入队卸载命令，然后反复轮询远程 mailbox 查 kernel 完成标志（真实硬件轮询间隔可达 100µs，每次轮询付 CXL.io 往返），确认后出队，最后经 CXL.mem 同步 load 结果。绝大多数 prior CCM 工作（如 UPMEM 风格、KNN 专用硬件 [37][19]）使用该机制。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
交互时序（AXLE Fig.1a）：(1) CXL.io enqueue 命令 → (2..n) 主机周期轮询 mailbox 直至 CCM 固件写入完成描述符 → (n+1) CXL.io dequeue → 最后 CXL.mem load 结果。特性：CXL.io 消息本身异步，主机可在远端执行期间做其他事（异步执行 ✓）；代价是无法卸载细粒度任务——µs 级 kernel 的纯执行只占 RP 总时间 16.7%（AXLE Fig.3：QKVProj 897K cycles vs BS 888K，轻量 kernel 则 RP 是 BS 的 6 倍开销），轮询间隔 + CXL.io 往返不可隐藏。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：设备 mailbox MMIO 寄存器 + 固件完成描述符；主机侧驱动轮询或中断。AXLE 在 M²NDP 模拟器上另行实现了 RP 模型作基线（配置：固件 2GHz、远程轮询间隔 1µs、CXL.io 往返 350ns）。使用方式：粗粒度（长执行时间）卸载尚可；不适合细粒度卸载与高频交互场景。系统侧后续演进方向是 BS 与异步背流。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
