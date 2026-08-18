## 隐蔽信道（Covert Channel）与信道容量/带宽建模（BAC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
隐蔽信道是两个不互通实体违反系统安全策略、通过共享资源调制状态/时序来传信息的通道：发送方（Source）主动调制共享资源占用，接收方（Sink）观测由此产生的可测变化解码数据；双方需协作，与侧信道的区别在于侧信道观察的是不知情受害者的附带泄漏。DarkStream 的隐蔽信道介质 = Intel DSA 设备内部 I/O fabric interface 的共享吞吐：Source 在每个 time slot 以 active（持续异步提交 1-byte Memory Move）/idle 两态编码 1/0，Sink 连续提交自己的 1-byte Memory Move 并记录延迟，slot 内延迟中位数高于阈值判 1、否则判 0；128-bit 帧 + 10101010b 前导码同步帧头。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
信道建模流程：带宽 = 容量 × 传输频率；容量按 Binary Asymmetric Channel 计算——ε0=P(0→1)、ε1=P(1→0) 为两方向 bit-flip 概率，

$$C_{BAC}=\frac{\epsilon_0 H_b(\epsilon_1)-(1-\epsilon_1)H_b(\epsilon_0)}{1-\epsilon_0-\epsilon_1}+\log_2\left(1+2^{\frac{H_b(\epsilon_0)-H_b(\epsilon_1)}{1-\epsilon_0-\epsilon_1}}\right)$$

其中 $H_b(p)=-p\log_2 p-(1-p)\log_2(1-p)$；ε0/ε1→0.5 时容量→0。传输频率 = slot 时长倒数，由 Source/Sink 共享约定。扫描 40–256 KHz：低频长 slot 让 Sink 每 slot 收集大量样本、中位数统计稳定、容量≈1；频率升高后每 slot 样本变少、容量下降；147 KHz 处带宽峰值 129 Kbps。威胁模型：Source 持有机密但无外网（如木马进程），Sink 有外网但无数据权限，两者都是普通用户态进程、分处不同 core 与不同 DSA group，仅共享 DSA 设备。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现细节：Linux idxd 驱动把 DSA WQ 暴露给用户态，进程经 write syscall / ENQCMD 直接提交 descriptor（无需 root，只要管理员分配 WQ 权限）。工程评估：(1) 跨处理器——DSA 虽在片上但以 PCIe RCiEP 暴露，双路 Xeon Gold 6554S 上 Local-Local 92 Kbps、Local-Remote 78 Kbps、Remote-Local 92 Kbps（远端访问经 socket 互连、方差略增）；(2) 抗背景干扰——stress-ng 施加 10/50/90% CPU 负载时容量仅小幅下降（DSA 执行与 CPU 解耦），仍 >100 Kbps；第三方 DSA 噪声（4 KB/64 KB/1 MB Memory Move）分别占约 3/18/28 GB/s 吞吐，1 MB 静态噪声下约 70 Kbps，随机噪声（每操作在 4 KB/64 KB/1 MB 随机选、固定阈值）下至多 49 Kbps，且 2–4 个噪声进程因聚合流量方差下降反而比单噪声进程更稳。Web 证据：TCSEC 对隐蔽信道的经典定义（违反安全策略的信息传递）与 timing channel"通过调制资源使用时序传信"的表述（trustworthy.systems Time Protection 背景页、IEEE 共享调度器时序泄漏文献）。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks


MDP 隐蔽信道补充视角（SSBench, ISCA 2026）：除 DSA/I/O 吞吐介质外，CPU 微架构的 Memory Dependence Predictor（MDP）表项状态也是隐蔽信道介质——发送方（源）按 bit 控制某 store-load 对的依赖关系从而更新共享 MDP 计数器（bit 1 → 计数器更新到 3，bit 0 → 保持 0），接收方（探测方）探测该表项计数器推断 bit；关键优势是该介质不依赖 cache/TLB 行为（MDP 更新结果只取决于 store/load 地址是否匹配），cache miss/inst ≈0.01、TLB miss ≈0，可绕过 kperf 等性能计数器检测器。SSBench 的 MDP-CC 在 Apple M1-M4 效率核上实现：瞬态隐蔽信道真容量 41129（M1）–152144 bps（M4）、BER ≤0.06；内核→用户空间信道（M2 kext）真容量 159578.30 bps；相对同平台 cache 隐蔽信道（真容量 ≤139320 bps、miss 0.27）与 TLB 隐蔽信道（≤1145 bps、miss 0.99）在吞吐与隐蔽性上均更优。带宽更高源于 Apple 用户态无法执行 cache/TLB flush 指令（驱逐需大量地址访问），而 MDP 更新/探测只需 4 个 store-load 对。


- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
