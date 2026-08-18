## Counting-thread 定时器（自建线程计数定时器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Counting-thread 定时器是在架构定时器不可用（timer-constrained）场景下攻击者自建的计时器：一个独立线程在紧循环中自增共享内存中的计数器，计时通过读写该计数器实现——计数器增量近似流逝时间。要求并发、不中断执行与共享内存资源。它比架构定时器（如 cntvct_el0、rdtsc）更不精确、更嘈杂，但能在定时器被限制/防御的环境下"复活"计时类攻击（SysBumps、S2C、ARMageddon 等使用）。
- 在本文中：counting-thread 定时器作为 TIDE 的对照基线：TIDE 检测到 100,000 个中断时，counting-thread（阈值 2,000）只检出 45.0%（M3 Pro）/62.9%（M1 Pro）/80.5%（M3 Air），不可靠；且它本身是一个 active 线程，会改变 Apple 中断投递（增加 active core 数）。论文用 TIDE 增强它（见下）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 定时线程与测量流程：
```
# 定时线程（独立核）：
while (1): counter++
# 测量线程：
t0 = counter;  <被测代码>; t1 = counter    # 用时 ≈ (t1 - t0) / 每秒增量
# 中断发生：定时线程所在核被抢占 → counter 停止增长 → t1-t0 偏小 → 误判/噪声
```
- TIDE 增强版（去噪）：
```
# 定时线程每个循环迭代：
if x18 == 0:            # x18 被清零 → 本核被中断过
    tide_counter += 1   # 记录中断次数
    x18 = 0x1           # 重置 x18
counter++
# 测量时同时读 counter 与 tide_counter，把被中断的测量丢弃
```
- Annotations：中断会让定时线程暂停，因此任何"被中断的测量"都偏短；TIDE 增强只在检测到中断时额外执行寄存器写+内存写，开销仅 0.06%（100 runs，M3 Air 空闲环境）；在视频噪声下把 SysBumps 成功率从 54% 提升到 81%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：用户态 C 线程 + 共享内存计数器（无需特权）；SysBumps 攻击（macOS KASLR 破解）与缓存侧信道（Prime+Probe）使用；SSBench 论文中亦作为软件计时基线（与 kperf 检测的对抗场景）。使用：测量被测代码执行时间、探测缓存/TLB 状态。限制：要求定时线程所在核不被中断（在 Apple silicon 上 SPI 均匀投递使该假设更难成立）；增加 active core 数会改变中断分布；比架构定时器精度低一个量级。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
