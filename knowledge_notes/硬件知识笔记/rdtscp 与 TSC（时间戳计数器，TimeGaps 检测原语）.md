## rdtscp 与 TSC（时间戳计数器，TimeGaps 检测原语）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TSC（Time Stamp Counter）是 x86 处理器的高分辨率单调递增计数器，以恒定速率（invariant TSC，基准频率）推进，与 CPU 当前运行频率解耦——这使其能"见证"CPU 挂起：CPU 停止执行期间 TSC 仍在计数。rdtscp 是读取 TSC 的指令，相对 rdtsc 额外通过 ECX 输出核心 ID 且带序列化语义（避免乱序执行造成读数错位）。本文用 rdtscp 实现 TimeGaps 采集：循环读 TSC、相邻读数差超过阈值（5000 cycles）记为 timestamp jump（Listing 1）。
- 关键性质：正常循环中相邻 rdtscp 读数差约 20–40 cycles；跳变可能来自中断/SMM（unhalted，CPU 仍执行）或 CPU/iGPU 频率切换导致的 halted（TimeGaps）；阈值 5000 cycles 用于忽略 cache/TLB miss 等延迟。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程（采集器在隔离核上运行）：
```
index = 0; prev = rdtscp();
loop {
  current = rdtscp();
  if (current - prev > 5000) { jumps[index] = current - prev; index++; }
  prev = current;
}
```
  CPU 挂起期间 TSC 推进、程序不执行 → 恢复后下一次 rdtscp 读到的大差值即 halted jump（TimeGap）；中断返回也产生差值但 CPU 未挂起（可用 PMC 或 SegScope 区分）。跨核同步观测：两个隔离核同时跑采集器，TimeGaps 一一对应（起止差 46.21 cycles/0.69%），因为共享时钟域同时挂起。
- 非特权替代：攻击场景无 PMC/隔离核权限，native 用 SegScope（rdtscp 前置 GS=1，返回用户态若 GS 被清零则判定为中断）过滤；浏览器无高分辨率定时器，用 JS 空循环计数（TimeGap 使完成迭代数下降）替代。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：x86 指令 rdtscp（Intel/AMD 均支持），GCC 内联汇编或 __rdtscp() 内建函数；C 采集器 gaps_collector_pmc/gaps_collector_5ms（artifact 开源，Zenodo https://doi.org/10.5281/zenodo.19450827）。使用：性能剖析中的精确计时基元、侧信道测量（本文 TimeGaps 采集、Flush+Reload 等 timing 测量）。注意事项：浏览器/Web 环境无 rdtscp（受 reduced timer precision 限制），需用 loop counting 替代；阈值选择需按平台噪声调整。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
