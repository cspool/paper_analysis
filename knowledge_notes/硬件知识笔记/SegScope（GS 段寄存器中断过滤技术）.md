## SegScope（GS 段寄存器中断过滤技术）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SegScope（Zhang et al., HPCA 2024，本文引用 [57]）是一种利用架构足迹（architectural footprints）细粒度探测中断的技术，本文将其用于非特权 TimeGaps 采集时的中断过滤。核心观察：处理器从中断返回、从内核态切回用户态时，段寄存器（如 GS）会被清零为 0。因此攻击者在读 TSC 前把 GS 设为非零值（如 1），若检测到 timestamp jump 后 GS 仍为 1，则该 jump 是 halted 的 TimeGap（无中断介入）；若 GS 被清零，则说明经过了内核→用户态切换，jump 由中断引起，判为噪声丢弃。
- 作用：替代需要特权（isolcpus 隔离核 + PMC）的 halted/unhalted 分类方法，使 TimeGaps 采集在无特权 native 环境下也可用，且比把中断当噪声的粗粒度方法更干净（94.3% 的 jump 是 halted，但残留中断噪声会污染信号）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程：
```
设置 GS = 1（非零哨兵）
loop {
  t1 = rdtscp();
  ... 执行测量 ...
  t2 = rdtscp();
  if (t2 - t1 > threshold) {
    if (GS == 1) 记录为 TimeGap;   # 未经过中断返回，CPU halted
    else         丢弃（中断噪声）;  # GS 被中断返回清零
  }
}
```
  中断处理路径：用户态 → 内核态中断处理（GS 保存/恢复）→ 返回用户态时 GS 清零 → 程序读到 GS==0。TimeGap 路径：CPU 挂起不涉及内核切换，GS 保持 1。
- 与浏览器场景对比：浏览器中无段寄存器可控，改用 loop counting（空循环完成迭代数下降）检测 TimeGap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C 程序内联汇编设置/读取 GS 段寄存器（FSGSBASE 或 wrgsbase/rdgsbase 指令；现代 Linux 用户态通常可用），与 rdtscp 采集器集成。使用场景：无特权 native 侧信道测量中的中断过滤（本文 TimeGaps 采集、prior work 的执行速度侧信道）。限制：只区分"是否经过中断返回"，无法区分 SMM 等特殊路径；需在支持 GS 基址操作的用户态环境实现。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
