## Time Dilation（时间膨胀调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Time dilation（时间膨胀）是 BULLETTIME（ISCA 2026）提出的运行时线程节奏均衡机制，名字借自《黑客帝国》"子弹时间"（时间放慢、视角正常移动）的比喻。其要解决的问题是：tracing 框架把 trace 落盘的 I/O 延迟是 bursty 且跨线程不对称的（只插桩内存指令时，访存密集线程被反复 I/O 停顿拖慢、计算密集线程几乎不受影响，内核守护线程则完全不受影响），导致应用线程之间、应用与系统守护线程之间的操作重排，破坏被研究行为（内存连续性、同步）相对 untraced 执行的保真度。
- 核心思想：把"tracing 延迟 : 执行进度"的比值在所有相关线程间拉平到最慢线程的水平——对"较快"（tracing 延迟少）的线程注入额外延迟，使其节奏与最慢线程一致，从而恢复 key operation 的原始顺序（正确性条件 C2）。窗口长度 L=1 即 lockstep 强制（任一线程发生 tracing 延迟时所有线程同时停等）；L>1 的窗口化把可能的重排限制在单个窗口内，以有界近似换取远低于逐操作 lockstep 的开销。
- BULLETTIME 的工程实现分两部分：对应用线程用 Buffer-Driven Delay Module 在每线程 trace buffer 填满的 I/O 事件上注入延迟；对内核守护线程（khugepaged 等，不被插桩、无 I/O 事件可用）用 Sleep Dilation Kernel Module 放大其睡眠时长。可选 zstd -7 在线压缩（BT-Comp）以空闲 CPU 换 I/O，抵消延迟注入带来的运行时间增长。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 理论算法（论文 Algorithm 1，窗口化时间膨胀）：输入为 traced 执行 Ops^t 按每线程 L 个操作切成的窗口流。
```
for each window of L operations per thread in Ops^t:
    tracingDelays = [#tracing operations in this window, per thread]
    maxDelay = max over threads of tracingDelays
    for each thread thd:
        injectedDelay[thd] = maxDelay - tracingDelays[thd]
        execute first (L - injectedDelay[thd]) operations of thd's window,
        then stall thd for injectedDelay[thd] time units
        defer last injectedDelay[thd] operations to next window
```
  每窗口把各线程的比值 tracingDelays[thd] : (L − tracingDelays[thd]) 拉平；L=1 退化为 lockstep（精确恢复顺序），L>1 时重排只可能发生在窗口内、跨窗口顺序由窗口粒度上的 lockstep 保证。
- BULLETTIME 的实际调度循环（论文 §IV-C，图 9）：应用线程执行指令 → Pin 把访存 trace 写进 2MB 线程内 buffer → buffer 满触发 I/O 事件（= 一次"窗口"边界）→ Controller 进程接管落盘（O_DIRECT 直写 SSD）并用 EWMA（窗口 5s、衰减率 0.5）更新每线程 progress（= 指令数，用户态/内核态按各自 IPC 加权）与 tracing delay → 计算各线程 progress-to-delay 比值、识别最慢线程 → 对较快线程计算出 injectedDelay，在其下一次 buffer 填满事件时注入（图 9 例子：Thread2 第 2 个 I/O 事件前 progress 4:delay 4，落后于 Thread1 的 2:2，注入延迟把二者拉平；已平衡时注入 0）。内核线程侧：Controller 周期性把最慢线程的减速因子（≈ 应用 CPU 时间 / 等待落盘时间）经内核模块放大 khugepaged 等守护线程的睡眠长度。
- 与 COZ causal profiler 互为镜像：COZ 按比例放慢其它代码段来估计某段代码的潜在加速收益，BULLETTIME 则按比例放慢其它线程来抵消某线程被 tracing 拖慢的效应。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 论文实现（Intel Pin 3.30，Linux 6.13，i7-8700 + SATA SSD）：Pin 内新增 BULLETTIME Controller（拦截 buffer 满事件、算延迟、管落盘）；Pin 外两个模块——Buffer-Driven Delay Module（向应用线程注入延迟）与 Sleep Dilation Kernel Module（kprobes 插桩 schedule_timeout 与 hrtimer_nanosleep、仅对 TASK_INTERRUPTIBLE 任务延长睡眠）。落地要点：(1) 不必跟踪单个 key operation，只需等比例拉平各"key thread"的进度/延迟比；(2) 进度用 EWMA 估计（无法前瞻未来窗口）；(3) O_DIRECT + hugetlbfs 内部 buffer 使框架自身不破坏内存连续性（条件 C1）。效果：Misplaced Memory 平均 8.89% vs Disk-Traced 35.49%、DynamoRIO 56.77%；同步研究 GET:UPDATE 比值保持在 untraced 的 10% 内（Disk-Traced 最高偏离 25×）；代价是运行时间较 Disk-Traced 平均 +35%（最高约 58–60%），BT-Comp 压缩后改善 >2× 且精度无损。代码开源：https://github.com/ysarch-lab/BulletTime。
- 使用方式：用户通过提供一组函数标识 key threads（线程执行到这些函数即被纳入膨胀）；适用于任何数据生成率显著超过存储带宽、且需要无损（完整）trace 的场景。

涉及论文标题：
- BULLETTIME: Time Dilation for High-Fidelity Tracing
