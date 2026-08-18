## kprobes（Linux 内核动态插桩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- kprobes 是 Linux 内核的动态插桩设施（自内核 2.6.9 起），可在运行时向内核函数入口（或任意指令地址）动态挂接探针，无需重新编译内核。BULLETTIME 用它实现 Sleep Dilation Kernel Module：插桩睡眠路径内核函数 schedule_timeout 与 hrtimer_nanosleep、修改其参数来放大睡眠时长，从而放慢 khugepaged 等内核守护线程。
- 工作机制（Web 证据：Linux 内核官方 Kprobes 文档）：注册 kprobe 时，内核保存被探针指令副本并把其首字节替换为断点指令（x86 上为 int3）；CPU 执行到断点时触发陷阱，保存寄存器，经 die notifier 进入 kprobe_int3_handler，依次执行 pre_handler（可修改保存的寄存器/参数）→ 单步执行原指令副本（避免其它 CPU 跳过探针点的竞争窗口；较新内核 x86 用 int3 模拟单步）→ post_handler → 继续原流程。kretprobes（返回探针）把返回地址替换为 trampoline，函数返回时先执行返回 handler。CONFIG_OPTPROBES 下常用探针点被优化为跳转到 detour buffer 以减少开销。
- 注意（Web 证据：linux-netdev 讨论）：编译器内联可能使探针"成功注册但永不命中"——如 Clang 把 hrtimer_nanosleep 内联进 SyS_nanosleep，kprobe 挂上也不生效。这是选择睡眠路径插桩点时的主要工程风险。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- kprobes 是内核侧"运行时插桩"原语，与用户态 DBI（Pin）在功能上对称：注册 → 断点替换 → 命中时劫持执行 → 修改语义后放行。BULLETTIME 的插桩流程：内核模块注册 schedule_timeout 与 hrtimer_nanosleep 上的 kprobe → 内核守护线程（如 khugepaged 周期工作、其余时间睡眠）每次调用睡眠函数时，pre_handler 按当前膨胀因子放大睡眠时长参数 → 线程以更慢节奏醒来工作，与最慢的被 trace 应用线程对齐。约束：只对 TASK_INTERRUPTIBLE 状态的 schedule_timeout 调用膨胀，避免干扰事件驱动/面向硬件的关键内核活动；膨胀因子 ≈ 被 trace 应用 CPU 时间 / 等待 trace 落盘时间，由用户态 Controller 周期性计算并经 sysfs 传给模块。
- 逻辑链：系统线程不被 Pin 插桩 → 没有 tracing I/O 事件可注入延迟 → 但其行为（页折叠）影响连续性研究 → 利用其"固定周期 = 工作 + 睡眠补足"的节律，把膨胀施加在睡眠时长上，等效于对系统线程做时间膨胀。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：经典 kprobe 模块注册 struct kprobe（pre_handler/post_handler），或经 BPF（kprobe/fentry 程序）动态加载；适合任何"运行时拦截内核函数、调整行为参数"的场景（性能分析、故障注入、BULLETTIME 的睡眠膨胀）。论文强调该机制遍布所有 Linux 系统，使 BULLETTIME 可跨机器与架构部署。局限：内联/尾调用优化可使探针静默失效；探针在热路径上引入陷阱开销；需 root/内核模块权限。信息缺口：论文未说明对 hrtimer_nanosleep 被内联的情形如何兜底。

涉及论文标题：
- BULLETTIME: Time Dilation for High-Fidelity Tracing
