## eBPF（Extended Berkeley Packet Filter，驱动-用户态通信层）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Linux 内核提供的安全可编程执行环境：允许在受内核验证器（verifier）校验后把小型程序安全地挂载到内核指定点（tracepoints、kprobes、XDP、tc 等），并通过共享 maps 与用户态双向通信；kfunc 特性允许 eBPF 程序调用/操作内核与驱动变量。ObservUVM 用它实现 UVM 驱动与 userspace 引擎之间的通信层：在修改的 UVM 驱动上添加 page fault、access counter 通知、prefetch、eviction 四类 tracepoint，eBPF 程序在这些 tracepoint 处把事件（含地址、bitmap 等参数）经 map 上行给 userspace；userspace 的决策（setEvictionRegion、setPrefetchThreshold、setPrefetchRegion、setObservabilityCandidate、setFeedbackCandidate）经 map 下行由驱动执行。相比在驱动内做策略，eBPF + tracepoint 实现"机制在驱动、策略在用户态"的分离：改策略不需重编译/重载驱动，规避崩溃与安全风险。
- 通信接口（论文 Table I）：上行 onPageFault(Address)、onAccessCounter(Address)、onEviction(Address)、onPrefetch(Address,Bitmap)；下行 setEvictionRegion(Address)、setPrefetchThreshold(Integer)、setPrefetchRegion(Address)、setObservabilityCandidate(Address)、setFeedbackCandidate(Address)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 ObservUVM 中的运转流程：GPU 触发页错误 → 修改的 UVM 驱动处理并发射 tracepoint（事件已带参数）→ 内核中 eBPF 程序被触发，把事件写入 BPF map（环形缓冲/数组）→ userspace 引擎轮询 map 取出事件 → 事件循环把事件分发给已注册策略的对应回调 → 策略决策写入下行 map → 驱动读取并强制执行（迁移/换出/预取/设观察候选）。全程零用户态-内核边界系统调用开销（map 共享内存），eBPF 程序经过 verifier 校验保证内核安全。root 权限即可运行 userspace（加载 eBPF 程序）。
- 工程证据（本地 vault）：human_notes 提及 Merlin: Multi-tier Optimization of eBPF Code（eBPF 性能/体积优化，score 22）与 SnapBPF（eBPF 用于 serverless 快照预取）等 eBPF 应用；paper_secs 中 BoostX/Serverless 论文引用 Exo（eBPF 加速存储半虚拟化）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C 写 eBPF 程序 + libbpf 加载（论文指定 libbpf）；在驱动代码加 tracepoints（Linux 内核 tracepoints API，docs.kernel.org/trace/tracepoints.html），BPF maps 做共享数据，kfunc 让 eBPF 程序操作驱动变量。使用：`sudo` 运行 userspace（eBPF 需要 root）；编译驱动（compile_drivers.sh）与 userspace（compile_userspace.sh）后 run_key.sh 跑实验。eBPF 是通用技术：广泛用于网络（XDP 卸载）、可观测性（BCC/bpftrace）、安全（Seccomp eBPF）与存储；ObservUVM 是其作为"驱动-用户态策略解耦"通信层的示例。Web 证据：Linux 内核 eBPF 文档 https://docs.kernel.org/bpf/ 、kfunc 文档 https://docs.kernel.org/bpf/kfuncs.html 、libbpf 文档 https://docs.kernel.org/bpf/libbpf/libbpf_overview.html（论文引用）。

涉及论文标题：
- Observability-aided GPU Memory Oversubscription
