## Key State 与 Key Operation（关键状态与关键操作正确性模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BULLETTIME（ISCA 2026）用来形式化"被 tracing 的应用行为"、进而建立 traced 执行正确性条件的一对建模概念。Key state（关键状态）s = 定义所研究行为所需的最小状态集合：内存连续性研究里 s 包括空闲物理内存布局与应用的虚拟-物理映射；若 trace 用于 cache/TLB 模拟，还要加应用数据结构的分配布局；若功能正确性依赖条件分支变量/系统调用变量，s 也要含它们。Key operation（关键操作）op = 任何会修改 key state 的事件（s' ← op(s)）：连续性研究里是所有内存分配/释放以及内核守护线程（如 khugepaged）的页迁移；同步研究里是修改锁等共享变量的操作。
- 在此模型上，untraced 执行建模为 key operation 的有序序列 Ops = {op1,...,opn} 及由初始状态 s0 推出的状态版本序列 S = {s0,...,sn}（si+1 ← opi+1(si)）；traced 执行建模为 Ops^t = Ops ∪ T（T 为 tracing 框架自身注入的操作）。两条正确性条件：C1——tracing 框架不得修改 key state（T 中没有任何 key operation）；C2——key operation 的顺序在 traced 与 untraced 执行中一致（Ops 与 Ops^t 去除 T 后逐序相同）。C1 由框架设计保证，C2 由 time dilation 保证。
- 价值在于 scoping：定义过宽（要求全系统所有线程所有操作全序一致）在数百上千并发组件的真实系统上不可解；定义过窄（手工挑选组件）会漏掉偶尔影响行为的冷门守护线程。key state 模型只保留"真正影响所研究行为"的组件与操作。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程（BULLETTIME 内存连续性研究）：研究者声明 key state = {空闲物理页布局, 应用页表映射} → 反推出 key operation = {应用 malloc/mmap 触发的分配、内核 khugepaged 的页折叠/迁移} → 识别执行这些操作的"key threads" = {应用工作线程, khugepaged} → 框架按 C1 消除自身对 key state 的修改（trace I/O 走 O_DIRECT 绕过 page cache 防 4KB 页碎片、内部 buffer 用 hugetlbfs 2MB 大页并远离应用分配、插桩锁不触碰应用 key state）→ 按 C2 对 key threads 做 time dilation（应用线程注入延迟 + 内核线程睡眠膨胀），使其操作顺序与 untraced 一致 → 端到端指标（页大小覆盖分布）恢复 untraced 行为。
- 关键简化：算法不要求识别单个 key operation，只要识别执行它们的 key threads 并等比例拉平其进度/延迟比。这使模型从"不可跟踪的操作全序"降维为"可管理的线程节奏均衡"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：BULLETTIME 提供一个简单接口——用户提供一组函数，任何线程执行到这些函数即被纳入 time dilation（对应 Algorithm 1 的 Threads 列表）。用户侧仍需自行声明 key state（连续性研究=空闲内存列表+页表；同步研究=锁变量），但无需枚举 key operation。
- 使用场景：任何依赖高保真 trace 的研究（cache/TLB 模拟、内存管理子系统设计、同步与竞态分析）在定义"什么行为必须保真"时使用；同样可推广到 profiling 与调试。评估端到端指标而非逐操作顺序，因为逐操作追踪本身会干扰被研究行为。信息缺口：论文未说明如何在完全自动化的通用研究（用户不声明 key state）中推导 key state。

涉及论文标题：
- BULLETTIME: Time Dilation for High-Fidelity Tracing
