## HATS（Hardware-Accelerated Traversal Scheduling，硬件加速遍历调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HATS（Hardware-Accelerated Traversal Scheduling，[43]）是加速图处理的系统：把难以预测的图遍历从 core 上解耦，利用可编程内存层级的回调在缓存内完成遍历，使主线程以顺序、局部性好（易预测）的模式访问数据。täkō 的 HATS 实现（täkō 论文 [55]）：创建 phantom 地址范围顺序存放图边——主线程顺序访问 phantom 地址（好局部性），OnMiss 回调在每次 miss 时执行图遍历并提供下一条边（每次只返回一条、不重复）；被过早逐出的边由 OnEvict 记录到普通内存日志，主线程遍历完 phantom 后处理日志。täkō 论文建议 OnMiss/OnEvict 应 side-effect-free（因可任意时刻触发）但未形式化原因，HATS 实现违反该建议——täkōFormal 用 MCM 分析 HATS 并回答"OnEvict 副作用何时可接受"。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（hatsR vs hatsNR litmus test，图 13）：[e] 是 phantom 边（0=合法边、1=非法），Core 0 用 RMW([e],r1,1) 原子读边并标记已处理（防"读与标记之间被逐出导致处理两次"），然后 FlushRange[e] 确保在途 OnEvict 完成，再读日志 [ℓ]；OnMiss 用引擎局部视图 [g] 决定返回合法边（[g]=0→[e]←0，并推进遍历）或非法边（[g]=1→[e]←1）；OnEvict 把边写进日志 [ℓ]。hatsR（无 (i9) 检查）racy：cache 可无 core 参与地反复 OnMiss-OnEvict（预取加载→逐出→再加载），第二次 OnMiss-OnEvict 的写日志与 (i3) 读日志竞争（真实应用中造成虚假边入日志）。hatsNR 修复：OnEvict 只在 [e]≠1（合法边）时记日志（(i9)/(i10)），OnMiss 遍历完成后返回非法边——post-traversal 的 OnEvict 永不记日志，race 消除（Alloy 大 bound 搜索无 race 佐证）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：OnMiss 持有引擎局部图状态（如 [g]），主线程 RMW 原子消费边 + FlushRange 同步 + 读日志；设计准则（täkōFormal 得出）：遍历完成后 OnMiss 不得提供虚假数据（返回非法边），OnMiss/OnEvict 除注册地址与引擎局部地址外不得访问其他地址。Leviathan（secs_2025/36）也实现 HATS 作为 decoupled graph traversal 应用（1.7× speedup，score 658 的搜索命中），说明该 workload 是 PMH 类系统的标准用例。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
