## 环境迁移（Environmental Transition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
环境迁移（environmental transitions，源自 Wickerson et al. [64]）是状态机建模中"不依赖指令、由环境（硬件）任意触发"的迁移，用于过近似（overapproximate）不受程序控制的硬件行为。täkōFormal 用它建模缓存行为：由于回调可被 prefetch/eviction 等任意触发，内存层级迁移被从 core/engine 的指令解耦——一个 load 的执行被拆成两个独立迁移：PerformLoad（仅当数据在 L1 时执行）与 SendGetS（数据不在缓存时任意执行），从而同时覆盖"指令触发请求"与"预取触发请求"两种因果；回调调度/执行本身也是环境迁移，且允许在前提满足时无限重复（如 OnMiss-OnEvict 序列可在无 core 请求时反复执行，过近似替换策略/预取器触发回调导致的意外结果）。Pensieve [67] 用 uninterpreted functions 过近似微架构安全行为，是同类思想。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（图 15）：两个 load（(1) 读 [x]、(2) 读 [y]）程序序中 (1) 在前，但环境迁移使它们的请求 (3)(4) 可任意交错（预取可乱序发 GetS）；(5) 的 [z] 逐出也可任意插入。模型不依赖具体 prefetching/替换策略即可覆盖所有"缓存何时搬/逐出数据"的时序——这正是 soundness 证明"对任意替换策略/预取策略/NoC 细节成立"的关键：即使未来架构师改策略，MCM 证明依然有效（参数化 + 环境迁移双重保证）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：在 Dafny operational model 中把内存层级迁移写成"前提满足即可执行"的环境迁移（非指令驱动），Network 中 coherence 消息用无序集合（任意重排过近似 NoC）、engine 回调请求按地址 FIFO（täkō 要求）。使用要点：环境迁移使模型可能包含现实中不可达的执行（过近似），soundness 只需"实现的所有执行⊆MCM"（单方向），过近似不会破坏该方向；与之相对，完备性（MCM⊆实现）不要求。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
