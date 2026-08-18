## UVM 换出策略（Eviction Policies：LRM / LRU / LFU / Cyclic Protection / Tournament）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPU UVM 内存超订下决定"从 HBM 换出哪个 2MB 区域"的运行时策略。默认 NVIDIA UVM 用 Least Recently Migrated（LRM）：驱动维护 HBM-resident 2MB 区域链表，最近迁移的放尾、最早迁移的放头，内存压力下换出链表头；区域内任一 64KB 页再 fault 时该区域晋升到尾部。由于驱动对 HBM 内访问无观测，LRM 常换出 GPU 正在活跃计算的区域（如矩阵乘中全生命周期都在访问的矩阵 B），造成反复页错误与迁移。ObservUVM 用采样可观测性实现三种近似策略 + 一个 meta-policy：
  - 近似 LRU：维护链表（头=最久未访问、尾=最近访问），page fault 或 access counter 通知都把区域 move_to_tail，换出链表头；观察链表头附近最多 100 个未观察区域。
  - 近似 LFU：按估计访问频率分 bin，区域按 fault 流和 access counter 通知在 bin 间晋升，换出最低频 bin 的头部；适合访问频率偏斜的应用（SRK、SR2、GMV、LU、SPM）。
  - Cyclic Protection（CP）：链表分 protected（头部，不换出）与 unprotected（尾部，从这里换出），用 observability 在线调整保护区大小——若 unprotected 头部区域被访问则保护区太小、扩大，若观察页换出前未被访问则缩小；适合大复用距离的循环访问应用（2DC、BLK、BFS）。
  - Tournament meta-policy：同时运行全部策略，round-robin 选一个策略出换出候选，记录 CauseMap{区域→策略}；被换出区域若再次页错误则给对应策略记 blame（BlmPts），总 blame 超阈值 T 时退役 blame 高于平均 20% 的策略（Algorithm 1）。实测无单一策略通吃：14 应用中 5 个偏好 LRU、6 个偏好 LFU、3 个偏好 CP；Tournament 逼近各应用最优策略，仅 ~2% 开销。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 ObservUVM 运行时中的流程（以 LRU 为例，Listing 1 伪代码）：onPageFault(addr) → lru_list.move_to_tail(addr)；onAccessCounter(addr) → move_to_tail(addr)（保护活跃区域）；onEviction(addr) → remove(addr)；setEvictionCandidate() → return lru_list.head；setObservabilityCandidate() → 返回链表中第一个未观察节点。驱动在每次 page fault/access counter/eviction 事件时调用这些回调，得到换出目标与观察候选，强制执行。Tournament 的 meta 逻辑：EvictionRequest 时 choosePolicyRR(A) 选策略出候选并记录 CauseMap；PageFault(region) 时 blamePolicy=CauseMap[region]、BlmPts[blame]++，ΣBlmPts>T 时对每策略按 BlmPts/ΣBlmPts>1.2/|A| 退役。
- 效果：LRU 使 MM/GMM/HEL 换出降 62%、执行时间平均降 14%（最高 20%）；CP 使 2DC/BLK 换出降 46%、时间降 34%；LFU 使 SRK/SR2/GMV/LU/SPM 平均提速 16%（最高 58%）、换出降 46%；TM 平均减少 40% 页错误；TM++（含预取）较 UVM 平均提速 34%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 userspace 策略对象（C++11），继承 EvictionPolicy 基类实现全部虚函数（onPageFault/onAccessCounter/onEviction/setEvictionCandidate/setObservabilityCandidate），引擎按事件循环调用。使用：deploy ObservUVM 后按工件说明注册策略、运行 run_key.sh 复现 fig10/fig11（不同换出策略的执行时间与 eviction 数）。与 LLM serving 中 KV cache 逐出策略（如 LRU/FLOP-aware eviction）同类——都是"容量受限缓存下的替换策略"，但作用对象是 GPU UVM 的 2MB HBM 区域、决策信号来自硬件 access counter 采样。

涉及论文标题：
- Observability-aided GPU Memory Oversubscription
