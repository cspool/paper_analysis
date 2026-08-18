## Sampled Observability（采样可观测性，ObservUVM 复用 access counters 模拟 access bits）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ObservUVM 提出的运行时机制：让 CPU 侧 UVM 驱动"看见"HBM-resident 页是否被 GPU 活跃访问，全程无需硬件修改。NVIDIA GPU 不提供 CPU 式 access bits，驱动只看到页错误流（HBM 外访问）而对 HBM 内访问完全失明。ObservUVM 把现有 PCIe access counters 从"判断 DRAM 热度以迁移"改为"观察 HBM 区域是否活跃"：策略把一个 2MB 区域设为 observable，驱动把该区域内一个 64KB 采样页迁到 DRAM 并 pin、映射 GPU 页表；GPU 访问该采样页（经 PCIe）即产生 access counter 通知（阈值=1），驱动把"该 2MB 区域正被访问"事件上抛给 userspace 策略。采样依赖 GPU 应用的空间局部性——论文实测 90%+ 的 2MB 区域在换出前其全部 32 个 64KB 子页都被访问过，故每区域采 1 页足够；并可按应用局部性动态增/减采样页数（默认观察 ≤100 个 key 区域，占计数器预算内）。
- 设计要点：观察对象是"濒临换出"的 key 区域（eviction 场景）或"刚被预取"的区域（prefetch 反馈场景）；采样页被访问后立即迁回 HBM 恢复高带宽访问，限制 PCIe 额外开销；用计数器做 observability 而非 migration 恰好规避其两大限制（256 个计数器不够跟踪 TB 级 DRAM、阈值难调）——只需观察少量 key 区域且阈值固定为 1。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 ObservUVM 运行时中的流程：userspace 策略经 setObservabilityCandidate(address) 选 key 区域 → 驱动把该 2MB 区域的一个 64KB 页迁移到 DRAM 并 pin、映射 GPU 页表 → GPU 持续运行访问该区域 → 活跃区域被采样页捕获：PCIe 访问 → 硬件 access counter 通知 → 驱动经 eBPF tracepoint 上行 onAccessCounter(address) 到 userspace 引擎 → 引擎调用策略回调（如 LRU 的 move_to_tail 保护该区域不被换出）→ 驱动执行策略决策。若被换出的区域随后又快速页错误（采样没抓到访问），驱动监控 100 个最近换出区域并计算"换出后又被访问"的比例，超过 0.5 则加倍采样页数提升分辨率，低于 0.8 未再访问则减半——自适应分辨率。
- 效果：LRU 近似策略减少 MM/GMM/HEL 的 eviction 达 62%；Tournament 组合策略在 14 个应用、30%-70% 超订下平均提速 34%（几何均值）。观察开销约 2.4%（采样页迁移）+1.8%（通信/执行）平均。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为修改的 NVIDIA 开源 UVM 驱动（v525）+ userspace C++11 引擎 + eBPF 通信层；用户策略实现 onPageFault/onAccessCounter/onEviction/onPrefetch 回调与 setEvictionRegion/setObservabilityCandidate/setFeedbackCandidate 等下行接口（驱动暴露的 API 见表 I）。开源 https://github.com/csl-iisc/ObservUVM。使用：部署 RTX 3090 + Ryzen 7950X（Linux 6.2 + libbpf）→ 编译 driver（compile_drivers.sh 生成 base/super 驱动）→ 编译 userspace（compile_userspace.sh + gen_configs.sh）→ run_key.sh 复现 fig9-13 → fig9-14.sh 出 csv。自定义策略：新建目录，继承 EvictionPolicy（换出）或 ShallowPrefetch/DeepPrefetch（预取）基类实现虚函数。

涉及论文标题：
- Observability-aided GPU Memory Oversubscription
