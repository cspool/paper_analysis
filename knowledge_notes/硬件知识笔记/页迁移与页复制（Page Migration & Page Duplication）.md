## 页迁移与页复制（Page Migration & Page Duplication）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
多 GPU UVM 中把远端页变为本地页的两种策略。迁移：请求 GPU 远端访问后，驱动把页搬到请求 GPU——此后请求方本地且可缓存，源 GPU 变远端（单属主；多 GPU 共享时产生乒乓，用访问计数阈值如 Volta 的 256 延迟触发来缓解）。复制：远端读触发把页复制到请求 GPU，多个 GPU 各自本地缓存访问，但写必须广播给所有副本（远程更新），传统方案首个写即去重。两者都要先 flush 属主 GPU 的在飞指令/cache/TLB 再传输。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
远端访问 → far-fault → 驱动决定迁移或复制 → flush（SM 在飞指令 + cache + TLB）→ 页传输 → 双方页表/TLB 更新 → 重放访问。代价主要是管理开销：TLB 失效、SM 流水线 flush、cache flush 与小页传输的低带宽。论文实测 GPS/GRIT（4KB/64KB 细粒度）复制/迁移开销占执行时间 37%/38%，平均迁移带宽利用率仅 0.1 GB/s（NVLink 8 GPU 下 4KB/64KB 传输仅 1.12/17.12 GB/s）——这是 CDFD 改用 32MB 粗复制的动机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NVIDIA UVM 原生支持迁移与 read-mostly 复制（cudaMemAdviseSetReadMostly：读访问生成只读副本、写时失效）。论文对比的 SOTA：GPS（订阅式复制 + 批量远端写更新）、GRIT（on-touch 迁移 + 访问计数迁移 + 复制三策略动态选择）、CoarseDup（消融：仅 32MB 粗复制 + 远端更新）。CDFD 改为 32MB 粗复制 + 2MB/64KB 细去重的 duplication-centric 方案。

LIBRA 补充视角（ISCA'26，多 GPU 页面迁移的预测式/反应式分类与成本收益）：多 GPU 页面迁移分两类——反应式（on-touch、access-counter-based（阈值 256）、page duplication 三策略，GRIT 为 SOTA 动态按页选择）与预测式（即页面预取：TBNP 系/Forest/HOPP）。LIBRA 指出反应式迁移全在 critical path（GRIT 花 36% 执行时间在迁移），预测式空间局部性预取器精度低（Forest accuracy 42%）且无成本收益/协调。LIBRA 成本收益公式式(2)：lat_remote*(acc_highest-acc_source)>page_migration_overhead——acc 为各 GPU 对该页的估计未来访问数（硬件访问计数反馈），lat_remote 为跨 GPU 平均远程访问延迟，page_migration_overhead 为迁移总延迟；模拟器标定一次迁移开销≈200 次远程访问（与 NVIDIA 阈值 256 一致），迁移必须减少 >200 次远程访问才算有益。指标 remote access changes=每次迁移相对不迁移直到下次迁移事件所改变的跨 GPU 远程访问总数。
ShadowUpdate 补充视角（ISCA'26，迁移处理机制本身的开销量化与重设计）：ShadowUpdate 用两 RTX 3090 Ti（PCIe Gen4，NVIDIA 开源驱动）实测五步迁移处理的延迟分解：invalidation 占 30.35%、map new page 占 26.02%（mapping 只做在 destination GPU 上），并比较了映射更新的两种实现——software-driven（host 下发请求、各 GPU 走查装映射）vs hardware-based（GPU 硬件处理走查与更新）：hardware-based 的 L2 TLB miss 延迟在 4 GPU 时低 2.37×、性能高 1.38×，且 GPU 越多优势越大（software-driven 在 host 侧集中争用），故 baseline 采用 hardware-based。ShadowUpdate 把 mapping 步骤并入 invalidation（overlap 后迁移处理延迟降 22.79%），这是与 LIBRA/GRIT（选策略、预取）不同的"改迁移机制本身"路线。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
