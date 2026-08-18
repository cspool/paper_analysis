## CDFD（Coarse-grained Duplication First, Fine-grained Deduplication Later）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CDFD 是论文提出的 duplication-centric 多 GPU 内存管理机制，两阶段：①粗粒度复制优先——GPU 发远端读时，把包含请求数据的 32MB 大页复制到请求 GPU，一次传输充分利用 NVLink（非线性延迟-大小、可忽略争用、充足带宽，4KB 与 32MB 传输延迟相当）；②细粒度去重后置——监视每个 2MB/64KB 子页的本地访问数与远端更新数，收益 = 本地访问数 − 远端更新数，对收益最低者细粒度去重。运行时维护 current/target duplication ratio：current 超 target 时先去重腾空间再做复制/常规页装载，否则 LRU 换出常规页；32MB 页去重过半后裂分为 1MB 页，继续监控 64KB 子项。硬件组件：duplicated TLB + sharer table、DDU + CDB、Access Count Monitor（共 12,800 B/GPU）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：远端读 far-fault → DDU 检查 current vs target duplication ratio → 超限则从 CDB 取收益最低的 2MB/64KB 子页去重（经 UVM runtime 通知其他 GPU 更新 sharer table、更新本地页表），腾够空间后复制 32MB 页；未超限则 LRU 换出常规页后直接复制。写重复页：L1 TLB 识别 1-bit 重复标志 → duplicated TLB（miss 查内存 sharer table）→ GMMU 向共享者发远端写更新（off critical path）→ CDB 更新计数。Access Count Monitor 周期向 CDB 同步本地访问计数（右移 1 位叠加，融合长/短期模式）。sys-scoped 写：fault → flush → 各副本合并为单一权威副本。效果：复制/迁移次数较 GPS/GRIT 降 >99%，组合开销（复制+迁移+远端访问）较 GPS/GRIT 降 92%、较 CoarseDup 降 58%；端到端性能 +66%/+65%/+8%（NVLink 4.0 下 +56%/+52%/+7%；8/16/32 GPU 增益稳定）；32MB 页占复制页约 91%、平均 duplication ratio 约 24%、coherence 广播约 68% 有用、额外功耗 +5.58 W（vs CoarseDup +5.74 W）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MGPUsim 4-GPU 配置上实现（duplicated TLB 32 项 1,152B、CDB 256 项 10,240B、ACM 256 项 1,408B），扩展 UVM runtime 的 far-fault 处理；NVLink 3.0/4.0 参数用真机 NCCL/cudaMemPrefetchAsync 实测注入；13 个 benchmark（AMDAPPSDK/Hetero-Mark/SHOC/DNN-MARK）与 2.5×/3× oversubscription 验证。开源代码论文未说明（无法确认）。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
