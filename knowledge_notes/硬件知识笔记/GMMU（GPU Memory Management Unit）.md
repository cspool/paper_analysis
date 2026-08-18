## GMMU（GPU Memory Management Unit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GMMU 是 GPU 的地址翻译硬件：把虚拟地址经多级页表走查翻译为物理地址，并处理 GPU 侧 TLB miss 与页表维护。离散多 GPU 系统每个 GPU 有自己的本地页表与 GMMU；UVM 下 CPU 侧驱动维护集中式页表，向各 GPU 下发最新翻译。论文建模配置：每 GPU 8 个共享页表走查器、每级走查 100 cycle。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
L3 TLB miss → GMMU 发起页表走查（逐级读页表项）→ 命中则把翻译装回 L2/L3 TLB 的 sub-entry；走查失败（页不在本 GPU）→ 触发 far-fault 交 UVM 驱动。CDFD 场景：写重复页时，GMMU 还负责把 duplicated TLB 提供的共享者物理地址转换为发往各共享 GPU 的远端写请求；Volta+ GMMU 在 TLB 查找时自动更新每页访问计数器，供 UVM 迁移决策与 CDFD 的 Access Count Monitor 复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 CPU MMU 同构的硬件页走查器 + TLB 填充逻辑，配合软件驱动（uvm 模块）完成远缺页处理。对程序员透明；性能上体现为 TLB 命中率与走查/失效延迟（论文建模 L1/L2/L3 TLB 查找 1/10/40 cycle，走查每级 100 cycle，GMMU 走查器数 8）。

LIBRA 补充视角（ISCA'26）：LIBRA 在 GMMU 流程上扩展——所有 L3 TLB miss 转发给 MMP 预取器学习访问模式（扩展 TLB-miss 元数据携带 source SM 信息，NVIDIA 默认未指定是否携带 [33][65]）；far-fault 记录进 GMMU MSHR 并触发中断到 CPU 侧 UVM driver；GMMU 按 PPC 协调结果解析原 far-fault 并迁移全部/部分/零个请求页。每页本地访问计数寄存器（GMMU 在 TLB lookup 时自动更新）被 LIBRA 复用为成本收益分析的估计依据：达阈值触发硬件中断写 ring buffer，CPU 侧 UVM support 调用修改后的 fetch_access_counter_buffer_entries(.) 比对 MMAT 的 monitored VPN 并回填计数（只改该函数一步）。配置同前（每 GPU 8 共享 walkers、100 cycle/级、page-walk cache）。


LÆGIS 补充视角（ISCA'26，GPU CC 下 GMMU 的 fault 聚合与批处理）：LÆGIS 在 GPGPU-Sim+UVMSmart 中建模共享 GMMU：GPU warp 访存经 SM TLB 未命中 → GMMU 页表走查（PTW）失败 → 触发 replayable fault，故障信息写入 GMMU 侧 fault buffer → GMMU 把同一时段的高并行故障（多 SM 的 warp 同时在不同 VA 上故障）聚合成批（Bf=128）→ 经中断报告 CPU 侧 UVM driver。LÆGIS 的观察：GMMU 调度某批次时并非服务全部 pending fault，而是跨多个批次拆分，批次间自然产生 idle 窗口（true idle）；driver 若能读取 fault buffer 内容（研究界逆向支持），即可把后续条目预取为预加密候选（Opportunity 2）。LÆGIS 的 IV Bank 索引复用页表遍历路径：用 PDE0 中 19 个未用位存 IV ID，
ShadowUpdate 补充视角（ISCA'26，GMMU 复用为映射传播引擎）：ShadowUpdate 让 GMMU 在 invalidation 阶段承担"一石二鸟"职责——invalidation 广播本就要求 GMMU 遍历本地页表清旧 PTE，ShadowUpdate 在该走查中顺带写入 piggyback 的新 PTE，因此不产生额外 page table walk 请求（PWQ queuing 延迟降到 baseline 的 0.76×，而 naive 广播为 1.25×）。同时把 IfMT（Cuckoo filter + UMPT）放在 L2 TLB 与 GMMU 之间做在途迁移翻译拦截；若 UMPT 满则暂停迁移，保证不丢在途页。GMMU 的 PTW/PWQ/PWC 参数（8 共享 PTW、64 项 PWQ、128 项 PWC、每级 100 cycle）与 host IOMMU（16 共享 PTW）共同构成 ShadowUpdate 的翻译争用模型。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
