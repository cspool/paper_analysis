## UVM（Unified Virtual Memory，统一虚拟内存，含 far-fault）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UVM 是 CUDA 的统一内存管理模型：CPU 与所有 GPU 共享单一虚拟地址空间，程序用 cudaMallocManaged() 分配 managed memory，数据按需在 CPU/GPU 间自动迁移或复制，由 UVM 驱动 + GPU MMU 配合完成，免去显式 cudaMemcpy。GPU 访问无本地映射的地址时发生缺页：GMMU 页表走查失败 → far-fault 经中断报告 CPU 侧 UVM 驱动 → 驱动查集中式页表定位数据所在处理器 → 迁移/复制该页 → 更新双方页表并 TLB 失效 → 重启访问。论文把多 GPU 下的 far-fault 按"页在另一 GPU（远端读）"与"页在 CPU"分流处理。相关 API：cudaMemPrefetchAsync（预取整段区域，走完整 UVM 迁移路径：驱动排队、CPU/GPU 页表更新、TLB 失效——论文用它做 4KB–32GB 的 UVM 开销微基准）；cudaMemAdviseSetReadMostly（声明读多写少：任何处理器读访问即在本地生成只读副本=页复制，写时失效其余副本，cudaMemPrefetchAsync 对该区域生成只读副本而非迁移——CDFD 的 sys-scoped 写副本合并即仿此语义）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
请求路径：SM 访存 → L1 TLB → L2 TLB → L3 TLB → GMMU 页走查 → far-fault（中断到 CPU 驱动）→ 页就位后重放访问。硬件支撑点：GPU 页表 + GMMU 走查器、各级 TLB、L1 MSHR、Volta+ 每页 32-bit 访问计数寄存器（GMMU 在 TLB 查找时自动更新，UVM 用它做访问计数迁移，阈值如 256）、far-fault 中断通道。CDFD 中 far-fault 交 DDU 分流：远端读 → 32MB 复制或先细粒度去重再复制；CPU 页 → 常规页装载（必要时先去重腾空间）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 NVIDIA 驱动（open-gpu-kernel-modules 的 nvidia-uvm 模块）+ 硬件 MMU：Ampere UVM 页大小支持 512M/2M/64K/4K；Pascal+ 支持多 GPU 并发 managed access 与 read-mostly 复制。使用：cudaMallocManaged → cudaMemAdvise（SetReadMostly / SetAccessedBy / SetPreferredLocation）→ kernel 直接访问，驱动按访问计数与 advice 决策迁移/复制/驻留。典型场景：多 GPU 训练/推理、内存 oversubscription（footprint > 物理显存）。

LIBRA 补充视角（ISCA'26，多 GPU UVM 页面预取）：UVM 管理的离散多 GPU 系统（DGX、Intel Xe 等，GPU 经 PCIe/NVLink 互连）中，非一致性 NUMA 使远程访问成本高且取回的数据不能缓存（缺一致性），而页面迁移有数据搬移开销（TLB 失效、SM/cache flush、数据传输），故预测式页面迁移（即页面预取）成为关键。LIBRA 强调多 GPU 下三个新问题：(1) 空间局部性预取器（TBNP 系，为 CPU-GPU 设计）精度低（Forest accuracy 仅 42%）——工作负载跨 GPU 分区削弱空间局部性、并发访问同页造成争用；(2) 迁移成本-收益权衡被忽视——NVLink 低延迟下远程访问可能比迁移更优（模拟器标定一次迁移开销≈200 次远程访问，与 NVIDIA UVM 访问计数迁移阈值 256 吻合，仅 >200 才视为有益）；(3) 无协调导致 ping-pong（多 GPU 短时间窗内来回迁移同页）。LIBRA 采用 stride-based 预取 + 成本收益分析（式(2) lat_remote*(acc_highest-acc_source)>page_migration_overhead）+ CPU 侧 PPC 协调器；首次 CPU-GPU 迁移沿用 first-touch 默认策略（CPU 访问延迟高）。


LÆGIS 补充视角（ISCA'26，GPU 机密计算下的 UVM 页迁移）：LÆGIS 把 UVM 作为 GPU-based Confidential Computing 的核心机制研究其性能开销。其 UVM 流程：GPU 访存缺页 → GMMU 聚合 fault 到 fault buffer（fault batching，Bf 默认 256）→ 中断 CPU → nvidia-uvm driver ISR 取批、预处理（fault preparation）→ 服务批次时对每个 4 KB 页做 AES-GCM 加密（TDX 页先经 TME-MK 解密，再 CPU 软件经 Linux Kernel Crypto API 加密，实测 1.3 GB/s）→ 经 PCIe（64 GB/s）DMA 到 GPU → CE 解密写 HBM 明文。三大开销来源：(i) IV 随访问顺序同步递增，CPU-GPU 需紧同步协商 IV，加密在关键路径；(ii) driver 线程在批次间大量 true idle（平均 87%）与 fault preparation 期 false idle；(iii) kernel-space LCA 软件加密未并行化（1.3 GB/s）。UVM 内存组织为 2 MB VABlock（含 64 KB 基本块），TBNp 树状预取按叶粒度迁移，预取阈值 Pt=51%（默认）/1%（aggressive）。LÆGIS 的解耦 IV（IV Bank）使预加密可乱序进行，从而利用 idle 窗口消除关键路径加密。


ObservUVM 补充视角（ISCA'26，GPU UVM 内存超订下的可观测性换出/预取）：ObservUVM 指出 UVM 驱动对 HBM-resident 页的 GPU 访问"无观测"（GPU 无 access bits、无低开销访问监控工具）是 LRM 换出决策差与 TBP 预取保守的根因，通过复用现有硬件 access counters 提供采样可观测性（sampled observability）：策略把濒临换出的 2MB key 区域（默认 ≤100 个）设为 observable，驱动将该区域内一个 64KB 采样页迁到 DRAM 并 pin、映射 GPU 页表，GPU 对该页的 PCIe 访问触发 access counter 通知（阈值=1），驱动把"区域被 GPU 活跃访问"信号上抛给 userspace 策略，据此做 LRU/LFU/Cyclic Protection/Tournament 换出与 FDP/RGP 预取决策。机制留在驱动（修改 NVIDIA 开源 UVM 驱动 v525），策略移 userspace（C++11 引擎 + eBPF tracepoint 通信层）。平台：RTX 3090 24GB + Ryzen 7950X + PCIe 4.0，14 个 UVM 应用在 30%-70% 超订下平均提速 34%（几何均值），最高 64%（SPM）。
ShadowUpdate 补充视角（ISCA'26，UVM 迁移机制重设计消除 re-fault）：ShadowUpdate 聚焦 access counter-based migration（阈值 256）下的迁移处理机制本身。关键新发现：baseline 迁移五步处理（分配新页 → 广播 invalidation 清所有 GPU 的 PTE + TLB shootdown → 拷贝 → 只更新 destination GPU 页表 → 释放旧页）中，非 destination GPU 的 PTE 被清后不会获得新映射，导致 re-fault（占全部 page faults 的 73.59%）。ShadowUpdate 把新 PA piggyback 到 invalidation 广播上，各 GPU 在 invalidation 的 GMMU 走查中同步写新 PTE，再用 IfMT（UMPT + Cuckoo filter）在拷贝期间阻塞翻译保证正确性，从而免去 host 侧 ATS/IOMMU 参与 re-fault 路径（页错误总数降 73.83%，性能平均 1.40×）。与 ObservUVM/LÆGIS 侧重驱动/预取/加密不同，ShadowUpdate 是纯硬件迁移处理机制重设计（MGPUSim 建模）。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
- Observability-aided GPU Memory Oversubscription
