## Access Counters（GPU 访问计数器）与 ACBM（Access Counter Based Migration，访问计数迁移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVIDIA 自 Volta 起在 GPU 中提供访问计数器（access counters），用于跟踪 GPU 经 PCIe 对 CPU DRAM-resident 页的访问，是 ACBM（Access Counter Based Migration）机制的基础。工作机制（ObservUVM 论文背景描述 + NVIDIA open-gpu-kernel-modules 的 uvm_gpu_access_counters.c）：从 HBM 换出到 DRAM 的区域仍映射在 GPU 页表上，GPU 继续经 PCIe 访问这些页；硬件在访问累计达到阈值（threshold，参数化 1-65535）时触发通知，驱动收到通知后把该"热"区域从 DRAM 迁回 HBM。跟踪粒度（granularity）可为 64KB、2MB、16MB 或 1GB，均在驱动加载时配置。设计目的是让"仅被频繁访问的区域"享受 HBM 高带宽，避免盲目迁移与 thrashing，且 DRAM 页仍可访问（不像 page-fault 迁移那样阻塞执行）。
- 关键限制（论文用微基准逆向出）：全 GPU 仅约 256 个硬件计数器（分配 p 个 2MB 区域、阈值设为 x、各访问 x 次，通知数随 p 线性增长至 256 后饱和，在多个 NVIDIA GPU 上一致）；用 256 个计数器跟踪 TB 级 DRAM 的"热度"杯水车薪；且阈值×粒度有约 25 万种配置组合，正确配置随应用变化（cuBLAS matmul 需低阈值如 1，rank 计算需高阈值）。因此 ACBM 在 UVM 驱动中默认关闭，论文实测其对部分应用（MM、GMM）甚至比默认 page-fault UVM 更差。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（ACBM）：HBM 超订 → 驱动把 2MB 区域换出到 DRAM ① 但仍映射 GPU 页表 ② → GPU 访问该区域 → 经 PCIe 的访问被 access counter 计数 ③ → 计数达阈值 → 硬件向驱动发通知 ④ → 驱动把该区域从 DRAM 迁回 HBM ⑤。ObservUVM 的复用（改变用途）：不再用它判断"DRAM 上哪些区域热以决定迁移"，而是把它当作"access bits"模拟器——把 HBM 上濒临换出的 2MB 区域中的一个 64KB 采样页迁到 DRAM 并 pin、映射 GPU 页表，GPU 对该采样页的 PCIe 访问（阈值=1）即通知驱动"该 2MB 区域正被 GPU 活跃访问"，从而提供对 HBM-resident 页访问的可观测性。硬件层面支撑：计数器按 PCIe 访问带宽（几十 GB/s 量级）计数可行，而按 HBM TB/s 级带宽跟踪 HBM 访问不现实——这正是计数器只能测 PCIe 侧访问的硬件原因。
- 工程证据（本地 vault）：human_notes/GPU架构笔记/NV各代GPU架构 记录 "Access Counter 缓存频繁的物理页" 与 "Hardware access counters allow delayed migrations over a page-fault-based method so that only hot pages are migrated"（score 176）；另有 GPGPU-Sim UVM Smart 仿真框架用于评估此类 UVM 迁移/预取设计（GPU Memory Manage 笔记）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 GPU 内的硬件计数结构 + UVM 驱动的软件处理（NVIDIA 开源驱动 open-gpu-kernel-modules 中 uvm_gpu_access_counters.c 实现 access counter 注册/通知处理），配置经驱动加载参数（granularity、threshold）。使用方式：正常 UVM 下驱动可选启用 ACBM 辅助迁移；ObservUVM 将其复用为 observability 原语（阈值固定为 1，只观察少数 key 区域而不观察 TB 级 DRAM）。除 UVM 外，访问计数思想也用于 GPU 内存虚拟化与 tiered memory（类似 CPU 侧 AMD IBS 采样 / Memtis 等 tiering 研究的硬件采样）。Web 证据：NVIDIA open-gpu-kernel-modules 仓库 uvm_gpu_access_counters.c（https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-uvm/uvm_gpu_access_counters.c）证实计数器的存在与配置项。

涉及论文标题：
- Observability-aided GPU Memory Oversubscription
