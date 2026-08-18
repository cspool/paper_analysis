## Page Prefetching Coordinator（PPC，页面预取协调器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LIBRA 引入的 CPU 侧 UVM runtime 软件模块：定量评估所有 GPU 发来的预取请求（含 MMP 估计的未来访问数），基于全局成本收益分析协调跨 GPU 的预取/迁移决策，避免 ping-pong 与无效迁移。PPC 支持 CPU-GPU（first-touch 迁移，因 CPU 访问延迟高）与 GPU-GPU（定量协调）两种迁移。组成：PPC runtime + 按 hashed VPN 索引的 PPC hash table（每条目：36-bit 实际 VPN、recent-migration 位(1b)、recent-use 位(1b)、3-bit 当前 GPU ID、8 个 8-bit 跨 GPU 估计访问计数）。决策式(2)：lat_remote*(acc_highest - acc_source) > page_migration_overhead 则迁移到估计访问数最高 GPU 并给请求 GPU 预取 PTE；否则仅建 PTE 不迁移。周期全局维护：右移访问计数衰减历史、清 recent-migration/recent-use 位、回收冷条目（hash 表用现有 hashtable 管理，含冲突解决与动态扩容 [54]）。层次归属说明：PPC 是纯软件运行时模块（驱动级），最接近"系统架构"层的运行时协调概念；它是 UVM 页面迁移系统的一部分（UVM/GMMU/MMAT 等硬件相关术语见硬件架构层条目）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
协调逻辑（图 12）：收预取 VPN → 查 hash 表（无则新建条目）→ 页不在 GPU（CPU 内存）则迁移到请求 GPU；页在 GPU 则更新请求 GPU 估计访问计数、置 recent-use 位、查 recent-migration 位（已迁移则跳过）→ 式(2) 判断：估计收益 > 迁移开销则置 recent-migration 位并调用 uvm_api_migrate(.) 迁移到 acc 最高 GPU、给请求 GPU 预取 PTE；否则仅 uvm_va_block_map(.) 建 PTE。GPU 侧 UVM support 与 CPU runtime 均按需响应中断/事件（不轮询、不读 mmap 寄存器）。效果：LIBRA CPU 开销 3.2%（vs TBNP 系 0.4%）；消融显示 coordination 额外减少 13% 总迁移、35% 不必要迁移、45% 低效迁移、+6% 性能；over 95% 迁移减少远程访问、69%+ 有益（Forest 仅 12%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 CPU 侧 UVM 驱动/runtime 模块（纯软件、无额外硬件），复用既有 uvm_api_migrate(.)（真实迁移）与 uvm_va_block_map(.)（只翻译建 PTE）函数；hash 表条目按需创建（首次请求时初始化字段并记录实际 VPN）。多 GPU 场景下协调所有 GPU 的预取请求以抑制 ping-pong 并选择最优目标 GPU；multi-rack 扩展为 per-rack PPC（管理该 rack UVM 内存的预取请求）。该模式可推广：任何"多 Agent 预取/迁移请求 + 中央成本收益仲裁"的运行时协调器。论文未开源实现（无法确认）。

涉及论文标题：
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
