## Morton-Code-Based Page Table System（Morton 码页表系统 / Morton-Ordered Builder）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Morton-Code-Based Page Table System 是 NS-FPS（ISCA'26）加速器中按 Morton 码组织、存储点云的三级页表式存储系统，受计算机体系结构内存管理（页表/按需分配）启发设计，应对真实点云在 Morton cube 间分布稀疏且极不均匀的问题（部分 cube 空、部分 cube 上千点）。三级结构：(1) **Occupancy Table**（片上 SRAM）——每个 Morton cube 一个 14-bit 项：1-bit 非空标志 + 13-bit 指向 Page Table 项；(2) **Page Table**（片上 SRAM）——存每 cube Page Memory 区间的 start/end 地址（28-bit）+ 4-bit 末页点数；(3) **Page Memory**（DRAM）——每项存 16 个点的 (x,y,z) 坐标与缓存距离 + 14-bit 指向同 cube 下一页的指针。配合 32 项片上 Morton Cube Buffer 缓存最近访问 cube，形成"SRAM 页表 + DRAM 页存储 + 片上 cube 缓冲"的三级内存方案。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - **Morton-Ordered Builder**（预处理阶段）四级流水：读点 p(x,y,z) → 坐标量化 15/15/11-bit → 取 7/7/3 MSB → 交织成 17-bit Morton 码 → 查 Occupancy Table（新 cube 建 Page Table 项，否则复用）→ 分配/追加 Page Memory 项。单遍构建、无比较排序，按需分配使存储只覆盖实际含点的 cube（120k 点场景可直接以 14/28/14-bit 紧凑表示）。**Cube Buffer 优化**：32 项（32 块）片上缓冲缓存最近 cube，命中直接写入缓存项、miss 从 DRAM 载入（LRU 驱逐、脏项回写），处理完所有点后最终 flush 提交脏 cube，兼顾空间局部性与 DRAM 流量。
  - 运转例子（邻居查询）：Neighbor Search Engine 发出 Morton 码 → 查 Occupancy Table 得 Page Table 项 → 取 Page Memory 区间（跨页沿 14-bit 链表追取）→ 16 点并行算距；同一 cube 的再次访问命中 Cube Buffer，避免重复 DRAM 读。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Verilog RTL + 28nm Synopsys 综合 + DRAMsim3（DDR4-2400）；页表/Occupancy/Record Table/Cube Buffer 在片上 SRAM（SRAM 共 1.907mm²、占芯片 92% 面积、307.38mW），Page Memory 与 Point Buffer（原始点 + 已采样点）在 DRAM（2066.69mW、占总功耗 82.7%，凸显减片外流量价值）。该设计把通用 OS 页表概念移植到点云加速：按 Morton cube 而非虚拟页分页、以 cube 为空间局部性粒度。论文未开源 RTL；算法对应物（桶排序 + 索引表）开源在 https://github.com/satreeby/ns-fps/。预处理（Morton 排序 + 页表构建）对 120k 帧仅 1.02ms、占端到端 <10%。

涉及论文标题：
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds
