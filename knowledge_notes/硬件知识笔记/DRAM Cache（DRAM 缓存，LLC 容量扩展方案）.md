## DRAM Cache（DRAM 缓存，LLC 容量扩展方案）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DRAM cache 是用 DRAM 阵列实现的大容量缓存（作为 LLC/L4 位于片上 SRAM 与主存之间），核心动机是 DRAM 的高集成密度：与 6T-SRAM 相比面积小得多，可在等面积下提供数倍容量（TDMSim 中 2D-1T1C 在 32MB SRAM 面积内做到 512MB；硅 1T1C 亦 4× 起步）。代价是 DRAM 电容电荷泄漏要求周期 refresh：refresh 既消耗额外能量，又使正在刷新的 bank 无法服务普通请求（Access Interference）。商用先例：Intel Sapphire Rapids die-stacked eDRAM L4、Intel 至强 eDRAM L3；学术主流是 die-stacked DRAM cache（Loh & Hill 的 LH-cache、TDRAM/NDC/BEAR 等优化）。TDMSim 论文把 DRAM cache 从硅扩展到 2D 材料（MoS2 晶体管 1T1C/3T0C cell），利用 2D 材料超低泄漏把刷新周期从硅的 64ms 延长到 0.5s（2D-1T1C）/0.1s（2D-3T0C），大幅削减刷新能量与干扰。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 论文中 DRAM cache 采用 LH-cache 组织（见 LH-cache 条目）：16-way、16 bank，单个 DRAM row 构成一个 cache set，row 内 cell 分区为 tag ways 与 data ways。一次访问的运转流程：内存控制器先发 activation+read 命令把整 row 载入 row buffer → 从 tag ways 读 tag 判定命中 → 命中则保留 row buffer（防被其他请求关闭，保证后续命中为 row buffer hit）→ 从 data ways 返回数据。refresh 按行周期执行：Silicon-1T1C 每 64ms、2D-1T1C 每 0.5s、2D-3T0C 每 0.1s（约 20× 实测最小 retention 裕量）。system 级评估中（gem5 MI300X 单 XCD），9 种配置对比显示：2D-1T1C 128 平均 speedup 28.8%、能耗较 Si SRAM 32 降 77.8%；Si-1T1C 128 因频繁 refresh 在多个 workload 上反而降速（访问干扰），且能耗约 2×（刷新+静态主导）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：片上/die-stacked DRAM 阵列 + DRAM 控制器（refresh 调度、行缓冲管理、FR-FCFS 等）+ cache 管理逻辑（tag 组织、替换、MissMap 等）。评估用周期级模拟器（gem5 全系统 + 修改版 CACTI/TDM-Memory 提供延迟/能量/面积参数）。TDRAM [63] 用 in-DRAM tag 探测重叠 tag 检查与数据传输；NDC [35] 在子阵列内原生完成 tag 匹配/way 选择/替换；BEAR [64] 协同优化 miss 处理与 tag 探测降低带宽开销；TDMSim 证明 2D 材料收益与这些优化正交（与 TDRAM 结合再 +5% 性能、-15.6% 能耗）。使用要点：刷新周期选择 = 最弱 cell retention × 安全裕量（保守则刷新频繁、激进则数据丢失风险）；2D 材料用更小存储电容 + 更低泄漏维持长 retention，从而在"密度-延迟-能量"三维空间全面优于硅 DRAM。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
