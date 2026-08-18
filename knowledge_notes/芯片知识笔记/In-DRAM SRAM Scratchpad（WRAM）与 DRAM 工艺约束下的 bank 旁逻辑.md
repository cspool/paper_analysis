## In-DRAM SRAM Scratchpad（WRAM）与 DRAM 工艺约束下的 bank 旁逻辑

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
商业 PIM 产品在 bank 旁放置小块 SRAM scratchpad，桥接"片上计算逻辑与 DRAM 存储的最后一段延迟鸿沟"：UPMEM 每 DPU 有 WRAM 64KB（数据工作区）+ IRAM 24KB（指令），Samsung FIMDRAM 每两 bank 共享一个 SIMD MAC 单元并同样用 SRAM 缓冲做数据暂存与命令/指令内存。DRAM 工艺约束（论文 §III-C 引 [11][21][29]）：约 3 层金属（逻辑工艺 >10 层）、逻辑密度 1/10、pitch 4×、晶体管慢 3×——这迫使银行级逻辑极简（UPMEM 需 14 级流水才能上 500MHz、无 bypass/stall、乘法 32 周期）。而规则网格结构的 SRAM 宏只需约 3 层金属即可布线闭合，正好落在 DRAM 厂商已为每 bank 分配的布线资源内——这是 BAAP 选择"改造 scratchpad 而非加 ALU/改 subarray"的工艺论据。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
BAAP 的物理改动全部限定在 WRAM 宏内：6T → 6T push-rule（+1 根 wordline）、灵敏放大器差分↔单端切换、每列 1 个 AND 门 + tag 锁存；动态活动只发生在 SRAM slice 的位线翻转，避开 subarray ALU 或位线计算方案所需的长全局总线与驱动。可集成性论证：Jeloka 28nm 测试芯片已验证在线重配置；BAAP 再保守降频 350MHz（低于 UPMEM 500MHz）并按 Stillmaker-Baas 缩放方程把功耗悲观缩至 65nm 平面 CMOS，DLP 负载仍低于 150mW/DPU TDP——不产生新的热/功耗瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 subarray 级 in-situ 方案（Fulcrum、Sieve 改 DRAM subarray 本身）对比：BAAP 不动 DRAM mats。使用方式：scratchpad 容量-算力权衡——把 25%/50%/75%/88% WRAM 重配置为 AP，对应 VL=96/224/320/384、剩余 48/32/16/8KB 留给标量线程；这一比例空间正是论文设计空间探索（PrIM 上 UPMEM 性能在 50%–25% 容量间开始显著退化）的结论。可迁移性：任何带 SRAM scratchpad 的 PIM（如 FIMDRAM/AiM 类）都可同样 BAAP 化。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
