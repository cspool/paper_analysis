## Workload-Driven Flash Simulator（SSD 事件驱动 flash 模拟器）

术语解释
- 论文用于评估 LOONG 的事件驱动 SSD 模拟器：输入真实 I/O workload trace（到达时间戳、类型、逻辑地址、大小），模拟 SSD 控制器（FTL 映射、GC、pSLC/重编程调度）与闪存阵列（channel/chip/die/plane 四层并行），以实测芯片延迟参数累计每个操作的耗时，输出平均延迟（含排队与 GC 阻塞）、GC 频率/开销、95/99 百分位尾延迟、IOPS。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文称"a workload-driven flash simulator"（基于 [23]/[52]/[56] 的原始版本"显著增强"），未给出模拟器具体名称或开源链接；模拟器基于真实 TLC 芯片 [27]（Samsung 1TB 3b/cell 3D-NAND，>300 层，ISSCC'23）扩展参数，组织为 520 GB SSD。配置（Table IV）：8 通道、每通道 2 chip、每 chip 2 plane、每 plane 278 block、每 block 1280 WL、320 层、页 16 KB、传输率 1.2 GB/s、TLC OSP 写 1.1 ms、SLC 写 96 µs、pSLC 写 114 µs、重编程 955 µs、擦除 3 ms、LSB/CSB/MSB 读 25/50/100 µs。因为是事件驱动，排队延迟与 GC 阻塞时间天然计入端到端延迟。
- 从硬件架构角度拆解术语：模拟器把"SSD 控制器 + 闪存阵列"抽象为事件驱动性能模型，模拟的是设备端 I/O 处理全流程——一条写请求到达 → FTL 逻辑-物理映射（LOONG 扩展：查 RBP 定位重编程块、SP/RP 指针决定 pSLC 编程或重编程位置）→ 命中 pSP-GC 重编程块则 pSLC 编程（114 µs/页）否则标准 OSP（1100 µs/页）或触发 GC（有效页 pSLC 迁移 + 冷数据/bg-GC 数据参与重编程恢复容量）→ 按 channel/chip/die/plane 并行性派发 → 累计延迟 → 输出平均/尾延迟、GC 次数/成本、IOPS。论文对模拟器的主要修改：加入 pSLC 编程、长 stride 重编程（含读 pre-programmed 数据的开销）、基于实测芯片的程序/重编程延迟参数（保证可靠性导致的延迟开销已被计入）、FTL 扩展（RBP/SP/RP、reduced-state 编码 1 bit/页、validity table 检查）、两种重编程类型（E/P1→前 5 态 4/7 电压 ≈710 µs，或 E/P1→全 8 态）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用流程：加载开源生产服务器 I/O trace（MSRC/MSPS/FIU/YCSB，含到达时间戳、请求类型、逻辑地址、大小），事件驱动推进模拟，输出性能指标。论文用它评估：GC 优化（baseline/SP-GC/HS/8-stride/pSP-GC，各带/不带热度分离）、编程优化（baseline/TSP/Midas-Touch/TSP-LOONG）、扩展案例（弹性 SLC 缓存 vs 传统 SLC 缓存）、最坏场景（无冷数据、全热数据）与 GC 不敏感 workload。可靠性参数来自真机验证（FPGA 测试台 + 真实 3D TLC 芯片，1K P/E + 1 年保留期），模拟采用统一读延迟（理由：每 workload 内 block 最多 12 次 P/E、时长几小时、不足以触发 read-retry/校准，且 LOONG 电压分布与 baseline 相似）。开源情况：论文未提供模拟器/artifact 链接，web 搜索未找到，无法确认。

涉及论文标题：
- LOONG: Utilizing Long-Stride Reprogramming to Enhance the Performance of SSDs
