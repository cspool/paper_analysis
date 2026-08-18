## Stream Flipping（无 pin DBI：单周期宽 µbump 接口上的架构级数据总线翻转）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stream Flipping 是 Raptor 提出的"pinless DBI"（无 pin 数据总线翻转）：在 3D-DRAM 逻辑-die 与 DRAM-die 之间的单周期宽 µbump 接口（每 µbump 每周期 1 bit、无 burst 结构、无专用 DBI pin）上，由内存控制器在架构层实现类 DBI 的位翻转编码，减少位跳变以降低 I/O 功耗。背景：JEDEC 内存（DDR4、HBM3）用多周期 burst 传输，PHY 级 DBI 看到整字后逐 beat 决定翻转并靠专用 DBI pin 传信号；Raptor 的 3D-DRAM 接口每周期传 256-bit 字、无 burst 无 DBI pin，失去了 DBI 依赖的 lookahead 与 sideband。3D-DRAM µbump 短垂直链路使 per-bit I/O 能量仅 0.45 pJ/bit（约 HBM3 的 1/6），但 100TB/s 聚合带宽把总开关功耗放大到 ~360W/card，降低位跳变至关重要。测量表明 DBI 等效编码可省 18% I/O 能量（0.45→0.376 pJ/bit），故在架构层实现。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Stream flipping 利用 Raptor 的两个性质：权重与 KV cache 以 stream-blocked tile 布局产生每 channel 长近单位步长 128B flit 流，且软件栈控制 tile 放置。写路径：控制器把每个 flit 与同 channel 上一个 flit 比较，选择是否翻转以最小化位跳变，每 flit 记录 1 bit metadata（与 [144,140] Reed-Solomon ECC 共置于 bank 末 8 列的 side region，仅 0.8% 容量开销）。读路径：控制器先取 ECC+DBI metadata 再取数据 flit，先做 ECC 纠正（如需）、再按 DBI bit 决定是否条件翻转后送入计算流水线。8 bank 最坏 100% 切换率下 0.455 pJ/bit → stream flipping 把有效切换降到 40-48% → 0.376 pJ/bit（-18%），无需改 DRAM PHY；1.1V array rail 占主动功耗 87%，功耗优化集中在 DRAM 阵列活动。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存控制器内建比较器（flit vs 前一 flit 的逐位 XOR 计数）+ 条件翻转逻辑 + 每 flit 1-bit metadata 读写，metadata 走与 stream-blocked 数据相同的映射（与 ECC 共置、每 bank 末 8 列）；写序为"算 ECC → 选翻转极性 → 写（可能翻转的）数据 → 提交 ECC+DBI metadata"，读序为"取 metadata → ECC 纠正 → 条件翻转"。使用方式：与 stream blocking 协同——对 128B flit 流式数据（权重/KV）逐 flit 编码，实测 18% I/O 能量节省（比 HBM3 低 ~6×），在 422W/MCM 热预算内支撑 100TB/s 级带宽；是"架构级无 pin DBI"的实例，无需 PHY/DRAM 改动。

涉及论文标题：
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
