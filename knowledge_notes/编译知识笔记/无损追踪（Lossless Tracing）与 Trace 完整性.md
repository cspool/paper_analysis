## 无损追踪（Lossless Tracing）与 Trace 完整性

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Lossless tracing（无损追踪）= 完整采集全部目标事件的 trace，不因降开销而采样、丢数据或有损压缩。BULLETTIME 将其限定为研究对象：采样（SimPoint 类）、周期性丢数据（Intel PT buffer 满丢包）、有损压缩虽然降低 tracing 开销，但损失 trace 完整性，无法完整代表应用行为——而行为失真的根源（不对称 I/O 延迟）在无损追踪下依旧存在且只会随计算/内存速度提升而加剧。
- Trace 完整性与开销的基本权衡（Web 证据：perf-book/perfwiki 与 ARM CoreSight 文档）：Intel Processor Trace 用硬件编码控制流包（<1B/指令、运行时开销通常 <5%），可重建完整控制流与时间戳，但数据率（编码约 100MB/s、解码后可达 GB/s）使其不适合长运行；Arm CoreSight ETM 在核执行旁硬件并行生成包（对核零开销），但 >10GB/s 的原始数据率受 ATB 带宽与 buffer 排出能力限制，溢出时丢包；HMTT 嗅探 DRAM 总线采集访存，开销低但漏掉 cache 命中的访问。这些方案都以"完整性换低开销"（或反过来），且未分析追踪延迟对应用行为本身的影响。
- BULLETTIME 的立场：行为失真与 tracing 开销/完整性正交——只要 trace 数据率超过存储带宽、I/O 延迟不对称，行为就会失真；time dilation 对任何开销水平的无损追踪都成立，其设计原则也可套用于上述硬件方案。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 追踪工具的数据管线（以 BULLETTIME 的无损访存追踪为例）：目标指令流 → 插桩/硬件编码（Pin analysis 例程或 PT/ETM 编码器）→ 每线程 buffer（2MB）→ buffer 满触发 flush → 落盘（O_DIRECT 直写或异步 write()）→ 后续离线解码/模拟。完整性要求贯穿每一级：不采样（每条访存都记录）、不丢包（buffer 必须全部落盘）、不有损压缩（可选 zstd 为无损）。BULLETTIME 对比的配置都在此管线上变化：Empty-Traced（只插桩不落盘）、Disk-Traced（异步 write() 落盘）、DynamoRIO drmemtrace（LZ4 无损压缩落盘）、BT-Comp（zstd -7 在线无损压缩 + time dilation）。
- 关键量化（论文 Fig.4/Table II）：访存 trace 数据率远超存储带宽——单线程每 20 条指令 1 次访存即超 SATA SSD（0.4–0.6GB/s），真实 benchmark 每 3 条指令 1 次访存（20–25GB/s）；GPU 配置（HBM3e 4.9TB/s vs InfiniBand 100GB/s）比值 49×。因此缓冲会被打满、异步 I/O 快速退化为同步，bursty 停顿不可避免。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：选择取决于研究目标——需要完整行为保真时用无损追踪（Pin 全量插桩、PT 全量解码、drmemtrace）并辅以 time dilation 消除失真；只做统计热点分析时可用采样（PEBS/LBR/SimPoint）。压缩是无损追踪的重要配套：BULLETTIME 用 zstd -7 在线压缩把 I/O 量降 ~10×、运行时间改善 >2×，同时把 Llama 推理的 CPU 利用率从 5% 提到 20%（tracing 是 I/O-bound，压缩用掉空闲算力）。信息缺口：论文未给出 zstd 压缩率之外的压缩前后 trace 体积绝对值。

涉及论文标题：
- BULLETTIME: Time Dilation for High-Fidelity Tracing
