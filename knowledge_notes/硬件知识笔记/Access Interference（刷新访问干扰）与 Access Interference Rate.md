## Access Interference（刷新访问干扰）与 Access Interference Rate

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Access Interference 指 DRAM 阵列在刷新（refresh）操作期间，被刷新 bank 无法服务普通访问请求，导致这些请求被阻塞/排队、平均访存延迟上升的现象。它是 DRAM cache 相对 SRAM cache 的固有性能代价：刷新必须周期性访问 bank（对每一行重写电荷），执行期间该 bank 的数据通路被占用。论文定义 access interference rate = 映射到正在刷新的 DRAM cache bank 而无法被服务的请求比例。相关研究与证据：eDRAM LLC 研究中刷新对性能影响虽小但能量占比最大（UMD "Refresh Matters"）；Kong 等的 selective fine-grain round-robin 刷新（2017）显示按行粒度轮转刷新可减少 bank 冲突、性能 +7.3%/能耗 -13.3%；TDMSim 中 Si-1T1C 128 的访问干扰率显著高于 2D-1T1C 128（后者长 retention 使刷新极低频、干扰近零），尽管二者 cell 访问延迟相当，Si 方案仍因干扰整体降速。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（gem5 MI300X 系统，LLC 为 128MB DRAM cache）：GPU kernel 的访存请求按地址映射到 16 个 bank 之一 → 若目标 bank 此刻处于 refresh（控制器按刷新周期每行轮流刷新），请求进入等待队列直到刷新完成（Silicon-1T1C 每 64ms 一轮、行数多则刷新占用频繁；2D-1T1C 每 0.5s 一轮、占用稀少）→ 等待拉高该请求的完成延迟，累积为 workload 降速。论文用 access interference rate 量化：Si SRAM 32 与 2D 1T1C 128 RF（无刷新）为 0；2D 1T1C 128 近 0；Si 1T1C 128 明显更高。Retention-aware 策略再降干扰：refresh scheduling 把中心行刷新周期从 0.5s 延到 1.5s，hot-page remapping 把热页移出高刷新区，最终访问干扰率较无策略降 75.6%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 缓解手段分三类：(1) 延长 retention/降低刷新频率——器件层面（2D 材料低泄漏）、刷新调度层面（按行 retention 分级刷新、RAIDR 式 retention-aware refresh）；(2) 调度错峰——bank-wise round-robin 刷新、刷新暂停（refresh pausing）、把刷新安排到空闲窗口；(3) 数据放置——把热数据移出高刷新区（hot-page remapping）。评估时在周期级模拟器统计"请求遇到 bank 正刷新"的比例（access interference rate）与刷新能量占比。TDMSim 中该指标是区分 2D 与硅 DRAM cache 系统级优劣的核心证据。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
