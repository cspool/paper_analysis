## Strong Scaling 与 Weak Scaling（强扩展与弱扩展：Amdahl / Gustafson）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
并行性能扩展的两种模型。Strong scaling（强扩展）：工作负载大小固定，通过把固定工作量分摊给更多处理器来缩短总完成时间——加速比受 Amdahl's Law 限制（S = 1/((1−P)+P/N)，P 为可并行比例、N 为处理器数，N→∞ 时加速比趋近 1/(1−P)），即串行部分成为不可逾越的上限。Weak scaling（弱扩展）：工作负载随处理器数同比例增长（每处理器工作固定），目标是"处理器多了、工作也多了，但完成时间保持恒定"——对应 Gustafson's Law（加速比 S = P·N + (1−P)，随 N 线性增长），反映真实应用"规模上去后反而更容易加速"的经验。SPEC CPU 的对应：SPECspeed（单任务时间）是强扩展场景，SPECrate（多副本吞吐）是弱扩展场景（Table III）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
两种模型回答不同的容量问题，运转流程：
```
强扩展 (SPECspeed, Amdahl):
  固定工作 W 分到 N 个处理器 → 理想 T = W/N → 实际 T = W_serial + W_parallel/N
  例: 823.llvm_s 编译固定源码集, 128 线程并行编译数千条命令行
      → 测"多核能把固定编译任务压到多快"
弱扩展 (SPECrate, Gustafson):
  每处理器固定工作 w, N 个处理器 → 总工作 = N·w → 理想时间恒定
  例: --copies=48 727.cppcheck_r, 48 个 copy 各跑完整静态分析
      → 测"系统并发扛 48 份同质负载的吞吐"
```
强扩展关注延迟（单任务多核加速比），弱扩展关注吞吐容量（多副本并发能力）；SPEC 用同一套 benchmark 源码支撑两种运行（_r 后缀 = rate/弱扩展，_s 后缀 = speed/强扩展）。CPU2026 中 22 个并行 SPECspeed benchmark 均为强扩展场景；SPECrate 为弱扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：强扩展靠 benchmark 内的线程/进程并行（OpenMP 3.0、C++ std::thread、Fortran DO CONCURRENT、task-based 进程 spawn——821.gcc/823.llvm 用 make -j N 式逐条 spawn 编译器进程），线程数可配置（论文附录 SPECspeed 特征化用 128 线程）；弱扩展靠 runcpu --copies 副本数（CPU2026 SPECrate footprint 保持每 copy 2GB）。使用场景：处理器评估中 rate 结果（弱扩展）用于吞吐容量对比、speed 结果（强扩展）用于单任务性能对比；研究多线程扩展性时（论文指出 800.pot3d/801.xz 表现出最多锁争用，鼓励深入分析线程扩展性与数据共享）。Amdahl/Gustafson 常被引用来论证加速比上界与"规模效应"。

涉及论文标题：
- SPEC CPU: The Next Generation
