## SPECrate 与 SPECspeed（同质多拷贝吞吐率 vs 单任务完成时间）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SPEC CPU 套件的两种运行/报告风格。SPECrate®（如 SPECrate®2026 Integer/Floating Point）测吞吐：同时运行 M 个 benchmark 副本（copy），报告"单位时间内完成的作业数"，属弱扩展（weak scaling，工作随 copy 数增长，Gustafson's Law）；SPECspeed®（如 SPECspeed®2026 Integer/Floating Point）测单任务完成时间：一次运行一个 benchmark 实例（其中 22/26 个为多线程 MT，用 OpenMP/std::thread/DO CONCURRENT/task-based 进程并行），报告最短完成时间，属强扩展（strong scaling，固定工作分摊到多处理器，Amdahl's Law）。两者共享同一套 benchmark 源码与输入（ref/test/train 三种大小），只是运行与评分方式不同；SPECrate 的 footprint 为每 copy 2GB（CPU2026 不变），SPECspeed large 多线程套件 footprint 从 CPU2017 的 16GB 增至 64GB 以匹配现代内存容量。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
两种风格对应两种系统容量建模：SPECrate 回答"系统并发能扛多少同质负载"（多租户吞吐容量），SPECspeed 回答"单任务能被多核加速多快"（并行扩展性）。运转流程（以 intrate 为例）：
```
SPECrate（弱扩展）: runcpu --rate --copies=48 727.cppcheck_r
  → 48 个同程序 copy 并发, 每 copy 跑固定 ref 工作, 全部完成后
    每 copy 时间 → SPECratio = 参考机时间/本机时间 → 各 benchmark ratio 几何平均 = 总分
SPECspeed（强扩展）: runcpu --speed 823.llvm_s
  → 单实例(多线程)跑 ref 工作, 线程数/进程数可配置, 测最短完成时间
    → SPECratio = 参考机时间/本机时间 → 几何平均 = 总分
```
SPECrate 强调弱扩展/容量（CPU 数翻倍、工作翻倍、时间保持恒定），SPECspeed 强调强扩展/延迟（CPU 数翻倍、固定工作时间减半）。CPU2026 中 13 个 fpspeed 全部并行、13 个 intspeed 中 9 个并行；首次加入两个 task-based 并行 benchmark（821.gcc/823.llvm，模拟 make -j N：数千条命令行逐条 spawn 编译器进程）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：runcpu 用 `--rate`/`--speed` 选择风格，`--copies` 控制 rate 的副本数、SPECspeed 的线程/进程数由 benchmark 自身配置（OpenMP 线程、std::thread、DO CONCURRENT、task spawning）。使用场景：处理器/编译器性能评估——rate 用于吞吐容量对比（多核服务器）、speed 用于单核/单任务性能对比（延迟敏感负载）；R 后缀（_r）是 rate 版 benchmark、S 后缀（_s）是 speed 版。SPECrate 的 raw 输出新增统计量（变异系数、四分位、min/max/平均 copy 时间、标准差）支持更严格的多副本分析。官方文档：https://www.spec.org/cpu2026/。

涉及论文标题：
- SPEC CPU: The Next Generation
