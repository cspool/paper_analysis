## Top-down 分析（front-end bound / back-end bound / bad speculation / retiring）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-down 性能分析方法（Yasin, MICRO'14）把流水线 slots 按瓶颈归为四类：front-end bound（取指/译码供给不足导致流水线饿死）、bad speculation（误预测与错误路径冲刷浪费的 slots）、back-end bound（执行端口/访存层次资源受限）、retiring（有效退役，理想情况应占绝大多数）。Bumper 用它刻画移动应用（Fig.2）：平均 41% 的 slots 因前端而停顿，说明前端是当代移动应用的首要瓶颈（对比服务器 15–30%），这是 Bumper 的动机基石。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
分解流程：从 retired slots 出发，先分 retiring vs wasted；wasted 再二分 bad speculation 与"非推测性浪费"；后者再分 front-end bound 与 back-end bound。Bumper 的逻辑链：41% front-end bound ← 取指频繁等 L2C miss ← 指令 L2C MPKI 高达 8.4 ← 错误路径 (pre)fetch 使 useless 行占 20.3% L2C ← BTB miss 引发的 FTQ 重定向。修复前端 bound 的手段包括提高取指供给（BTB/预测器）、预取覆盖（FDIP/DJOLT）与本文的缓存容量管理（Bumper）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖性能计数器事件（如 FRONTEND_RETIRED.LATENCY_GE_*、BR_MISP_RETIRED、CYCLE_ACTIVITY.STALLS_* 等）在硬件/模拟器中采集，按 Yasin 的公式逐级分解。Bumper 论文在工业级模拟器中实现该分解并输出每应用四类占比。
- SPEC CPU2026 的套件级 Top-down 使用（论文 §VII-A 与附录 B/C，AMD EPYC 9755 单 copy / 128 线程）：对全部 intrate/fprate/intspeed/fpspeed benchmark 输出 IPC + Frontend/Backend/Lost/Retiring 四类占比表（Table V-VIII）。结论：整数套件前后端瓶颈较均衡（前端受限代表 727.cppcheck Frontend 0.57、753.ns3 0.54、714.cpython 0.53），浮点套件更一致地后端受限（765.roms Backend 0.58、749.fotonik3d 0.56、800.pot3d 0.64）；压缩类（777.zstd、731.astcenc）lost 比例最高（0.14/0.17，控制流不规则导致投机浪费）；750.sealcrypto/772.marian 高 IPC（5.23/5.33）且 lost≈0。这些特征化数据直接支撑 CPU2026 的选型（新增前端受限整数负载）与多 workload 多样性论证。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
- SPEC CPU: The Next Generation
