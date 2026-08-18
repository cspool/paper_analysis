## SPEC ratio 与几何平均（参考机归一化评分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SPEC ratio 是 SPEC CPU 的报告指标：每 benchmark 的 SPECratio = 参考机运行时间 / 被测系统（SUT）运行时间（rate 版还要结合副本数换算吞吐），>1 表示被测系统比参考机快。总分用各 benchmark SPECratio 的几何平均（geomean）汇总。SPEC 选择一台参考机（reference machine）在套件发布时实测各 benchmark 的参考时间与参考能量，用于归一化后续所有提交结果；参考机刻意选老硬件使现代机器分数 >1.0。CPU2026 参考机：Lenovo ThinkSystem HR330A，3.0 GHz Ampere eMAG 8180（ARMv8 aarch64，32 核，2018 年产品；Table II 列出历代参考机：VAX-11/780 → SPARCstation → UltraSPARC 系列）。关键性质：同一套件内任意两台系统的相对性能与参考机选择无关（比值相消的数学性质），因此参考机更换不影响系统间比较。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
归一化评分是系统性能评估的基准框架：原始运行时间（不可跨系统比较，因 workload 难度不同）→ 相对参考机的比值（跨系统可比的纯数）→ 几何平均（跨 benchmark 的单值汇总，对异常值不敏感、乘法对称）。运转流程：
```
参考阶段: 参考机(HR330A/eMAG 8180)实测每 benchmark 的 ref 时间 T_ref,i 与能量
提交阶段: SUT 实测 T_sut,i (rate: 每 copy 时间与副本数换算) → ratio_i = T_ref,i / T_sut,i
报告阶段: 总分 = geomean(ratio_1..ratio_N) = (Π ratio_i)^(1/N)
比较: 系统 A vs B → geomean(A)/geomean(B) 与参考机无关（T_ref 相消）
```
SPECratio 把"运行时间"这种有量纲、依赖 workload 的原始量转为无量纲可比分数；几何平均使各 benchmark 权重对数对称（1 个 benchmark 快 2 倍 = 2 个 benchmark 各快 1.414 倍）。CPU2026 增加两类报告类别：云平台裸金属官方提交（不再只是 estimated）与"vendor 支持 vs 社区开源编译器（GCC/LLVM）"区分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：runcpu 在每个 benchmark 运行后按配置文件中的参考时间计算 ratio，报告脚本（reports/ 下）汇总几何平均；RRR 模式不产生官方 ratio/geomean（exhibition 状态）。使用场景：处理器/编译器/系统横向对比（同套件内）、性能与能耗（Joules，可选指标）归一化、跨代性能跟踪（CPU DB 等纵向研究用 SPEC ratio 归一化微架构改进）。注意：官方可提交结果必须遵守 SPEC 运行规则（编译标志、线程数、OS 需文档化），但套件作为研究 harness 时配置空间不受限、只需记录透明。官方文档：https://www.spec.org/cpu2026/。

涉及论文标题：
- SPEC CPU: The Next Generation
