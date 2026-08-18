## EVF / IRF 数据流（evk-flowing / intermediate-results-flowing）

术语解释
异构 FHE 加速器中 keyswitch 的两种数据流方案：EVF（evk-flowing，evk 流式）把整个 keyswitch 留在 xPU、evk 预载片上（单体 ASIC 的做法，evk 可复用）；IRF（intermediate-results-flowing，中间结果流式）把 IP 映射到近存 xMU、ModUp 输出经 HBM 送到 xMU 做 IP、结果回传 xPU 做 ModDown（省 xPU 内存、用近存带宽，但每次 keyswitch 增加两次 xPU-xMU 传输且落在关键路径）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- EVF：keyswitch 全部在 xPU 执行，evk 从 HBM 预载到片上 SRAM 供 IP 复用（单体 ASIC 的标准方案）。优点：evk 可重用、预载隐藏 off-chip 访问；缺点：程序出现低 evk 复用的串行 keyswitch 时，性能受大 on-xPU 内存限制（SHARP 需 180+18 MB）。
- IRF：IP 移到 xMU（避免 evk 上片），ModUp 输出流到 xMU 做 IP、结果回 xPU 做 ModDown。优点：消除大 evk 存储、利用 xMU 近存带宽，且 hoisting 后中间结果复用率提升、通信频率下降；缺点：每次 keyswitch 增加两次传输（ModUp→IP、IP→ModDown），中间密文（最高 144 MB）在关键路径上。hoisting 对两种数据流影响相反：EVF 下 evk 复用下降导致 off-chip stall 上升（性能下降），IRF 下中间结果复用上升（性能提升）——这使 IRF 成为 hoisting 的受益场景。hybrid 方案按 PKB 的 IP 并行度选择：并行 keyswitch（IP 并行度>1）用 IRF，单 keyswitch 用 EVF（预载 1 个 evk 的开销小于搬中间结果）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 两种数据流的 keyswitch 对比（同一 IP）：
```
EVF（xPU 内完成）:  evk 预载 SRAM → ModUp → IP(on-xPU, 读 evk) → ModDown
IRF（xPU+xMU）:    ModUp(xPU) ──HBM──► IP(xMU, 近存带宽) ──HBM──► ModDown(xPU)
```
- Annotations：EVF 的 IP 每次从 SRAM 读 evk（复用高则摊薄）、IRF 的 IP 从 row buffer 读操作数（近存带宽高）；IRF 的关键路径 = 计算 + 2 次 1 TB/s 传输，需双级流水 xPU 用组间计算-通信重叠隐藏；hybrid 需额外 84 MB 片上存 1 个 evk（HE²-LM），HE²-SM（44 MB）只支持 IRF。通信量结论：IRF 对高并行 PKB 显著降通信（中间结果复用），对低并行 PKB 通信高于 EVF（单 evk 加载更便宜）——hybrid 综合最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：数据流由调度/映射决定——HE² 的 HERO 框架把优化后 DFG 按 PKB 并行度映射到 IRF/EVF（IRF：ComOps→xPU、MemOps→xMU、PKB 间 EWO→xPU；hybrid：IP 并行度>1 用 IRF 否则 EVF）。硬件侧 IRF 依赖 xPU 双级流水（边算边流式写出 ModUp 输出、ModDown 收到即算，只需 44/84 MB 部分缓存）；EVF 依赖 evk 预载通道。用途：CKKS keyswitch 数据流选择；结论（Fig. 15）：IRF 使高并行 PKB 通信大幅下降，配合 HERO 的 PKB 融合（高并行化）让 IRF 全程高效，通信 stall 占比 68.2%→6.67%。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
