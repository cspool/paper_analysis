## Rolling Round-Robin Rate（RRR，滚动轮询率：异构多程序负载标准调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RRR 是 SPEC CPU®2026 新增的基准运行风格（run style）：把 SPECrate 套件（如 14 个 intrate benchmark）当作异构多程序负载来运行的标准调度方法，处于 exhibition 状态（评分方法未定稿）。核心规则：M 个核（copies）各自持有一条包含全部 N 个 benchmark 的队列，队列起点按"进程号 mod N"错开、按固定步长 --rrrrate_inc 轮转（默认 1），每个核顺序跑完 N 个 benchmark、可重复 --iterations 轮。机制保证：每个 benchmark 在每个核上恰好各跑一遍，任何 benchmark 的指令总量在所有核上完全相等——从用户态 CPU 执行角度看"等量工作"，从机制上消除 sample imbalance（各程序独立运行时长不同造成的不均衡），只保留 schedule imbalance（并发争用造成的非对称干扰）。因为 benchmark 经过可移植性硬化（去随机化、去环境查询、去系统调用干扰、≥95% 用户态），RRR 负载的干扰完全来自用户态 CPU 执行，可公平研究多核争用、OS 调度与资源划分。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
RRR 是系统级多程序负载生成与调度架构：多请求（benchmark copy）→ 队列构建 → 并发调度 → 共享资源争用 → 逐 benchmark 统计。运转流程（runcpu 文档 + 论文 Figure 2）：
```
输入: runcpu --rrrrate -c <cfg> --iterations=2 --copies=4 <4 个 benchmark>
队列构建: 每 copy 一条队列(含全部 N 个 benchmark), 起点 = 进程号 mod N
  copy0: B0→B1→B2→B3 | copy1: B1→B2→B3→B0 | copy2: B2→B3→B0→B1 | copy3: B3→B0→B1→B2
并发调度: M 个进程同时启动、独立运行不同步 → 在共享 L3/DRAM/ITLB/预取器上产生真实异构争用
统计: 每 benchmark 先取各 copy 内迭代中位数 → 再对 M 个 copy 中位数取均值(时间) + CV
输出: 无官方 SPECratio/geomean; 每 benchmark 报 time_avg/time_cv/四分位等 RSF 字段
```
作用：以标准化、确定性、可复现的轮转调度产生异构多程序负载，替代研究界"自定义 benchmark 混合 + ad-hoc 调度策略"的碎片化现状（论文引 FIESTA [165] 与多程序指标文献 [161-164]）；使多核争用/OS 调度/资源划分研究可直接跨研究对比。RRR 保留"unwanted cross-effects"（完整执行剖面下的真实干扰），优于用手工 kernel 隔离代替完整应用的做法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：runcpu 工具集新增两个标志——`--rrrrate`（启用滚动轮询 rate 模式）与 `--rrrrate_inc[=N]`（队列步长，默认 1；0 表示只跑初始 benchmark 的快速验证模式）；`--copies`、`--iterations` 复用既有参数。RRR 模式下 copies 由 runcpu 直接 fork（非 specinvoke），`$SPECCOPYNUM`/`$BIND` 由 runcpu 展开使既有 submit 命令无需改动；结果校验推迟到独立并行阶段（每 copy 一个进程）；每 benchmark×copy×iteration 各建一个运行目录；RRR 模式不报能耗字段、`--minimize_rundirs` 不兼容。配套 `scripts.misc/rate_timeplot.py` 生成 refrate（同质）vs rrrrate（异构）时间图。使用场景：异构多核吞吐研究、OS 调度策略评估、资源划分方案评估、多程序性能/公平性指标研究（累计 IPC、平均吞吐、调和平均、公平性指数等）。官方文档：https://www.spec.org/cpu2026/docs/runcpu.html（section 1.8）。

涉及论文标题：
- SPEC CPU: The Next Generation
