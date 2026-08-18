## SPEC CPU: The Next Generation（近似层次匹配：本文是 CPU 基准套件与异构多程序负载调度方法论论文，本层取其 RRR（Rolling Round-Robin Rate）异构多程序调度运行风格；"框架"为 SPEC CPU 的 runcpu 工具集而非 LLM Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 SPEC CPU 2026 新增的 RRR（Rolling Round-Robin Rate）运行模式：把原有 SPECrate 套件（intrate/fprate，N 个 benchmark）当作异构多程序负载来运行的标准调度方法。核心规则：M 个核（copies）各自持有一条包含全部 N 个 benchmark 的队列，队列起点错开（按进程号 mod benchmark 数取初始 benchmark，随后按固定步长 --rrrrate_inc 循环），每个核把整条队列顺序跑完 N 个 benchmark、重复 --iterations 轮。由此每个 benchmark 在每个核上恰好各跑一遍，任何 benchmark 的指令总量在所有核上完全相等——从用户态 CPU 执行角度看"等量工作"，从机制上消除 sample imbalance（各程序独立运行时长不同造成的不均衡），只保留 schedule imbalance（并发争用造成的非对称干扰）。
  - 实验比较：同机同核数下对比同质负载（refrate，每个 copy 跑同一个 benchmark 的经典 SPECrate 同质容量方法，自 1992 年沿用）与异构负载（rrrrate，所有 copy 并发滚动调度全部 intrate benchmark）的时间图/性能差异；以及不同 --rrrrate_inc 步长（默认 1，可自定义）对轮转编排的定制。Figure 2 给出 AmpereOne 系统 48 copies 下三种调度风格对比。RRR 仅处于 exhibition 状态（评分方法未定稿，SPEC 公开征求评测指标——累计 IPC、平均吞吐、调和平均、公平性指数等仍在学界讨论中），因此 RRR 结果不产生官方 SPECratio/geomean 可比分。
  - 硬件平台是什么，配置是什么。
  - 特征分析机：AMD EPYC 9755（Zen 5，2.7 GHz、最高 Boost 4.1 GHz，L1 32 KiB I + 48 KiB D、L2 1 MiB、L3 512 MiB，2.3 TiB DDR5-6400），GCC 15.2 -O3，Ubuntu 24.04 LTS / Linux 6.8.0-44-generic（Table IV）。RRR 演示：AmpereOne 系统 48 copies（Figure 2）。基准参考机（用于计算 SPEC ratio 的归一化基准）：Lenovo ThinkSystem HR330A，3.0 GHz Ampere eMAG 8180（ARMv8 aarch64，32 核，2018 年产品）。评估工具链：PMC（Performance Monitoring Counters）+ Top-down 微架构分析、BBV（Basic Block Vector，Valgrind 提供）自相似性 recurrence 图、perf 时间序列图（IPC/frontend-bound/backend-bound）。
  - 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 Serving 框架（本文不涉及 LLM Serving）。最接近的匹配是 SPEC CPU 自带工具集 runcpu（随套件分发，C++/Perl 实现，社区有第三方构建工具 jiegec/spec2026）。修改/新增：runcpu 新增两个标志——`--rrrrate`（启用滚动轮询 rate 模式）与 `--rrrrate_inc[=N]`（队列步长，默认 1，0 表示只跑初始 benchmark 的快速验证模式）；`--copies`（进程队列数）、`--iterations`（每队列重复轮数）复用既有参数。实现要点：RRR 模式下 copies 由 runcpu 直接 fork（而非 specinvoke），`$SPECCOPYNUM`/`$BIND` 由 runcpu 展开使既有 submit 命令无需改动；结果校验推迟到独立并行阶段（每 copy 一个进程）；每个 benchmark×copy×iteration 各建一个运行目录（磁盘占用显著增加）；结果表无 SPECratio/geomean，每个 benchmark 报 per-copy 中位数再取均值的时间、CV（标准差/均值）、以及 min/max/mean/median/variance/sigma/cv/四分位/IQR/Tukey fences 等统计字段（RSF 中 spec.cpu2026.results.<benchmark>.<tune>.*）；RRR 模式不报能耗字段、`--minimize_rundirs` 不兼容。
  - 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：SPEC CPU 2026 不是自由开源软件——是商业授权套件（新客户 $3,000、2017 老客户优惠 $2,000、合格非营利 $750、认证学术机构免费，须经教授申请），分发源码、不供二进制，用户自行编译；ISO 内含 redistributable_sources/（第三方开源组件源码与许可）。官方文档：https://www.spec.org/cpu2026/docs/runcpu.html（RRR 见 section 1.8）、https://www.spec.org/cpu2026/。社区工具：https://github.com/jiegec/spec2026（需自备授权副本的 benchspec/ 与 bin/）。论文 arXiv: 2605.01575。
  - 使用例子（一次 RRR 多程序调度运行，框架输入到硬件执行全过程）：
    ```
    # 输入：runcpu 命令 + 选中的 benchmark 名 + copies/iterations/inc 参数
    runcpu --rrrrate -c oct29a --iterations=2 --copies=4 727.cppcheck_r 714.cpython_r 750.sealcrypto_r 734.vpr_r
    # → runcpu 为 4 个 copy 各建一条队列（含全部 4 个 benchmark），起点按进程号 mod 4 错开：
    #   copy0: 714→727→734→750 | copy1: 727→734→750→714 | copy2: 734→750→714→727 | copy3: 750→714→727→734
    #   每轮按 --rrrrate_inc=1 步进，队列重复 --iterations=2 轮
    # → runcpu 直接 fork 4 个编译好的 benchmark 进程（$SPECCOPYNUM/$BIND 由 runcpu 展开），
    #   4 个进程同时启动、独立运行不再同步，期间它们并发射出 CPU 上的共享资源
    #   （取指带宽、ITLB、分支预测器、L1/L2/L3、DRAM 带宽、预取器）→ 产生跨程序 cache/TLB/带宽争用
    # → 每个 benchmark 在每个 copy 上恰好完成 2 次（4 copies × 2 iterations = 8 次运行/benchmark），
    #   结果校验在独立并行阶段逐 copy 进行；每 benchmark 先取各 copy 内 2 次运行的中位数，
    #   再对 4 个 copy 的中位数取均值作为该 benchmark 的运行时间，并报 CV 等统计量
    # → 输出：每 copy 每 benchmark 的运行时间与 CV（如 220/225/228/231s → mean 226s, CV≈0.018）、
    #   无 SPECratio/geomean；配合 scripts.misc 中的 rate_timeplot.py 生成 Figure 2 式时间图
    #   （refrate 同质 vs rrrrate 异构对比），供研究者按自选指标（累计 IPC/吞吐/调和平均/公平性）分析
    ```
    作用：以标准化、确定性、可复现的轮转调度产生异构多程序负载，使多核争用/OS 调度/资源划分研究摆脱"自定义 benchmark 混合 + 临时调度策略"的碎片化现状，可直接对比不同研究的结果。
