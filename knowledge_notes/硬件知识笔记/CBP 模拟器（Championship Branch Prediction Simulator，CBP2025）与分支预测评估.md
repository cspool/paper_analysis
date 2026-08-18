## CBP 模拟器（Championship Branch Prediction Simulator，CBP2025）与分支预测评估

术语解释
CBP（Championship Branch Prediction）是由工业界（ARM 等）赞助、与 ISCA 同期举办的国际分支预测竞赛及其官方 trace-driven 模拟器。CBP2025（第 6 届，ISCA 2025 东京）的官方模拟器仓库为 ramisheikh/cbp2025，评估用 673 条公开 trace（Zenodo），存储预算 192 KiB，指标为 BrMisPKI/CycWpPKI。RUNLTS 在 CBP2025 获得第一名（RUNLTS-Log）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：(1) 背景：CBP 是分支预测领域的事实标准评测平台（CBP-1~CBP-6），组织者提供统一模拟器框架与固定存储预算、公开 trace，参赛者提交预测器实现；(2) CBP2025 模拟器（https://github.com/ramisheikh/cbp2025）：trace-driven，读取 .gz 分支 trace（`./cbp trace.gz`，`-E N` 每 N 条输出一次误预测统计），`cond_branch_predictor_interface.cc`/`cbp.h` 定义参赛接口，内置 CBP2016 冠军 64KB TAGE-SC-L 作基础条件预测器，参赛者可加最多 128KB 自己的组件（或用满 192KB 预算）；(3) CBP2025 评估 trace：673 条（含 training 105 条 + secret test），由 CBP2025 组织者发布于 Zenodo（https://doi.org/10.5281/zenodo.15883615，CC BY 4.0，约 78 GB，原可执行文件不公开）；(4) 指标：BrMisPKI（每千条指令条件分支误预测数，即 MPKI）与 CycWpPKI（每千条指令错误路径周期数）双赛道；(5) 局限：trace-driven、不模拟错误路径执行（CBP simulator 虽建模分支预测与乱序后端流水线模型，但不仿真错误路径）——这正是 RUNLTS 同时用 gem5 评估的原因（验证 wrong-path 对 RBias digest 的影响）。
- RUNLTS 的用法：673 条评估 trace 上比较 RUNLTS-Log（CBP2025 第 1 名）、RUNLTS-Seq、RUNLTS 无 RBias 与 baseline TAGE-SC-L（192 KiB、Sheikh 等适配版）及五个对比预测器（MPP-2025、TAGE-SC-2025、LVCP、TASQ-SC-L、Bullseye）；RUNLTS-Log 在 597/673 条 trace 上优于 TAGE-SC-L、平均 MPKI 降低 0.137、整体约 5.0%；机制分解（RBias -2.46%、IMLI tweaks -0.74%、call-stack GEHL -0.43% 等）与 S-curve（Figure 10-12）均由 CBP simulator 产出。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CBP 模拟器位于"分支预测器设计 → 精度评估"的评测环节：硬件预测器以 C++ 类实现、挂接到模拟器接口，模拟器按处理器模型（RUNLTS 的 Table I：16-wide 前端 10 cycles、24 issue、1024 ROB、128 KiB L1I/L1D 3 cycles、4 MiB L2 12 cycles、32 MiB L3 50 cycles、150 cycle 主存）对每条 trace 分支做预测与更新时序 → 统计误预测。运转流程例子（复现 Figure 10）：`cd ISCA_2026_Artifact_RUNLTS && ./docker/launch.sh && ./run.sh` → 下载 673 条 trace 与 5 个对比预测器、克隆固定 commit 的 cbp2025 → 对 8 个配置 × 673 条 trace 逐条跑 `./cbp trace.gz` → 日志聚合为 result CSV → `python3 plot_s_curve_mpki.py` 生成 s-curve-mpki.pdf（对应论文 Figure 10，baseline 只用于算 MPKI reduction 不画图）。磁盘峰值 163 GiB（删 .tar.xz 后 91 GiB）、trace 下载约 10 小时、64 核机器跑完全部约 4 小时。
- 例子（预测器接口时序）：每条分支事件进入模拟器 → 接口按流水线模型依次调用预测器的方法（TAGE 各表 tag 匹配、SC 各组件加权求和、RBias digest/WT/UT 访问）→ 返回预测方向 → 与实际方向比对统计 → 按规则触发训练更新。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C++17 模拟器（g++ 13.3.0、GNU make 构建），Python 3 驱动工作流（multiprocessing 并行跑 predictor-trace 对、聚合 CSV、matplotlib 画图），Docker 可选（Ubuntu 24.04 环境）。开源情况：官方模拟器 https://github.com/ramisheikh/cbp2025 公开；RUNLTS 实现随 artifact 开源（Zenodo 10.5281/zenodo.19453058，BSD 3-Clause Clear License，含 CBP simulator 与 gem5 两套代码与完整工作流脚本）；评估 trace 公开（Zenodo，CC BY 4.0）。使用场景：分支预测研究的事实标准评测平台——新预测器与 SOTA（TAGE-SC-L、Multiperspective Perceptron 等）在统一预算与统一 trace 下比较 MPKI。
- Web 证据：CBP2025 官网 https://ericrotenberg.wordpress.ncsu.edu/cbp2025/ 与模拟器框架页 https://ericrotenberg.wordpress.ncsu.edu/cbp2025-simulator-framework/；官方模拟器 GitHub https://github.com/ramisheikh/cbp2025；CBP2025 Full Traces Zenodo https://zenodo.org/records/15883615。

涉及论文标题：
- RUNLTS Branch Prediction with Register-Value Correlations and Hierarchical Table Orchestration
