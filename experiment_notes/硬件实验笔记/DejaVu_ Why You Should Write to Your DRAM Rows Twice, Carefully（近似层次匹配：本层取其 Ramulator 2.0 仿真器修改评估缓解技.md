## DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully（近似层次匹配：本层取其 Ramulator 2.0 仿真器修改评估缓解技术性能开销部分；主体为真实 DRAM 芯片表征，见芯片设计层条目）

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现：在 Ramulator 2.0 DRAM 模拟器中实现并评估两种读干扰缓解技术 PARA 与 PRAC，量化"因 DejaVu 而需把读干扰阈值 NRH 调低作 guardband"带来的系统性能开销。PARA：按配置的 NRH 值确定概率，每次 ACT 时以该概率判定目标行为 aggressor 行并预防性刷新其邻居行；PRAC：in-DRAM per-row counter 跟踪 aggressor 行激活计数，达到配置阈值前预防性刷新其邻居行。实验比较：NRH 五档扫描（−20%…+20%，{Mitigation}{Difference} 标注），以不实现任何缓解的基线系统为 1.0 归一化系统性能，覆盖 60 个随机四核 workload mix。
- 硬件平台是什么，配置是什么。
  - 主流规格 x86 PC/server（无特殊硬件要求），Docker 容器（镜像 richardluo831/ramulator2，内含 python、c++-20 工具链、pandas、matplotlib 等依赖）；仿真对象为 DDR4 内存子系统，具体 DDR4 时序/容量/控制器配置论文未明确说明。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - Ramulator 2.0（CMU-SAFARI 现代模块化可扩展 DRAM 模拟器，开源 https://github.com/CMU-SAFARI/ramulator2，论文 ref [205–208]）；DejaVu 修改版经 Dockerhub 镜像 richardluo831/ramulator2 发布。
- 模拟器模拟什么的性能，修改了什么。
  - 模拟带 PARA/PRAC 缓解逻辑的 DDR4 内存子系统对多核 workload 的性能影响。修改：在 Ramulator 2.0 控制器前端实现 PARA 概率性邻居刷新与 PRAC in-DRAM per-row counter 模型，并把 NRH 阈值做成可扫描参数（−20%…+20%）。输入：57 个单核 workload（SPEC CPU2006、SPEC CPU2017、TPC、MediaBench、YCSB）组成的 60 个随机四核 mix 访存 trace；输出：归一化系统性能（IPC）随 NRH 的变化曲线（Fig.24/25）。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源：artifact 公开（Zenodo DOI 10.5281/zenodo.19444878，MIT）；基座 Ramulator 2.0 开源（链接如上）。运行：docker run 镜像 richardluo831/ramulator2 → 拷贝 artifact 的 perf_eval 目录入容器 → run_artifact.sh 跑全部仿真 → parse_results.sh 收集解析结果 → plot_all_figures.sh 生成 Fig.23/24（ae_results/dejavu/_plots/）。
  - 模拟原理与输入→输出全过程：输入 = 60 个四核 workload mix 的访存 trace + NRH 配置 → Ramulator 2.0 前端将请求翻译为 DDR4 命令流 → PARA/PRAC 模型介入 ACT 路径：PARA 按与 NRH 相关的概率标记目标行为 aggressor 并插入其邻居行预防性 REF；PRAC 用 per-row counter 累积激活计数、达 NRH 前触发邻居 REF → 命令流经内存控制器排队与 DDR4 时序模型执行 → 统计各 mix 完成时间/IPC，对无缓解基线归一化。作用：量化 DejaVu 引发的阈值 guardband 代价——NRH=64 时 −20% guardband 使 PARA 平均性能开销 6.3%（PRAC 因开销主要来自时序参数放宽而较小），+20% 时 PARA 性能提升 7.8%、PRAC 提升 2.1%；表明 DejaVu 使"读干扰阈值配置"成为性能与安全性的直接权衡点。
