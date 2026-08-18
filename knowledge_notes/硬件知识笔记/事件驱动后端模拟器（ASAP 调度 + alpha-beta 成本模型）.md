## 事件驱动后端模拟器（ASAP 调度 + alpha-beta 成本模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
事件驱动（event-driven）后端模拟器是 BusyBarn 自研的 cycle-level 性能评估工具（10K+ 行 Python）：以 as-soon-as-possible（ASAP）策略作为后端调度逻辑，同时评估计算事件、通信事件与 off-die DRAM 访问。核心机制：事件驱动后端在每个事件的目标设备（计算单元或链路）空闲且该事件的全部依赖满足时立即派发执行，因此计算与通信自动重叠（天然依赖检查机制）；计算单元按输入数据 shape 计时，链路用 alpha-beta 成本模型 [67] 建模（总时间 = 固定开销 α + 数据量/带宽项 β）。它把 BusyBarn 三个阶段（LR 记号 → 层次化映射 → BALD 通信调度）产出的完全调度事件集转为延迟/吞吐指标。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中的运转流程（Qwen2.5-7B prefill，bs=1、seq=512 验证例）：输入 = 硬件配置（TPUv5e 模型：4 个 1.5 GHz systolic array、16 GB HBM 819 GB/s、片间 800 GB/s；或 wafer 系统 Table I 参数）+ 完全调度事件集 → 模拟器维护每设备（tensor core/vector unit/链路/HBM）的可用时间与事件依赖 → 每当设备空闲且依赖满足即派发（ASAP）→ 计算事件按 shape 占用计算单元、通信事件按 alpha-beta 模型占用链路（片上 1 ns/256 GB/s、D2D 20 ns/256 GB/s、HBM 100 ns/256 GB/s per die）→ 输出端到端延迟、纯通信/纯计算/重叠时间占比。校准：对真实 2×2 TPUv5e 集群（vllm-tpu v0.13、TP=4）测 17.22 ms vs 模拟 16.6 ms（3.6% 偏差，剩余偏差来自未建模的软件栈开销与系统噪声）。相关技术：HD-MoE 的 discrete-event simulation（优先级队列 + 链路占用追踪，XY routing 路径缓存）同属事件驱动建模路线。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python 事件队列 + 每设备时间线 + 依赖图；事件类型分计算/通信/DRAM 访问。使用：`bash run_all.sh 16` 运行全部 12 图实验（16 并行任务），输出 output/ 下 12 张 PDF 图与 summary.txt；`bash run_quick_test.sh 16` 快速验证（<3 小时）。开源：https://github.com/redbird-arch/isca2026-busybarn-artifact.git（Zenodo: https://doi.org/10.5281/zenodo.19686855）；Python 3.9 + numpy/networkx/simanneal/matplotlib/tqdm/PyYAML。局限：假设理想执行环境、不建模软件栈与系统噪声（论文自述 3.6% 偏差来源）；ablation 实验内存密集（~32 GB/任务）。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
