## Cornserve Efficiently Serving Any-to-Any Multimodal Models (Cornfigurator)

- baseline方法是什么？
  **手动/专家驱动固定部署策略**：现有系统（vLLM, vLLM-Omni, SGLang-Omni, ModServe, EPD）使用预定义的固定部署策略：monolithic（全组件 colocation）、encoder-disaggregated（仅 encoder 解耦）、encoder-prefill-decode disaggregation（EPD）、或 fully disaggregated（全部组件解耦）。executor 级别配置（batch size, tensor parallelism degree, 实例数）需要人类专家手动调优。这些系统要么仅针对 A2A 的特例（如 ModServe 仅 MLLM, EPD 仅 encoder-prefill-decode），要么只提供解耦机制但不提供自动规划器。

  全栈执行例子（Qwen 3 Omni 30B on 16×A100-80GB，1/3 audio output，baseline=vLLM-Omni 专家调优方案）：
  - **模型推理算法层**：Qwen 3 Omni 的 DAG 组件图：E_img + E_vid + E_aud（多模态 encoder） → L_th（thinker LLM，自回归 text 生成） → L_ta（talker LLM，自回归 audio token 生成） → G_aud（vocoder，audio waveform 生成）。不同 request type 遍历不同子图（如有 audio output 则需 L_ta+G_aud，无则仅需 L_th）。
  - **系统框架层**：vLLM-Omni 采用固定解耦策略——预定义的组件分组方式，人工指定各 executor 的 GPU 分配、batch size、tensor parallel degree。无法根据 workload 中不同 request type 的比例自动调整分组或资源分配。例如 audio output 请求占 1/3 时，talker+voco 需要较多 GPU 资源，但专家调优的固定方案可能未能充分分配。
  - **编译框架层**：论文未明确说明编译框架层修改。使用标准 PyTorch/CUDA 编译路径。
  - **kernel调度层**：各 executor 独立运行，组件间通过 NCD（network collective communication）传输中间 tensor（~10ms 中位延迟）。vLLM-Omni 的 fixed plan 下，encoder executor 和 LLM executor 之间的数据传输 latency 固定，无自适应调度。
  - **硬件架构层**：16×NVIDIA A100-80GB GPU，NVSwitch 互联。GPU 分配固定，某些 GPU 可能因 colocation/batching 不当而利用率低。

  Baseline 核心缺陷：
  - (a) **固定策略无法适应模型和 workload 变化**：专家方案的 colocation/disaggregation 决策固定，无法根据 request type 分布（如 audio output 比例）、GPU budget 变化自动调整。图 3 显示即使对简单的单 encoder MLLM（InternVL 3 38B），不同 workload 下最优策略也显著不同——无 silver bullet。
  - (b) **全局延迟约束导致轻量 request type 不受保护**：使用单一全局延迟目标时，仅最重的 request type（如 audio output）约束生效，轻量 type（如 text output）可被无限制降级（图 5）。
  - (c) **缺乏 per-request-type 精细化推理**：将所有 request type 混在一起优化，无法为不同类型定制专用 subplan（如为 heavy video-input 请求准备 disaggregated video encoder 分支）。
  - (d) **搜索空间巨大（~500M candidate plans）但没有高效搜索机制**：手动探索不可行（Qwen 3 Omni + 16 GPU 产生 483M candidate plans），无粗到细评估剪枝 pipeline。
  - (e) **解耦非普遍有利**：将组件放到独立 GPU 上虽允许独立扩展，但消耗的 GPU 资源无法被其他组件使用（如 encoder 专用 GPU 无法存 LLM 的 KV cache），在某些场景下 monolithic 反而更优。固定策略无法根据具体情况权衡。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Cornfigurator：通用 Any-to-Any 多模态模型推理 Serving 的自动化部署规划器**。核心设计：(i) per-request-type 独立延迟约束和 goodput 优化——每种 request type 有独立延迟目标 L_t，最大化各 type goodput 之和；(ii) 系统化探索 colocation/disaggregation 组合——从 model DAG 枚举 simple→compound subplans→logical plans→physical plans，而非预设固定策略；(iii) 粗到细三阶段统计评估——Network flow（吞吐量上界）→ Monte Carlo（延迟估计）→ Request-level simulator（精确队列建模），每阶段剪枝劣化方案；(iv) 支持 subplan specialization 和 composition——允许不同 subplan 为不同 request type 子集优化，组合为完整 logical plan。

  全栈执行例子（同样 Qwen 3 Omni 30B on 16×A100-80GB，1/3 audio output，Cornfigurator 自动生成的 plan）：
  - **模型推理算法层**：同一 Qwen 3 Omni 模型定义不变。
  - **系统框架层**：Cornfigurator 接收 model definition（DAG）、configuration space（executor types）、workload（8 request types 及其比例）、GPU budget=16、per-type latency targets。Profiler 先对各 component 在目标硬件上 sweep batch size 和 parallelism degree，记录稳态吞吐+延迟。Planner 枚举后生成 physical plan: 1×(E_aud) + 4×(E_img+E_vid+L_th) + 11×(L_ta+G_aud)——将 audio encoder 解耦到独立 GPU，其余 encoder 与 thinker LLM colocation 为 4 个 executor，talker LLM 和 vocoder 共享 11 个 executor。Cornserve runtime 按此方案在 16 GPU 上部署 executor 实例。
  - **编译框架层**：论文未明确说明编译框架层修改。
  - **kernel调度层**：Plan 中 routing probabilities 决定各 request type 在各 parallel path 间的流量分配。Network flow phase 识别 bottleneck node，确保各 node 的 executor capacity 按 request type 比例加权后不超过 node 总容量。Monte Carlo 和 Simulator 进一步精细化 per-type goodput 估计，考虑 CPU-GPU overlap（encoder 的 CPU preprocessing 与 GPU 执行流水线化）、inter-type contention at shared nodes、occupancy-aware latency scaling（非 bottleneck node 按有效 batch size 缩放延迟）。
  - **硬件架构层**：同一 16×A100 GPU。Cornfigurator 的 plan 通过自动化的 colocation 决策（audio encoder 独立 GPU 以匹配其低吞吐特性，image+video encoder 与 thinker LLM colocation 以共享 GPU 资源），比 Fixed Disaggregation 减少 GPU 碎片。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 固定策略无法适应模型和 workload 变化** → 方案：Plan enumeration（Algorithm 1）从 model graph 系统化探索所有 colocation/disaggregation 组合（simple subplans → compound subplans → logical plans），不预设策略。图 18 显示当 InternVL 3 的 image input 概率从 25% 升至 75%，planner 自动从 monolithic 过渡到 encoder-disaggregated 再增大 batch size。
  - **defect: 全局延迟约束导致轻量 type 不受保护** → 方案：Per-type latency targets（§3.2）。Appendix A 证明当 L_t ∝ compute cost of type t 时，所有 type 的延迟约束 equally tight（scale factor ℓ_t 被约去，CDF 均在相同参数 L/ℓ_max 评估）。确保 Planner 不能以牺牲 text output 延迟为代价提升 audio output 吞吐。
  - **defect: 缺乏 per-request-type 精细化推理** → 方案：Subplan specialization（§4.2）。每个 simple/compound subplan 可仅覆盖部分 request type，在 logical plan 中组合（k_s=2）。图 10 的 Qwen 3 Omni on 16 GPU plan 展示了 compound subplan——一个分支用 disaggregated video encoder 服务 heavy video-input 请求，另一个分支用 monolithic 配置服务其余请求，共享 talker+vocoder executor。
  - **defect: 搜索空间巨大无高效探索** → 方案：Coarse-to-fine 三阶段评估+精确剪枝（§4.3, Algorithm 2）。Network flow（3.48s, 483M→1.95M candidates）剪枝冗余 GPU 配置，Monte Carlo（34.23s, 1.95M→25 candidates）剪枝 Pareto-suboptimal per-type goodput，Simulator（0.83s, 25→5 candidates）精确建模排队。总计 < 2 分钟完成，若全用 simulator 需 4400+ 小时。剪枝规则精确（仅丢弃保证冗余/劣化的 plan）。
  - **defect: 解耦非普遍有利** → 方案：Planner 在枚举阶段同时考虑 colocation（MERGE edge）和 disaggregation（KEEP edge），对每种组合评估 goodput。Qwen-Image（§6.4）上 planner 正确识别 monolithic 为最优（2-component model, LLM prefill 轻量且解耦会浪费 GPU），而 Full Disaggregation baseline 因强制解耦导致 GPU 碎片。
  - **额外设计: workload drift 自适应** → 当 request type 比例变化时，仅需 re-weight profiling 样本（无需重新 profiling），planner 重新规划耗时仅 single-digit seconds（§6.6）。GPU budget 变化也仅需重新运行规划（profiling 有效）。
  - **额外设计: 规划器 runtime-agnostic** → Cornfigurator 的 plan space 是 vLLM-Omni, ModServe, EPD, vLLM, Full Disaggregation 等所有 baseline 的严格超集，所有 baseline 方案可表达为 Cornfigurator plan。实验中将所有方案部署在同一 Cornserve runtime 上，消除框架实现差异的影响。
