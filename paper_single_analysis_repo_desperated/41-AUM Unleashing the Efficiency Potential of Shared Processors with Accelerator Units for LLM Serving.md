论文标题：AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

CPU cluster的调度优化：LLM负载使用AU，通用负载使用其他模块，资源隔离的调度优化。

本地条目说明：
    - 本地编号：paper_2026 第 41 篇
    - 本地 PDF：paper_2026/41-AUM Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving.pdf
    - 本地文本抽取：paper_2026/41-AUM Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving.txt
    - 发表信息：HPCA 2026，DOI：10.1109/HPCA68181.2026.11408539
    - 原文入口：https://www.cs.sjtu.edu.cn/~lichao/publications/AUM_Unleashing_HPCA-2026-Wang.pdf

开源仓库确认：
    - 状态：未找到明确官方开源仓库
    - 链接：N/A
    - 说明：论文 PDF、HPCA 2026 会议页面、第一作者主页和 Chao Li 教授 publications 页面均能确认论文与 PDF，但未看到 AUM 的官方 GitHub / artifact 链接。论文只说明原型基于 xFasterTransformer，以 Python 实现 background profiler 和 runtime controller；引用中出现的是 xFasterTransformer、oneDNN、pmu-tools、sysbench、Ktransformers 等依赖或相关系统仓库，不是 AUM 自身仓库。因此本文只能确认论文公开可读，不能确认官方实现已开源。

1、论文工作：
    - 论文要解决的核心问题：
      AUM 解决的是带片上 Accelerator Unit（AU）的通用 CPU 在 LLM serving 与通用负载共用时的资源管理问题。现代 CPU 已把 Intel AMX、AVX、ARM SME、RISC-V M-extension 这类加速单元放进核心执行流水线，用于矩阵乘等 AI 算子。工业实践为了保证 LLM serving 的性能，常把整颗 AU-enabled CPU 专门分配给 AU 应用，不与普通 workload 共用；论文指出这种 exclusive AU usage 会浪费大量非瓶颈资源，并且在 CPU-only LLM serving 相对 GPU 仍有能效差距时，单独占满 CPU 并不是高效方案。真正需要的是在保证 LLM serving TTFT / TPOT SLO 的前提下，把 AU 应用没有用满的 CPU 资源收割给 Compute、OLAP、SPECjbb 等 best-effort 负载。
    - 瓶颈来源：
      瓶颈不是单一的算力或内存，而是 AU 引入后出现的三维 Accelerator Unit Variations（AUV）：第一，usage pattern 可变，prefill 的 AMX/GEMM 使用率高，而 decode 常退化为更适合 AVX/GEMV 的低 AU 使用；第二，frequency interference 是强制性的，更多 AMX/AVX 使用会因 TDP 约束触发核心频率下降，且与 co-running workload 的功耗压力叠加后更难预测；第三，resource bound 与传统 CPU workload 不同，AU 前端资源相对富余，但后端资源，尤其 ROB / serializing operation、cache hierarchy、DRAM bandwidth，会随 prefill/decode、模型尺寸、平台内存带宽而变化。
    - 论文的主要贡献：
      论文首先系统刻画 AU-exclusive 与 AUV-oblivious sharing 的不足，展示现有 SMT sharing 和 resource partitioning 在共享环境中可造成 10%-50% 的 AU 应用性能或 CPU 效率下降。随后提出 AUM，一个 AU-aware resource manager，用两类组件处理三维 AUV：离线 Background AU Profiler 把 usage、frequency、resource profile 离散化成 AUV Model；在线 Runtime AU Controller 根据运行时 SLO slack、频率区域和资源碰撞情况选择 core division 与 cache / bandwidth allocation。实验表明 AUM 相比 AU-exclusive 提升最高 8.8% CPU performance-per-watt，相比 AUV-oblivious sharing 平均提升 4.7%，并将 SLO violation 降低 7%-11%。
    - 论文所处背景：
      目标场景是云数据中心中的 CPU-based LLM serving，尤其是在 GPU 供给紧张、CPU 中已有 AMX/AVX 等 AI 加速能力、并且数据中心仍存在大量空闲 CPU core 的情况下，将 AU-enabled CPU 用作 LLM prompt / decode serving 或 GPU 辅助资源。论文实验围绕 Intel Sapphire Rapids / Granite Rapids 系列 production CPU，使用 xFasterTransformer serving Llama 模型，并把 LLM serving 视作 latency-critical workload，把普通应用视作 best-effort co-runner。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：
      第一个 baseline 是 ALL-AU，即整颗 AU-enabled CPU 只跑 LLM serving，不共享。它可以获得最好的 AU 应用性能，但 shared application 性能为零，且 CPU 的 frontend、cache、memory bandwidth 或非 AU core 周期会被浪费。论文还指出，在 GenA 平台上 Llama2-7B BF16 serving 的绝对性能约 188 tokens/s、功耗约 270W、CPU 成本约 7200 美元；相比 A100 GPU，CPU 的 perf-per-watt 仍较差，因此 exclusive AU 并不是能效最优的数据中心部署方式。
    - AUV-oblivious SMT sharing 的问题：
      SMT-AU 试图把 AU 应用和普通线程放到同一物理核心上，收割冗余周期。但它不知道 prefill/decode 的 AU 使用率、强制降频和后端瓶颈差异。论文在 OLAP co-runner 下观察到，SMT sharing 可让 AU performance degradation 超过 200%，OLAP 也受 40% 以上影响，主要来自严重 memory contention；而 compute-intensive co-runner 对 AU 干扰小于 10%，但自身会因 AU 引发的 compulsory frequency reduction 受到接近 40% 的性能损失。也就是说，SMT 的成败强依赖负载类型和 AU phase，静态共享策略无法稳定保证 SLO。
    - AUV-oblivious resource partitioning 的问题：
      RP-AU 使用 Intel RDT / CAT / MBA 等资源隔离机制切分 L2、LLC、memory bandwidth。问题在于它通常只看传统 workload interference，不能判断不同 AU usage 下哪个资源是真正紧张的。论文发现单独隔离某个 backend resource 只能略微缓解 LLM slowdown，不同资源的 exclusive / inclusive 效果不同，静态或普通 workload-aware partitioning 无法找到 AU performance 与 overall efficiency 的最优组合。
    - 论文的设计方法：
      AUM 的设计核心是“离线建模三维 AUV + 在线按 SLO 与效率调节资源”。Background AU Profiler 做三件事：用 arithmetic intensity（ARI）推断算子 AU usage，把 prefill 这类 high-AU、decode 这类 low-AU、普通 workload 的 none-AU 分成不同 processor regions，并在不同 core division、frequency、L2/LLC/bandwidth 配置下记录 AU 与 shared workload 的平均性能、尾部性能和 CPU power。Runtime AU Controller 也做三件事：Slack-aware SLO Analyzer 计算 prefill 的 TTFT slack 和 decode 的 LAG / TPOT slack；Efficiency-aware Core Switcher 在满足 AU SLO 的约束下最大化 weighted performance-per-watt；Collision-aware Allocation Tuner 根据当前 token latency 等轻量指标调节 LLC / bandwidth 等资源，必要时切换 core/frequency division。
    - 方法如何对冲 Baseline 缺陷：
      对 ALL-AU，AUM 通过 none-AU region 和低敏感资源收割，把 AU 未使用的 CPU 资源分给 shared workload，从而提高整体 perf-per-watt。对 SMT-AU，AUM 不把共享看成固定 hyperthread policy，而是显式分出 high-AU / low-AU / none-AU 区域，减少 AU 降频和 memory contention 的级联影响。对 RP-AU，AUM 不静态切分资源，而是根据 AUV Model 中的 resource affinity 和运行时 SLO slack 动态决定是否收回或释放 LLC / memory bandwidth。对 prefill 和 decode 的差异，AUM 以 TTFT 和 TPOT/LAG 分别建模，避免把两个阶段当成同一种 AU workload。
    - 关键 trade-off：
      AUM 接受了 profiling、模型存储和 runtime daemon 的复杂性，以换取共享环境下更好的效率与 SLO。离线 profiling 需要重复实验构建 AUV Model，论文报告一次模型构建约 450 次 AU-enabled executions；不过它可摊销到相同模型和平台的许多 CPU core。在线控制采用 rule-based lookup，而不是更强的 online learning，因此控制开销小于 1ms、内存约 15MB，但适应未见过的模型、算子实现、硬件拓扑或突发 workload 时，需要重新 profiling 或扩展模型维度。另一个 trade-off 是 AUM 优先保证 AU serving SLO，对 shared workload 的 SLO 支持仍较弱，论文也把 best-effort co-runner 的 SLO-aware tuning 留作未来工作。

3、论文实现：
    - Baseline 如何实现：
      ALL-AU 使用整颗 AU-enabled CPU 运行 LLM serving，不部署 co-runner。SMT-AU 采用 SMT sharing，把 LLM serving 与 co-running workload 共享 CPU core，代表 SOTA SMT / co-location 管理方式。RP-AU 采用 workload-aware resource partitioning，使用类似 Intel RDT 的资源控制机制，对 L2 cache、LLC、memory bandwidth 等关键 backend resources 做隔离或分配。论文还构造 AU-UP、AU-FI、AU-RB 三个 AUM 变体，分别只考虑 usage pattern、frequency interference、resource bound，用于消融三维 AUV 的贡献。
    - 新设计如何实现：
      AUM 原型基于 Intel xFasterTransformer，两个核心组件均以 Python 实现，作为系统级管理组件运行。Background AU Profiler 在 dedicated nodes 上重复运行模型与配置，记录 AU usage、core division、frequency lower bound、L2/LLC/memory bandwidth resource affinity、AU/shared performance 和 CPU power，并离散化为 AUV Model / AU Bucket。Runtime AU Controller 作为 system daemon 监控 SLO 与 token latency，通过查表选择资源配置。控制逻辑包含 prefill FCFS + TTFT slack、decode LAG analysis、weighted efficiency objective，以及在性能偏差超过阈值时的 processor division switch。论文使用 Linux perf、pmu-tools、pqos、turbostat 等工具采集 AMX busy、AMX uop ratio、AVX instructions、top-down cycle breakdown、频率、RDT 资源分配等信息。
    - 实验 / 实现平台：
      论文在三类 production Intel AU-enabled CPU 上评估：GenA 为 Sapphire Rapids Xeon 8475B，双 socket，每 socket 48 cores，DDR5 1TB；GenB 为 Sapphire Rapids Xeon Max 9468，双 socket，每 socket 48 cores，HBM 128GB；GenC 为 Granite Rapids Xeon 6982P-C，单 socket 120 cores，MCR 768GB。三者都有 AVX-512 / AMX，论文计算的 AU TFLOPS 随平台和 base frequency 不同而变化。LLM serving 使用 xFasterTransformer 的 AMX 支持，服务 Llama2-7B 和 Llama2-13B，BF16 precision，batch size 16，并通过不同 output length 模拟 prefill/decode phase。额外 AU workload 分析包括 Faiss、Vocoder、DeepFM；共享 workload 包括 sysbench Compute、TPC-H OLAP 和 SPECjbb 2015。
    - 关键实验设置与指标：
      LLM serving 的 SLO 指标是 prefill 的 TTFT 和 decode 的 TPOT，吞吐用 tokens/s 衡量。三类用户场景包括 ChatGPT-like chatbot（ShareGPT，TTFT 250ms，TPOT 100ms，平均输入 755、输出 200）、Cursor-like code completion（HumanEval，TTFT 75ms，TPOT 150ms，平均输入 171、输出 98）、Summarization（LongBench，TTFT 1.5s，TPOT 100ms，平均输入 1738、输出 91）。效率指标是 weighted application performance / CPU power，AUM 默认把 high-AU prefill token 权重设为 1.8、low-AU decode token 权重设为 0.2，shared workload 权重按每个 query 的 CPU time 经验设置。
    - 主要实验结果：
      在 CPU performance-per-watt 上，AUM 平均相比 ALL-AU、SMT-AU、RP-AU 分别提升 8.8%、6.7%、4.7%；最大场景提升达到 15.3%，最小也有 1.6%。在跨平台实验中，随着 GenA 到 GenC 的 AU 与 memory device 增强，AUM 相对 AU-exclusive 的效率提升在 chatbot、code completion、summarization 场景分别达到 19%、11%、17%。在 SLO guarantee 上，对于 loose TTFT 的 summarization，AUM 达到 93.6% SLO guarantee ratio，比 AUV-oblivious schemes 高 11%；对低 AU decode 阶段，AUM 的 TPOT SLO 表现接近 ALL-AU，并比 AUV-oblivious sharing 高约 7%。开销方面，runtime controller 单次查表决策小于 1ms，相比 100ms 量级 token SLO 可忽略。
    - 局限性与适用边界：
      论文明确指出 AUM 当前主要是 single-machine management，还未直接做 cluster-level load balancing；当前 profiling 聚焦 xFasterTransformer + oneDNN 的 AU operators，不同稀疏、tiling 或手写 AMX kernel 会增加 AUV，需要重新适配；当前硬件假设主要是 AMX-enabled CPU core，若未来 ARM SME 在物理 core 之间共享，或 CPU AU 与 GPU/NPU hybrid topology 深度耦合，profiler 需要增加新的 contention 维度；其他 ISA 如 ARM SVE/SME、RISC-V V/M-extension 理论上可套用三维设计，但需要实际硬件事件和控制接口支持。

4、pipeline/kernel 解析：
    - 新 pipeline/kernel 是什么：
      AUM 不是提出一个新的矩阵乘 kernel，而是提出一条 AU-aware processor sharing control pipeline。最接近“kernel/执行流”的新机制是 Background AU Profiler + Runtime AU Controller 的闭环资源管理路径：离线把 AU operator / phase 的 usage、frequency、resource bounds profile 成 AUV Model；在线把 LLM request 的 TTFT/TPOT slack、当前 token latency、core/frequency regions、LLC / memory bandwidth allocations 结合起来，生成 AU-aware resource sharing decision。它管理的是 CPU core、frequency region、cache ways、memory bandwidth 和 co-runner placement，而不是替换 xFasterTransformer 内部 AMX kernel。
    - 离线 Background AU Profiler 流程：
      第一步，Usage-aware AU Selecting 根据算子的 arithmetic intensity 和模型/阶段参数判断 AU usage。prefill 中 QKV / FFN GEMM 的 batch × sequence 维度大，AMX busy 和 GEMM TFLOPS 高，归为 high-AU；decode 以单 token 或小矩阵 GEMV 为主，AMX 使用低且 AVX 可能更适合，归为 low-AU；普通 best-effort workload 归为 none-AU。第二步，Frequency-aware Processor Dividing 对 high-AU、low-AU、none-AU 三类区域分别记录 core set 和 frequency lower bound，例如 prefill 高 AMX 使用会把频率压到更低，decode 降频较轻但与 co-runner 功耗压力叠加会出现不稳定频率干扰。第三步，Bound-aware Resource Profiling 通过 CAT / MBA 等接口改变 L2、LLC、memory bandwidth allocation，记录不同资源配置下 AU 与 shared workload 的平均性能、尾部性能和功耗，形成离散 AU Bucket。
    - 在线 Runtime AU Controller 流程：
      第一步，Slack-aware SLO Analyzer 读取 LLM queue 和已执行 token 信息。对 prefill，它用 SLOH = dTTFT - twait 表示 prompt 已等待后剩余的 TTFT slack；对 decode，它用 LAG 衡量某个 request 的生成进度相对理想 TPOT schedule 是提前还是落后，并设 SLOL = dTPOT + LAG。第二步，Efficiency-aware Core Switcher 在 AUV Model 中查找满足 PH < SLOH 且 PL < dTPOT 的候选配置，最大化 ECPU = (alpha * PH + beta * PL + gamma * PN) / WCPU，并选择对应的 usage/core/frequency 配置。第三步，Collision-aware Allocation Tuner 持续监控 AU performance，如果 AU 仍满足 SLO，就优先把对 AU 性能影响最小的资源让给 shared workload；如果 AU 落后 SLO，就用 tail performance profile 保守地把资源还给 AU，偏差超过阈值时切换 core division。
    - 一个请求如何流过 AUM：
      假设一批 Llama serving 请求到达，部分是长 prompt summarization，部分是短 prompt code completion。请求先进入 xFasterTransformer 的 serving queue。prefill prompt 作为 high-AU work item，AUM 用等待时间和 TTFT deadline 算 slack，若 slack 紧张，就在 AUV Model 中选择更多 high-AU cores、更保守的 LLC / bandwidth allocation 和较少 shared co-runner 压力。随后 decode token 进入 low-AU 区域，AUM 按每个 request 的 LAG 判断是否落后理想 TPOT；如果 decode 仍有余量，AUM 可能减少 decode 的 LLC ways 或 memory bandwidth，把 none-AU region 分给 SPECjbb / Compute；如果 token latency 上升导致 LAG 变负，controller 会收回 bandwidth 或切换到对 AU 更友好的 region division。整个过程中 AMX/AVX kernel 仍由 xFasterTransformer / oneDNN 执行，AUM 改变的是它们运行时所在的 CPU 区域和资源边界。
    - 与传统 sharing pipeline 的区别：
      传统 exclusive pipeline 是“LLM serving 独占整颗 AU CPU”，路径简单但浪费资源。传统 SMT sharing pipeline 是“把 co-runner 塞进同一核心或硬件线程”，不理解 AU phase，也不理解强制降频和后端资源差异。传统 RP pipeline 是“按普通 workload interference 切 cache / bandwidth”，无法判断 high-AU prefill、low-AU decode 与 none-AU workload 的不同 resource affinity。AUM 的执行路径则把 AU 使用率、频率区间和资源需求都变成显式 profile，再用 SLO slack 决定 runtime harvesting 强度，因此它能在不明显伤害 LLM SLO 的情况下给普通 workload 留出执行空间。
    - 关键机制的直观例子：
      如果当前是 chatbot 场景，prefill 和 decode 的 SLO 都比较紧，且 co-runner 是 SPECjbb，AUM 会更保守地配置 memory bandwidth，避免 SPECjbb 的复杂 backend demand 与 decode 争抢 DRAM。若当前是 code completion 场景，decode TPOT SLO 较松，AUM 可以把更多 none-AU cores 和部分 LLC / bandwidth 交给 shared workload，从而提升整体 weighted efficiency。若 co-runner 是 OLAP，论文观察到 memory-intensive sharing 的收益有限，AUM 即使能更精细分配，也不会像 Compute 场景那样获得很高效率提升；这体现出 AUM 不是盲目共享，而是按 workload 类型、AU phase 和 SLO slack 选择共享程度。
