## Cornserve Efficiently Serving Any-to-Any Multimodal Models (Cornfigurator)

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Cornfigurator，一个面向通用 Any-to-Any（A2A）多模态模型推理 Serving 的自动化部署规划器。它根据模型定义（DAG 组件图）、配置空间（executor 类型及其配置）、workload（请求类型分布）和 GPU 预算，自动搜索最优的 colocation/disaggregation 组合、executor 配置（batch size、parallelism degree）和请求路由策略，以最大化每种请求类型的 goodput（满足 per-type 延迟目标的吞吐量）。规划器使用三层粗到细评估管道：network flow 估算吞吐量上限 → Monte Carlo 采样估算延迟 → request-level 模拟器精确建模排队动态，每层后剪枝淘汰劣化方案。
  - 实验比较：Cornfigurator 生成的部署方案 vs. vLLM-Omni（专家手动调优的固定策略）、Full Disaggregation（完全解耦的受限版本）、vLLM（monolithic）、ModServe（MLLM 专用解耦）、EPD（encoder-prefill-decode 解耦）在多种 A2A 模型上的 goodput。

- 硬件平台是什么，配置是什么。
  - 2× AWS p4de.24xlarge 实例，每实例 8× NVIDIA A100-80GB GPU（NVSwitch 互联），跨节点 400 Gbps 带宽。实验使用 8 GPU 和 16 GPU 配置。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：Cornserve（https://github.com/cornserve-ai/cornserve），一个通用 A2A 分布式 Serving 平台。Cornfigurator 是基于 Cornserve 之上的自动化规划器（https://github.com/cornserve-ai/cornfigurator）。规划器本身约 5K 行 Rust 实现。它不修改 Cornserve 的内部机制，而是作为规划层，生成 physical plan（节点拓扑、executor 数量、配置、路由概率）交给 Cornserve 执行。Planning 流程：Profiler 先对每个 model component 在各种配置下进行 benchmark → Planner 枚举 logical subplans（simple/compound）→ 组合成 logical plans → 注解 GPU 分配和 executor 配置生成 physical plans → 三阶段评估选最优。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：是。Cornfigurator 开源在 https://github.com/cornserve-ai/cornfigurator，Cornserve 开源在 https://github.com/cornserve-ai/cornserve。
  - 使用例子（Qwen 3 Omni 30B, 16 GPU, 2/3 audio output workload）：
    1. **输入**：Model definition（DAG: E_img, E_vid, E_aud, L_th(thinker LLM), L_ta(talker LLM), G_aud(vocoder)），Configuration space（executor 类型如 encoder/LLM/DiT executor，各支持的 batch size、parallelism degree），Workload（8 种 request type 的分布，如 T+I→T, T+I+V→T, T+I→A, T+I+V→A 等），Latency targets（per-type），GPU budget N=16。
    2. **Profiler**：对每个 executor type 在 A100-80GB 上 sweep batch size 和 parallelism degree，记录稳态吞吐和延迟（去除排队延迟），输出 per-executor-config 的 profile。
    3. **Planner 枚举**：从 model graph 生成 simple subplans（通过枚举每条 colocation edge 的 Keep/Merge 决策），合并共享节点的 subplans 为 compound subplans（k_c=2），组合为 logical plans（k_s=2），注解 executor 分配和路由概率生成 physical plans（约 483M 候选）。
    4. **粗到细评估**：
       - Phase 1 (Network flow, 3.48s)：计算每个 plan 在各 node 汇聚的瓶颈吞吐量 R_d，剪枝冗余配置（483M→1.95M）。
       - Phase 2 (Monte Carlo, 34.23s)：采样请求、按路由概率流经 plan 各 executor，累积 per-executor 处理延迟得出 per-type 延迟 CDF，计算 goodput，剪枝 Pareto-suboptimal plans（1.95M→25）。
       - Phase 3 (Simulator, 0.83s)：request-level 模拟器以 α·R_d 速率运行 workload，建模排队动态和 inter-type 竞争，计算最终 goodput（25→5）。
    5. **输出 Physical Plan**：一个 compound subplan 结构——一个分支用 disaggregated video encoder 处理 heavy video-input 请求，另一个分支用 monolithic 配置处理其余请求，共享 13×(L_ta+G_aud) talker+vocoder executor。
    6. **部署执行**：Cornserve runtime 接收 physical plan，在 16×A100 GPU 上按配置启动 executor 实例，根据路由概率将各 request type 的请求分发到对应 subplan 路径，各 executor 按配置的 batch size 和 parallelism 执行推理计算。
    - **作用**：自动为通用 A2A 模型找到最大化 per-type goodput 的部署方案，避免人工专家调优，1.12×–6.32× 优于 baseline。
