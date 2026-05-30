## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- 属于Serving调度的实现是什么？实验比较什么？
  - LYNX 在 vLLM 上实现了一个批处理级别的动态专家选择系统，核心 Serving 层面的实现包括：
    1. **Phase-aware Optimizer（相位感知优化器）**：集成在 vLLM 的 batch scheduler 内，判断当前迭代是否处于 memory-bound 的 decode 阶段。对于 co-located 部署（prefill/decode 同机但不同时混合），识别 pure-decode batches 为 memory-bound；对于 disaggregated 部署（prefill/decode 分离节点），直接标记 decode 节点为 memory-bound；对于 chunked prefill，标记仅含 decode tokens 的 batch 为 memory-bound。只有 memory-bound 迭代才触发 LYNX 的专家重映射，compute-bound 迭代（如 prefill）直接绕过。
    2. **Batch-Aware Expert Remapping Pipeline**：在每层 MoE router 输出后插入三个 fused kernel——confidence analyzer（对每 token 的 router logits 做 AffinityBinning 离散化）、adaptive expert scorer（batch 级别加权打分，选出最小关键专家集）、expert remapper（将低置信度 token 重映射到保留的专家集上）。最终减少每个 batch 激活的专家总数，降低从 HBM 加载专家权重的内存带宽压力。
    3. **Continuous Batching 兼容**：LYNX 完全在每次 forward pass 的 runtime 内执行，不依赖 workload 的先验知识，自适应 continuous batching 导致的 batch 组合每次迭代变化。
  - 实验比较：
    - Baseline：vLLM v0.10.1 默认推理（v1 scheduler，所有默认优化开启）
    - LYNX vs Baseline 在 TPOT（time-per-output-token）上的对比
    - 两类服务场景：co-located prefill/decode 和 disaggregated prefill/decode
    - 下游准确率：GSM8K, HumanEval, MBPP, MATH, ChartQA, MMMU, AIME, GPQA
    - 真实 trace：ShareGPT（对话）和 Mooncake（工具代理）
    - SLO-aware throughput：20ms/25ms/30ms P99 TPOT 约束下的系统吞吐量
    - 与 offloading（Fiddler）和量化（INT4 GPTQ/AWQ）的互补性

- 硬件平台是什么，配置是什么。
  - **主评测平台**：NVIDIA H200 GPU (141 GB HBM)，SXM NVLink 互联
  - **CPU**：2x AMD EPYC 9554 64-Core (128 cores total)，1.5 TB DRAM
  - **OS**：Ubuntu 22.04.4 LTS，NVIDIA driver 560.35.05，CUDA 12.6
  - **Offloading 实验**：单卡 NVIDIA A100 GPU (94 GB)，19 GB offload 到 CPU
  - **并行策略**：Mixtral-8x7B/Qwen2-57B/Llama-4-Scout 用 TP=2，Qwen3-30B 单卡，DeepSeek-Coder/Llama-Maverick/Qwen3-235B 用 TP=4
  - EP 实验：TP=2,EP=2 和 TP=4,EP=4

- 开源Serving框架是什么。修改了什么。
  - **框架**：vLLM v0.10.1 (v1 scheduler)
  - **LYNX 自身开源**：论文未提供 GitHub 链接或开源仓库。通过 CLI flag 启用。
  - 核心修改：
    1. **Batch Scheduler 集成 Phase-aware Optimizer**：在 vLLM scheduler 中增加 memory-bound 判断逻辑，对三种服务策略分别识别 decode-only iterations。
    2. **Router 输出拦截**：在每层 MoE router top-k 后插入 4 个 fused kernel（confidence analyzer → adaptive expert scorer → expert remapper），在 expert computation kernel 启动前完成专家集缩减和 token 重映射。
    3. **Expert Kernel Launch 参数调整**：以缩减后的 active expert set 作为 dispatch 参数启动专家计算。
    4. **CUDA Graph 兼容**：4 个 fused kernel 保持静态控制流，支持 vLLM CUDAGraph capture。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文未提供开源代码链接。实现基于 vLLM v0.10.1，通过 CLI flag 启用。
  - **使用例子与全过程**（基于论文描述还原）：
    1. **[Batch Scheduler]** 接收请求 → continuous batching → Phase-aware Optimizer 判断 memory-bound → 若 decode-only batch 则设置 ENABLE_LYNX flag
    2. **[Model Forward - per layer]** Attention 计算 → MoE Router 产生 logits/top-k → LYNX Confidence Analyzer (Kernel 1)：拦截 router probability，对每 token 做 log-ratio AffinityBinning → Adaptive Expert Scorer (Kernel 2-3)：batch 级指数加权打分，确定最小关键专家集 → Expert Remapper (Kernel 4)：低置信度 token 重映射到 active expert set，保留高置信度 token 的 top-ranked expert，compaction + renormalize → Expert Computation Kernel：以缩减的 expert set 从 HBM 加载权重执行计算
    3. **[硬件执行 - H200]** CUDA Graph 已捕获静态执行图 → 4 次 fused kernel launch (替代 700+ PyTorch ops) → Expert weights 从 HBM 加载量减少 → Decode latency 降低
    **作用**：在不修改模型、不依赖校准数据的情况下，每次 decode iteration 动态缩减 batch 级活跃专家数，直接减少 HBM 数据搬运量，缓解 MoE decode 的 memory bandwidth 瓶颈。median TPOT 降低 1.09-1.30x，SLO 约束下系统吞吐量提升最多 2.1x。
