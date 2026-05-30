## Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **SpecMoEOff**，在 SGLang 上构建的 MoE offloading serving 系统，通过 speculative decoding 增大每次前向 workload 来隐藏 offloading 延迟。核心 Serving 调度设计：(1) **Target Model Execution Engine**——基于 MoE-Lightning 的 CPU-GPU 流水线架构，batch 拆分为两个 micro-batch 交替执行 GPU Other1 → CPU Attention → GPU Other2 → GPU MoE，同时异步预取下一层 expert weights；(2) **Draft Model Execution Engine**——draft model KV cache 按 batch 维度切分为 GPU Part 和 CPU Part，两部分 attention 并行执行后统一在 GPU 做 FFN，动态调整 GPU/CPU 分离比例；(3) **Hyperparameter Optimizer**——自动搜索最优 batch size、micro-batch size、draft token 数量 k、内存管理策略和 execution strategy；(4) **Memory Manager**——管理 GPU HBM 和 CPU DRAM 的 KV cache 和 expert cache 分配。

  实验比较：
  - SpecMoEOff vs DeepSpeed-ZeRO-Inference vs MoE-Lightning 的端到端吞吐量和解码吞吐量
  - A30 vs 4090D 硬件环境下的性能 (不同 CPU memory 限制)
  - APPS vs CNN/DailyMail 数据集 (不同输入长度和 acceptance rate)
  - 不同输出长度 (128/256/512/1024) 下的 decode throughput
  - Micro-benchmark: varying draft length, input/output len, CPU/GPU memory

- 硬件平台是什么，配置是什么。
  A30: NVIDIA A30 GPU (165 TFLOPS) + Intel Xeon Gold 6426Y CPU, 250 GB CPU memory, CPU-GPU 25 GB/s。
  4090D: NVIDIA 4090D GPU (83 TFLOPS) + Intel Xeon Gold 5418Y CPU, 190 GB CPU memory, CPU-GPU 23 GB/s。

- 开源Serving框架是什么。修改了什么。
  基于 **SGLang** [33]，采纳 MoE-Lightning [6] 的 FFN/expert cache 设计，增加 20,000+ 行 Python/C++/CUDA。

  **修改内容**：
  1. **Speculative Decoding Pipeline 集成**：在 SGLang 的 decoding 循环中插入 draft model 生成 + target model 验证的双阶段执行流程
  2. **CPU-GPU Pipeline 编排**：target model 端将 batch 拆分为两个 micro-batch，交替执行 GPU Other1/CPU Attention/GPU Other2/GPU MoE，同时异步预取下一层 expert weights；使用分离的 CUDA Streams 管理 GPU 计算、expert 加载、activation 加载和 offloading
  3. **Draft Model Execution 分离**：draft model KV cache 按 batch 维度分片，GPU Part 全 GPU 执行，CPU Part 的 attention 在 CPU 计算后 hidden states 传回 GPU 做 FFN
  4. **Pin Memory + Dynamic Allocation**：使用 pin memory 减少 CPU→GPU 传输开销，动态内存分配避免内存碎片
  5. **Hyperparameter Optimizer**：自动搜索最优 (b, m, k, S_memory, S_execution)，凸优化预决定 b/m/S 参数，profiling estimator + DAG 模拟估计不同 k 的吞吐量，选择最优 k

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未公开独立开源仓库。基于 SGLang 构建。

  **SpecMoEOff Serving 框架输入到硬件执行全过程（以 Mixtral-8x7B + EAGLE draft, batch b 拆分为 2 micro-batches 为例）**：

  1. **请求到达与 Batch 组装**：b 个请求到达 → Scheduler 组装 batch → Hyperparameter Optimizer 决定 m (micro-batch size= b/2)、k (draft tokens)、expert cache 策略

  2. **Prefill 阶段**：chunk global batch 为 micro-batches → 每层加载 expert 参数到 GPU → 迭代 micro-batches → offload KV cache 和 hidden states 到 CPU DRAM。Draft model 所有参数放在 GPU HBM，执行 GPU-based prefill。

  3. **Decoding 阶段（Target Model Execution）**：
     - microbatch 1: GPU Other1 (LayerNorm, residual) → CPU Attention (Intel MKL chunked attention, KV cache from CPU DRAM) → GPU Other2 (router, etc.) → GPU MoE (expert weights 从 CPU DRAM 加载)
     - microbatch 2: 与 microbatch 1 交错执行（CPU Attention of microbatch 2 与 GPU MoE of microbatch 1 重叠）
     - 下一层 expert weights 在当前层计算期间通过独立 CUDA Stream 异步传输

  4. **Draft Model Execution**：
     - GPU Part requests: attention + FFN 全在 GPU（KV cache 在 GPU HBM）
     - CPU Part requests: attention 在 CPU 计算 → hidden states 传回 GPU → FFN 在 GPU 执行
     - 两部分并行执行 → 迭代 k 次生成 k 个 draft tokens
     - 动态调整：初始阶段更多在 GPU，序列变长后部分迁移至 CPU；请求完成后动态回迁

  5. **验证与同步**：draft tokens 与 original tokens 拼接 → target model 一次性前向验证 → 确定接受的 token 数量 a(k) → 更新 KV cache → 下一 iteration

  6. **Hyperparameter 动态调整**：随 sequence length 增长动态调整 k——生成初期 k 较大，后期减小；请求完成后增大 k

  7. **输出返回**：流式返回生成的 tokens
