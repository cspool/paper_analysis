## MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-GEN 提出 **module-based batching** 策略，核心修改在离线推理 serving 框架的批处理调度层：
    1. **Module-based batching**：将 MoE 模型分解为 attention 和 expert 两类计算密集型模块，分别为其设置不同的微批次大小（$b_a$ 和 $b_e$）。在 attention 模块以小批次运行，累计多个 attention 批次的 token 后在 expert 模块合并为大批次运行，从而最大化 expert 模块的 GPU 利用率。
    2. **Full KV-cache offloading**：将 KV-cache 完全卸载到 host memory，节省 GPU memory 给更大批次，相比部分卸载减少最高 20× 的 expert weight fetching 流量。
    3. **CPU attention offloading**：将 self-attention 机制（$QK^T$）的计算卸载到 CPU（自定义 AVX 内核），节省 HtoD 带宽给 expert 预取。
    4. **DAG-based batching strategy search**：将 MoE 卸载推理建模为 DAG，通过动态规划求解 critical path 最小化执行时间，自动搜索最优的 $B$、$b_a$、$b_e$、$\omega$（CPU split ratio）、$S_{Expert}$、$S_{Params}$ 配置。
    5. **Single GPU buffer for dense modules**：dense 模块（attention、shared expert）使用单 GPU 预取缓冲区，大小为单层 dense 模块大小即可充分 overlap。
    6. **Engine 实现**：约 3000 行 C++ 和 2000 行 Python 代码，集成 HuggingFace generation pipeline。
  - 实验比较：(1) MoE-GEN(G)（纯 GPU 计算）和 MoE-GEN(H)（CPU+GPU 混合 attention）vs baselines：Llama.cpp、vLLM（continuous batching）、DeepSpeed-Inference、FlexGen*、MoE-Lightning*（model-based batching）；(2) 不同模型（Mixtral-8x7B、Mixtral-8x22B、DeepSeek-V2 236B、DeepSeek-R1 671B）下的 prefill throughput 和 decoding throughput；(3) 不同 context length（512-24K tokens）的 long context performance；(4) 不同 batch size（1, 32）下的小批次性能；(5) 不同 CPU attention ratio $\omega$ 的影响（0-100%）；(6) 完整 dataset 完成时间（MMLU 116K、GSM8K 8.5K、ChatbotArena 36K sequences）。

- 硬件平台是什么，配置是什么。
  - C1: NVIDIA A5000 24GB + AMD EPYC 7453 28-Core + 256GB Host Memory
  - C2: NVIDIA A5000 24GB + AMD EPYC 7453 28-Core + 512GB Host Memory
  - C3: NVIDIA A6000 48GB + AMD EPYC 7313P 16-Core + 480GB Host Memory
  - PCIe 4.0 互连（32 GB/s HtoD 带宽）。

- 开源Serving框架是什么。修改了什么。
  - 开源：https://github.com/EfficientMoE/MoE-Gen。
  - 未基于现有开源 serving 框架修改，而是自研 MoE-GEN Engine。对比的 serving 框架包括：
    - **FlexGen**：model-based batching，按轮次重用已加载的模型权重进行多次 forward pass，未针对 MoE expert sparsity 优化 batch size。
    - **DeepSpeed-Inference**：将 MoE layer 视为 dense MLP 处理，batch size 受 attention peak memory 限制。
    - **MoE-Lightning**：优化 GPU-CPU-I/O overlap 但保留 model-based batching。
    - **vLLM / Llama.cpp**：continuous batching，面向 interactive inference 的 TTFT 优化，解码阶段 batch 更小。
  - MoE-GEN 的核心修改：将 batching unit 从 model level 下沉到 module（attention/expert）level，累计 token 形成大 batch 后才在 GPU 执行。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？
  - **MoE-GEN Engine 执行流程**（以解码阶段为例）：
    1. **Batching Scheduler**：基于硬件/软件 profiling 数据在搜索空间枚举候选配置，对每个配置通过 DAG Constructor 估算 runtime，选择最短完成时间的配置。确定 $B, b_a, b_e, \omega, S_{Expert}, S_{Params}$。
    2. **Engine 初始化**：按配置在 GPU 上分配 KV-cache buffer、expert module buffer、dense module buffer。
    3. **Attention 阶段**：按照 attention micro-batch size $b_a$ 重复执行 attention 模块。对于每一批：
       - GPU 端：执行 Pre-Attention（QKV projection），同时 HtoD engine 预取下一批 attention weights 和对应 KV-cache。
       - CPU 端：按 split ratio $\omega$ 并行执行 self-attention mechanism（$QK^T$），CPU 可直接访问 host memory 中的 KV-cache，无需 HtoD 拷贝。
       - GPU 端：执行 self-attention mechanism（需要先完成 KV-cache HtoD copy）。
       - GPU 端：执行 Post-Attention（output projection）。
    4. **Expert 阶段**：将多个 attention 批次累计的 token 合并为大批次 $B$。由于大 batch 下 token 均匀分配到各 expert，顺序执行所有 experts：HtoD engine 预取下一个 expert weights（利用 PCIe idle 时间），GPU 执行当前 expert 计算。每次只加载一个 expert weights 到 $S_{Expert}$ buffer。
    5. **KV-cache 更新**：DtoH engine 将新生成的 KV-cache 异步写回 host memory。
    6. 以上步骤逐层迭代，直到所有 layers 完成。
