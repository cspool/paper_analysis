## Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pre-gated MoE 的 system 部分——基于 CPU offloading 的 MoE 推理系统，通过 pre-gate function 实现 preemptive expert migration。核心设计：(1) 分层存储策略：dense non-MoE 参数（attention weights, embeddings, layernorm）常驻 GPU memory；sparse MoE 参数（全部 expert weights）完全 offload 到 CPU DRAM；(2) Preemptive Expert Migration：利用第 N 个 block 的 pre-gate function 提前知道 (N+1) 个 block 需要哪些 experts，在 GPU 执行第 N 个 block 的 expert computation 期间，从 CPU 经 PCIe 异步迁移仅激活的 experts 到 GPU；(3) 通信-计算重叠：pre-gate function 是轻量 MLP（计算量极小），所以 expert migration 阶段（蓝色，PCIe communication-bound）可与 expert execution 阶段（绿色，compute-bound）完全并行；(4) GPU 峰值内存公式：Peak_GPU_mem = max(Non_MoE_M + Σ_{L=N}^{N+1} Act_Exp_L)，即非 MoE 参数 + 连续两个 block 的激活 expert 参数之和。
  - 实验比较：(1) 单 MoE block 延迟——Pre-gated MoE vs MoE-OnDemand (按需加载) vs MoE-Prefetch (全量预取) vs GPU-only (oracular 上界，全部参数在 GPU)；(2) 端到端推理吞吐 (tokens/sec)；(3) 峰值 GPU 内存使用——Pre-gated MoE 仅占 GPU-only 的 23%，与 memory-optimal 的 MoE-OnDemand 几乎相同；(4) 模型准确率 vs 原始 SwitchTransformer；(5) Sensitivity studies——pre-gate activation level (N=0/1/2/3)、激活 expert 数量 (1~64)、叠加 expert caching (LIFO/LFU/LRU)、SSD offloading 场景。
- 硬件平台是什么，配置是什么。
  - CPU: AMD EPYC 7V12 64-Core, 1.8TB DDR4 memory。GPU: 单卡 NVIDIA A100 80GB HBM。互联: PCIe Gen4, 32 GB/s 单向数据带宽。系统配置：CPU-GPU (MoE 参数在 CPU，non-MoE 参数在 GPU) vs GPU-only (全部参数在 GPU，oracular 上界)。
- 开源Serving框架是什么。修改了什么。
  - 基于 NVIDIA FasterTransformer（https://github.com/NVIDIA/FasterTransformer），state-of-the-art CUDA 推理库。修改包括：(1) 实现分层参数存储——non-MoE 参数常驻 GPU，expert 参数 offload 到 CPU；(2) 实现 preemptive expert migration pipeline——在 MoE block N 的 expert execution 期间，异步启动 CPU→GPU cudaMemcpy 传输 (N+1) block 的激活 experts；(3) 利用 CUDA stream 实现通信与计算的重叠——expert migration 在一个 stream，expert computation 在另一个 stream；(4) 实现 pre-gate function 的 forward 逻辑——在 FasterTransformer 的 MoE block 中插入 pre-gate linear layer，输出传递给下一个 block；(5) 修改第一个 MoE block 使用双 gate（传统 gate + pre-gate），最后一个 block 无 pre-gate。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/ranggihwang/Pregated_MoE, Zenodo DOI: 10.5281/zenodo.10976343。Docker 镜像: nvcr.io/nvidia/pytorch:22.09-py3。编译：cmake -DSM=80 -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON。
  - 框架输入到硬件执行全过程（Switch-Base 128 experts, batch=1, single A100）：
    
    **阶段 0 — 模型加载与初始化**：
    1. 从 HuggingFace 下载 SwitchTransformer pretrained weights，fine-tune 为 Pre-gated MoE（2,048 steps）。
    2. 加载模型：non-MoE 参数（attention, embedding, layernorm）→ GPU HBM；所有 expert 参数 → CPU DRAM（1.8TB 充足）。
    3. 第一个 MoE block 加载两个 gate function（传统 gate + pre-gate），其余 block 各一个 pre-gate，最后一个 block 无 pre-gate。
    
    **阶段 1 — 第一个 MoE Block 执行（例外，无法重叠）**：
    4. Input hidden states x_0 ∈ R^{B×H} 进入 MoE block 0。
    5. 传统 gate: logits = W_gate @ x_0 → softmax → TopK → 选择激活 experts 集合 A_0。
    6. On-demand migration: cudaMemcpy(A_0 的 expert weights, CPU→GPU) —— 串行暴露 PCIe 延迟。
    7. Expert execution: GPU 计算 Σ w_i · Expert_i(x_0) for i ∈ A_0。
    8. 同时 pre-gate: logits' = W_pre_gate @ x_0 → softmax → TopK → 确定 block 1 的激活 experts A_1。
    
    **阶段 2 — 后续 MoE Block 执行 (N ≥ 1，核心优化)**：
    9. Input x_N 进入 MoE block N。A_N 已由 block (N-1) 的 pre-gate 提前确定。
    10. Expert execution 立即开始: GPU SM 执行 Σ w_i · Expert_i(x_N) for i ∈ A_N（compute-bound，约 2ms）。
    11. 同时 pre-gate: logits' = W_pre_gate @ x_N → softmax → TopK → A_{N+1}。
    12. 同时 preemptive migration: cudaMemcpy(A_{N+1} 的 expert weights, CPU→GPU) 在独立 CUDA stream 上异步执行（communication-bound，约 1-2ms 取决于 expert 大小）。
    13. Step 10 与 Step 11-12 完全重叠——expert execution 的 compute 时间 ≥ expert migration 的 PCIe 时间。
    14. 循环回到 Step 9 处理下一个 block。
    
    **阶段 3 — 最后一个 MoE Block**：
    15. 最后一个 block 无 pre-gate function（无需为不存在下一个 block 选择 experts）。
    16. Expert execution 完成后，输出经 layernorm → LM head → next token prediction。
    
    **阶段 4 — 测量**：
    17. block_lats.csv: 每个 MoE block 的平均延迟。
    18. throughputs.csv: end-to-end tokens/sec。
    19. peak_mems.csv: 峰值 GPU 内存使用。
    
    Pre-gated MoE 的核心作用：通过 pre-gate function 解耦 expert selection 与 expert execution 的数据依赖，使 CPU→GPU expert migration 与 GPU expert computation 完全重叠。对比 MoE-OnDemand（串行暴露 PCIe 延迟）和 MoE-Prefetch（传输全部 experts 浪费带宽和 GPU 内存），Pre-gated MoE 同时实现了接近 GPU-only 的性能和接近 MoE-OnDemand 的内存效率。
