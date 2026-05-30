## Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ES-MoE 在 Fairseq 框架上实现 expert 参数 offload 到 CPU 内存/SSD，并通过三项核心运行时调度机制提升 MoE 训练效率：(1) **Pipelined Expert Processing**：expert 级别的流水线——在 token permutation 阶段上传第一个 expert（重叠 permutation 延迟），后续 experts 串行处理时并发上传与计算（expert I/O 与 expert FFN 重叠）；(2) **Dynamic Expert Placement**：基于 per-batch gating network 输出的 token 分布，使用贪心近似调度算法（Graham 1969, 4/3-approximation）动态将 n 个 experts 分配到 k 个 GPUs，使各 GPU 的聚合负载均衡，消除 zero-padding（复杂度 O(m*log n + m*log m)，CPU 执行 < 2.69μs）；(3) **Expert-wise CPU Optimization**：将 CPU Adam optimizer 从 layer-wise 改为 expert-wise 粒度——每个 expert 完成 backward pass 后立即启动 CPU 端参数更新，与后续 layers 的 GPU 计算重叠；(4) **Adaptive Offloading**：根据 expert 数量与 GPU 内存比值自动选择 GPU-only / CPU offload / CPU+SSD offload 模式，expert pinning 将 top 25% 热门 expert 固定在 GPU 上。
  - 实验比较：(a) 训练吞吐量对比（words/s）：ES-MoE vs Zero-Offload^E / FairSeq GShard / Tutel，覆盖 MoE-S/M/L 模型 × 8/16/32/64 experts（Table 1）；(b) 可扩展性：各框架最大支持 expert 数量（Figure 4）；(c) 微 batch size 对吞吐量影响（Figure 5）；(d) Component-wise 分析：pipelined expert processing 带来的 GPU 利用率提升（+61.1%）、dynamic expert placement 的 token 负载均衡效果（102% → 15% 差异）、adaptive offloading 模式切换（Figure 7）；(e) Ablation study：逐一去除 expert pinning / optimizer overlapping / larger batch size / zero-padding elimination 对吞吐量的影响（Table 3）；(f) Fine-tuning：Fairseq-MoE-15B 在 4 GPUs 上用 SST-2/MNLI/BoolQ 数据集微调 6.5 小时（Table 2）；(g) Pretraining 端到端对比（Table 5）；(h) GPU 利用率分析（Table 6）。

- 后端平台是什么，配置是什么。
  - GPU：4× NVIDIA A100 40GB（PCIe 4.0 for CPU-GPU, NVLink 600 GB/s for GPU-GPU）
  - CPU：AMD EPYC 7543 32-core
  - CPU Memory：512 GiB DDR4
  - Storage：SSD（实验中最高配 4 TB 用于 SSD offloading 模式）
  - Software：Fairseq framework（基于 PyTorch），DeepSpeed CPU Adam optimizer
  - 实现代码：3.3k lines Python + 3.0k lines C++

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 Fairseq 训练框架。修改内容：(1) 新增 expert offload 模块——将 expert 参数和 optimizer states 从 GPU 迁移到 CPU pinned memory 和 SSD；(2) 新增 pipelined expert scheduler——在 MoE block forward pass 中，token permutation 与首个 expert upload 重叠，后续 expert upload 与 FFN 计算重叠；(3) 新增 dynamic expert placement 模块——CPU 端贪心调度算法，per-batch 决定 expert→GPU 映射；(4) 新增 expert-wise CPU optimizer——override Fairseq 的 layer-wise optimizer，改为 per-expert 触发 CPU Adam step；(5) 新增 adaptive offloading 控制器——运行时决定使用 GPU-only / CPU-offload / CPU+SSD-offload 模式。
  - 评估脚本评估原理：每个 training iteration 测量 wall-clock time，计算 training throughput (words/s) = batch_size × sequence_length / iteration_time。GPU 利用率通过 PyTorch profiler 测量。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/kaist-ina/es-moe
  - 评估原理：在 Fairseq 框架中，每个 training iteration 的 MoE layer 执行流程如下：
    ```
    输入：Input tokens [B, S, H] 分布在 k 个 GPUs 上（Expert Parallelism）
    
    Forward Pass per MoE Block:
    1. Gating Network (GPU): x → Linear(W_g) → softmax → Top-1 expert index per token
    2. Dynamic Expert Placement (CPU, <2.69μs):
       - 收集 per-expert token counts from all GPUs
       - Greedy scheduling: sort experts by (upload_time + compute_time)
       - Assign each expert to GPU with minimum accumulated load
       - 输出：expert→GPU 映射表
    3. Token Permutation (GPU): All-to-All scatter tokens to target GPUs
       【同时：异步上传第一个 expert 权重 CPU→GPU via PCIe】
    4. Expert Processing Loop (per GPU):
       for expert in assigned_experts:
         a) 若 expert 不在 GPU: 异步上传 expert weights CPU→GPU
         b) Expert FFN: gate_proj(x) → SiLU ⊙ up_proj(x) → down_proj (与上传重叠)
         c) 输出 intermediate activations
    5. Token Un-permutation (GPU): All-to-All gather expert outputs back
    
    Backward Pass:
    6. Expert FFN backward (GPU): 计算 expert weight gradients
    7. Expert-wise CPU Optimizer:
       - 每个 expert backward 完成后立即触发:
         a) 下载 gradients GPU→CPU
         b) CPU Adam step: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²
            w = w - lr * m̂ / (√v̂ + ε)
       - Expert N 的 CPU optimizer 与 Expert N+1 的 GPU backward 重叠
    8. Non-Expert backward (GPU): Attention 等 dense 参数梯度计算（GPU optimizer）
    
    性能输出：iteration_time → throughput = tokens / iteration_time
    ```
  - 关键技术点：(1) 不使用 batched matrix multiplication——sequential expert processing 避免了 dispatch mask（节省 >48 GiB for MoE-L batch 32），允许 8× 更大的 microbatch；(2) Expert 从 CPU memory 到 SSD 的 eviction 使用 LRU cache policy + prefetching（基于 forward/backward 可预测序列），使用 DMA-able pinned memory 避免 naïve VM page fault stall。
