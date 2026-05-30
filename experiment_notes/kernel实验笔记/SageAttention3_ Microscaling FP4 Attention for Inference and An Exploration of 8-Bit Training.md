## SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - SageAttention3 的 FP4 attention 使用 CUTLASS + CUDA 实现 kernel，包含三项关键硬件优化：(1) **K 的 Permutation**：FP4 MMA 的 FP32 accumulator 内存布局与 operand A 寄存器布局不匹配，通过重排 accumulator 布局（permute P tile 的列），并对应重排 K 的列（fuse 到量化 kernel 中），避免 thread shuffle 开销。(2) **Reuse Shuffle**：P̃ 的 micro-scaling 量化需要在 16 个连续行元素上找 max，但这 16 个元素分布在 4 个 thread 中。将量化与 online softmax 融合，复用 S 的 16 元素 max 给量化使用，减少 50% shuffle 和 max 操作，整体 kernel 加速约 10%。(3) **Producer Warp Epilogue**：传统 warp-specialized kernel 由 consumer warp 同时处理 MatMul 和 store，producer 只加载。由于寄存器约束，改为 producer warp 间 ping-pong：一个 producer 加载下一轮输入时，另一个 producer 存上一轮输出到 global memory。Consumer warp 只负责将 MatMul 结果从寄存器搬到 shared memory。实现 MatMul 与 global memory store 的 overlap。
  - SageBwd 使用 OpenAI Triton 实现 INT8 前向+反向 attention kernel。
  - 实验比较：与 FlashAttention2（CUDA）、xformers、FlashAttention2 Triton 版本对比 kernel speed（TOPS）和延迟。SageAttention3 在 RTX5090 上达到 1038 TOPS，是 FlashAttention2 的 5×。SageBwd 前向 2× 加速（最高），反向 1.2~1.6× 加速（最高），端到端 forward+backward 最高 1.67× 加速。

- 后端平台是什么，配置是什么。
  - SageAttention3 kernel：NVIDIA RTX5090 (Blackwell, FP4 Tensor Core)
  - SageBwd kernel：NVIDIA RTX4090 (INT8 Tensor Core)
  - 对比 head_dim=64 和 head_dim=128 两种配置

- 评估性能的软件/脚本是什么。修改了什么。
  - SageAttention3：基于 CUTLASS [22] 和 CUDA 自研 kernel，在 FlashAttention tiling 框架上替换 MatMul 为 FP4MMA 指令
  - SageBwd：基于 OpenAI Triton [23] 实现，修改了 attention 前向+反向的量化策略
  - 对比 baselines：FlashAttention2 (CUDA)、xformers、SageAttention、SageAttention2

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/thu-ml/SageAttention
  - Kernel 执行流程：输入 FP16 Q, K, V 分块 → Quantization kernel（含 K transpose fuse 和 Smoothing）将 Q, K, V 量化为 NVFP4（E2M1 + E4M3 scale）→ FP4MMA 指令执行 QK^T → Online Softmax（含 two-level quantization for P，复用 rowmax 做 shuffle reduction）→ FP4MMA 指令执行 PV → Producer warp ping-pong store 输出 O。其中 permutation 优化在 K 量化阶段完成列重排；reuse shuffle 在 softmax 阶段将 max 值共享给 P 量化；producer warp epilogue 通过双 producer warp 交替完成 load 和 store。
