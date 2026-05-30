## MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-GEN 实现了 **CPU attention kernel** 用于卸载 self-attention mechanism（$QK^T$）计算到 CPU，核心包含：
    1. 基于 **AVX intrinsics** 的 Grouped Query Attention（GQA）实现，使用 BF16 数据格式。由于原生 BF16 硬件支持仅限较新高端 CPU，在 FP32 中表示 BF16 数据（显式清零低 16 位尾数），所有计算和累加在 FP32 精度，每次点积累加后按 BF16 舍入规则舍入并清零尾数位，保证与 PyTorch GPU attention 数值一致。
    2. CPU kernel 针对 cache 性能优化（类似 FlashAttention 的 CPU 版本思想），使 CPU 处理 GEMV（$QK^T$ 计算是 matrix-vector 乘法）的速率达到与 PCIe4.0 传输 KV-cache + GPU 计算的时间可比。
    3. 运行时调度：在 attention 阶段，按 $\omega$ 比例将 tokens 分配至 CPU 执行 self-attention，GPU 和 CPU 并行执行各自的 attention 计算，结果在 Post-Attention 前 concatenate。CPU kernel 直接访问 host memory 中的 KV-cache，无需 HtoD 拷贝，节省 PCIe 带宽给 expert weight 预取。
  - 实验比较：(1) MoE-GEN(G)（纯 GPU，$\omega=0$）vs MoE-GEN(H)（CPU attention 卸载，$\omega > 0$）的 decoding throughput；(2) 不同 $\omega$ 值（0-100%）对 throughput 的影响曲线；(3) 不同 CPU 计算能力（C1/C2: AMD 7453 28-Core vs C3: AMD 7313P 16-Core）下的最优 $\omega$ 选择；(4) 不同模型下的 CPU:GPU 最优 split ratio（Mixtral-8x7B: 6:4, Mixtral-8x22B: 7:3, DeepSeek-V2: 0:10）。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A5000 (24GB, PCIe 4.0), NVIDIA A6000 (48GB, PCIe 4.0)
  - CPU: AMD EPYC 7453 28-Core (C1/C2), AMD EPYC 7313P 16-Core (C3)
  - Host Memory: 256GB (C1), 512GB (C2), 480GB (C3)

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 MoE-GEN Engine（约 3000 行 C++ + 2000 行 Python）。benchmark 脚本位于开源仓库 https://github.com/EfficientMoE/MoE-Gen。
  - 修改：在 MoE-GEN Engine 的 attention 阶段插入 CPU kernel dispatch 路径，按 $\omega$ 比例将 self-attention 计算分流到 CPU。
  - 对比的 baseline 系统：vLLM、Llama.cpp、DeepSpeed-Inference、FlexGen*、MoE-Lightning* 均在 GPU 上执行所有 attention 计算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？
  - **评估原理**：通过 CUDA events 插桩每个 module 的 forward pass 测量计算延迟，通过 `torch.memory_stats` API 测量峰值内存（CUDA context + KV-cache + activations），通过 `cudaMemcpy` 计时测量 PCIe 带宽。离线 profiling 各模块在不同 batch size 和 sequence length 下的数据后，由 DAG-based scheduler 选择最优配置（包括 $\omega$）。
  - **从 kernel 输入到性能输出的过程**：
    1. Profiling 阶段：对 attention 模块在 GPU 和 CPU 上分别测量不同 $b_a$ 下的 latency（CUDA events / CPU timer），以及 KV-cache HtoD copy 时间。
    2. Search 阶段：scheduler 枚举 $\omega \in \{0, 0.1, ..., 1.0\}$，对每个配置估算 attention 阶段总时间 = $\max(\omega \cdot T_\text{CPU\_attn}, (1-\omega) \cdot (T_\text{KV\_copy} + T_\text{GPU\_attn}))$ + 其它固定开销。选择使 critical path 最短的 $\omega$。
    3. Runtime 阶段：Engine 在执行 attention 时，将 $\omega \times b_a$ 个 tokens dispatch 到 CPU kernel（使用 AVX BF16-GQA），$(1-\omega) \times b_a$ 个 tokens 走 GPU 路径。结果通过 concatenate 合并后进入 Post-Attention。
    4. 输出：throughput = $B / T_\text{DAG}$（tokens/s）。
