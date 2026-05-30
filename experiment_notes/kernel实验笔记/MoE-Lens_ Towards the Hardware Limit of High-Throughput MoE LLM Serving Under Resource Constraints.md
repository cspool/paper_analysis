## MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-Lens 实现了 **手工优化的 CPU Decode Attention kernel** 用于 CPU-GPU 混合 MoE 推理系统的 CPU 端 attention 计算：
    1. **AVX512 SIMD intrinsics 实现**：使用 hand-written SIMD intrinsics（manual vectorization）实现 decode attention，支持 BF16 KV cache 数据格式，upconvert 到 FP32 进行 dot product 和 saxpby 计算（Equation 6）。包含 loop unrolling 和 data prefetching 优化。
    2. **Cache-optimized memory access**：针对 CPU 端 KV cache 访问模式优化，通过 prefetch 指令减少 cache miss。
    3. **多线程 scaling**：支持 multi-threaded execution，实测在超过 20 threads 后吞吐增益饱和（memory controller contention）。
  - 实验比较：(1) MoE-Lens intrinsics kernel vs auto-vectorized baseline（均使用 AVX512 ISA）的 KV cache tokens attended per second；(2) 不同线程数（1-28 threads）下两者的吞吐对比和 scaling behavior；(3) 与 system throughput requirement（假设 KV cache size = 2× model size）的对比。

- 后端平台是什么，配置是什么。
  - CPU: Intel Platinum 8380（支持 AVX512），单 socket 使用（numactl 限制）。
  - GPU: NVIDIA A40 48GB（simulated to 16-24GB）。
  - Memory: DDR4-3200，单 socket 8 channels，~150GB/s measured aggregate bandwidth。

- 评估性能的软件/脚本是什么。修改了什么。
  - MoE-Lens 自身实现的 C++ CPU decode attention kernel（PyTorch extension）。论文未开源（arXiv: 2504.09345），无公开代码。
  - 对比 baseline：auto-vectorized 版本（依赖编译器自动向量化，同样使用 AVX512 ISA）。
  - 修改：手工替换编译器自动生成的向量化代码为 hand-tuned SIMD intrinsics 实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？
  - **MoE-Lens 未开源**。
  - **评估原理（基于论文描述）**：
    1. **Kernel 输入**：decode 序列的 query vectors（shape: $[n_q, d]$）和 KV cache blocks（存储在 pinned CPU memory，BF16 格式），以及 GQA group size s。
    2. **Kernel 计算**：对每个 query 在 KV cache 中做 attention——vector dot product（query·key）→ softmax → saxpby（attention weights × value）。手动向量化利用 AVX512 512-bit registers 一次处理 16 个 BF16 元素（converted to FP32 = 8 elements per register）。
    3. **优化策略**：loop unrolling 减少分支和循环开销；data prefetching 指令提前将下一轮 KV cache 数据加载到 CPU cache；FP32 累加保证数值精度；BF16→FP32 upconvert 和 FP32→BF16 rounding 每一步显式处理。
    4. **性能输出**：throughput = KV cache tokens attended per second。Intrinsics kernel 单线程 4.7× auto-vectorized、全线程 3.1× auto-vectorized。满足 system target（KV cache = 2× model size 时所需的 attention throughput）。
  - **运行时集成**：CPU attention kernel 在 VSLPipe 的 CPU Task (C) 阶段被调用，与 GPU Task B 的 GEMM 计算并行执行（图 9 pipeline）。CPU attention 结果通过 H2D transfer 回传 GPU 用于后续 O projection 和 MoE layer。
