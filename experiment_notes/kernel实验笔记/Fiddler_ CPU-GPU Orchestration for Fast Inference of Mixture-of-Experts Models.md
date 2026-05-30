## Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Fiddler 包含两个 kernel 调度/运行时计算层面的实现：
    1. **CPU AVX512_BF16 Expert 计算 kernel**：利用 Intel AVX512_BF16 指令集实现的自定义 CPU expert FFN 计算 kernel。PyTorch 原生不支持 BF16 的 AVX512 指令，Fiddler 手动实现以提升 CPU 端 expert 计算吞吐。
    2. **异构后端运行时调度（Algorithm 1）**：在 CPU 和 GPU 两种后端间动态调度 expert 计算。基于 latency model（GPU 延迟恒定，CPU 延迟随输入量线性增长）和输入 token 数量 s，在每个 MoE 层运行时决定每个 expert 的执行后端——GPU 直接执行、GPU+PCIe weight transfer 执行、或 CPU 执行（activation copy + CPU compute + output copy back）。
  - 实验比较：
    - 微基准：测量 weight copy (CPU→GPU)、activation copy (GPU→CPU)、GPU expert execution (不同 input size)、CPU expert execution (不同 input size) 的延迟
    - 宏基准：Fiddler vs DeepSpeed-MII vs Mixtral-Offloading vs llama.cpp 在单 batch 推理、长 prefill、beam search 三种场景

- 后端平台是什么，配置是什么。
  - GPU 后端：NVIDIA Quadro RTX 6000 (24GB) / RTX 6000 Ada (48GB)
  - CPU 后端：Intel Xeon Gold 6126 (48 cores, Env1) / Intel Xeon Platinum 8480+ (112 cores, Env2)
  - CPU-GPU 传输：PCIe Gen3 x16 (32GB/s, Env1) / PCIe Gen4 x16 (64GB/s, Env2)
  - CPU 指令集：AVX512_BF16（Intel Xeon Platinum 8480+ 支持 AMX/AVX512_BF16）

- 评估性能的软件/脚本是什么。修改了什么。
  - Fiddler 基于 PyTorch 构建，自建 microbenchmark 脚本测量各操作延迟：
    - Weight copy latency (CPU→GPU)：每个 expert weight ~300MB，测量 32 层平均和标准差
    - Activation copy latency (GPU→CPU)：测量 32 层平均和标准差
    - GPU expert execution latency：分别在 input size N=1,2,4,8,16,32,64 下测量
    - CPU expert execution latency：分别在 input size N=1,2,4,8,16,32,64 下测量
  - 修改内容：
    - **新增 CPU AVX512_BF16 kernel**：替代 PyTorch 默认 CPU GEMM，针对 expert FFN 的 (input×4096)×(4096×14336) 矩阵乘优化
    - **运行时调度逻辑**：在 PyTorch forward path 插入 `cpu_lat(s)` / `gpu_lat(s)` / `trans_lat()` 决策逻辑
    - **Latency model 校准**：初始化阶段运行微基准测量三个函数所需的常数参数

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：代码开源在 https://github.com/efeslab/fiddler
  - **CPU AVX512_BF16 Kernel 执行原理全过程**：

    ```
    ┌── Kernel Input ──────────────────────────────────────────┐
    │ activation: float32/bf16 tensor [s, 4096]  (GPU→CPU copy) │
    │ expert_weights: {                                         │
    │   W_gate: [4096, 14336]  // gate projection               │
    │   W_up:   [4096, 14336]  // up projection                 │
    │   W_down: [14336, 4096]  // down projection               │
    │ }  // 常驻 CPU pinned memory, 16-bit precision            │
    │ s: number of input tokens (1 for single-batch decode,     │
    │    up to thousands for prefill)                            │
    └──────────────────────────────────────────────────────────┘

    ┌── CPU AVX512_BF16 Expert FFN Kernel ──────────────────┐
    │ // 利用 AVX512_BF16 VDPBF16PS 指令 (每周期 32 个 BF16 MAC) │
    │ // PyTorch 默认使用 FP32 GEMM, 无法利用 BF16 硬件加速      │
    │                                                           │
    │ // Step 1: gate projection                                │
    │ gate_out = matmul_avx512_bf16(activation, W_gate)         │
    │ // [s, 4096] × [4096, 14336] → [s, 14336]                │
    │                                                           │
    │ // Step 2: up projection                                  │
    │ up_out = matmul_avx512_bf16(activation, W_up)             │
    │ // [s, 4096] × [4096, 14336] → [s, 14336]                │
    │                                                           │
    │ // Step 3: SiLU activation                                │
    │ gate_act = SiLU(gate_out)  // element-wise                │
    │ // SiLU(x) = x * sigmoid(x)                               │
    │                                                           │
    │ // Step 4: gated fusion                                   │
    │ fused = gate_act * up_out  // element-wise multiply       │
    │                                                           │
    │ // Step 5: down projection                                │
    │ output = matmul_avx512_bf16(fused, W_down)                │
    │ // [s, 14336] × [14336, 4096] → [s, 4096]                │
    │                                                           │
    │ // 关键：每个 matmul 内部使用 AVX512 tile 分块：            │
    │ // - 每次加载 32 个 BF16 元素到 ZMM 寄存器                  │
    │ // - VDPBF16PS 指令计算 32 个 BF16 点积                    │
    │ // - 累加结果到 FP32 accumulator                          │
    │ // - Tile 大小选择最小化 CPU cache miss                    │
    └──────────────────────────────────────────────────────────┘

    ┌── Runtime 调度决策 (Algorithm 1) ─────────────────────┐
    │ for each expert j in layer l:                             │
    │   s = inp_size[j]  // #tokens routed to expert j          │
    │   if s == 0: skip                                         │
    │   if is_at_gpu(l, j):                                     │
    │     // Strategy (a): 纯 GPU 执行                          │
    │     output = cuda_expert_ffn(activation_gpu, W_gpu)       │
    │   elif cpu_lat(s) > gpu_lat(s) + trans_lat():             │
    │     // Strategy (b): GPU+CPU→GPU weight transfer          │
    │     W_gpu = cudaMemcpyAsync(W_cpu → W_gpu, PCIe)         │
    │     output = cuda_expert_ffn(activation_gpu, W_gpu)       │
    │   else:                                                   │
    │     // Strategy (c): CPU execution                        │
    │     act_cpu = cudaMemcpyAsync(act_gpu → act_cpu, PCIe)   │
    │     output = avx512_bf16_expert_ffn(act_cpu, W_cpu)      │
    │     cudaMemcpyAsync(output_cpu → output_gpu, PCIe)        │
    └──────────────────────────────────────────────────────────┘
    ```

    **微基准数据（Figure 7, Appendix A）**：
    - Weight copy: GPU computation 的 2-5× 时间（主要开销）
    - GPU execution: 基本恒定于 batch size（Env1 batch=1 时因 PyTorch 单 batch 使用不同实现有约 10% 差异）
    - CPU execution: 随 input size 线性增长
    - Activation copy: <1% of CPU single-input latency（可忽略）
    - 建模简化：gpu_lat(s) = constant, cpu_lat(s) ∝ s, activation transfer latency ≈ 0

    **策略 (b) vs (c) 的 trade-off 分析**：
    | Input size s | Strategy (b) 延迟 | Strategy (c) 延迟 | 最优 |
    |-------------|-------------------|-------------------|------|
    | s=1 (decode) | trans_lat + gpu_const | cpu_const × 1 | (c) CPU |
    | s=256 (prefill) | trans_lat + gpu_const | cpu_const × 256 | (b) GPU+transfer |
    | 阈值 s_threshold | = trans_lat / cpu_slope | — | 切换点 |
